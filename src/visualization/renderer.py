"""
雷达回波伪彩色渲染器 - 将预测张量转化为 PNG 图像流
"""
import io
from datetime import datetime
from typing import Optional, List
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .colormap import WeatherColorMap, get_colorbar_labels


class RadarImageRenderer:
    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        vis_cfg = self.config.get("visualization", {})
        self.dpi = vis_cfg.get("dpi", 100)
        self.format = vis_cfg.get("format", "png")
        self.colormap = WeatherColorMap()

    def render_single(
        self,
        data: np.ndarray,
        title: str = None,
        timestamp: datetime = None,
        radar_id: str = None,
        add_colorbar: bool = True,
    ) -> bytes:
        rgb = self.colormap(data)
        img = Image.fromarray(rgb, mode="RGB")

        if add_colorbar:
            img = self._add_colorbar(img)

        if title or timestamp or radar_id:
            img = self._add_overlay(img, title, timestamp, radar_id)

        buf = io.BytesIO()
        img.save(buf, format=self.format.upper(), dpi=(self.dpi, self.dpi))
        return buf.getvalue()

    def render_sequence(
        self,
        sequence: np.ndarray,
        start_time: datetime = None,
        interval_minutes: int = 10,
        radar_id: str = None,
    ) -> List[bytes]:
        images = []
        for i, frame in enumerate(sequence):
            title = None
            ts = None
            if start_time is not None:
                from datetime import timedelta
                ts = start_time + timedelta(minutes=i * interval_minutes)
                title = f"T+{i * interval_minutes}min"

            img_bytes = self.render_single(
                frame,
                title=title,
                timestamp=ts,
                radar_id=radar_id,
                add_colorbar=(i == len(sequence) - 1),
            )
            images.append(img_bytes)
        return images

    def render_comparison(
        self,
        observed: np.ndarray,
        predicted: np.ndarray,
        timestamp: datetime = None,
    ) -> bytes:
        obs_rgb = self.colormap(observed)
        pred_rgb = self.colormap(predicted)

        obs_img = Image.fromarray(obs_rgb, mode="RGB")
        pred_img = Image.fromarray(pred_rgb, mode="RGB")

        h, w = observed.shape
        gap = 20
        canvas_w = w * 2 + gap * 3
        canvas_h = h + 100
        canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))

        canvas.paste(obs_img, (gap, 60))
        canvas.paste(pred_img, (w + gap * 2, 60))

        draw = ImageDraw.Draw(canvas)
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except Exception:
            font = ImageFont.load_default()

        draw.text((gap + w // 2 - 40, 20), "观测值", fill=(0, 0, 0), font=font)
        draw.text((w + gap * 2 + w // 2 - 40, 20), "预测值", fill=(0, 0, 0), font=font)

        if timestamp:
            draw.text((gap, 5), timestamp.strftime("%Y-%m-%d %H:%M"), fill=(0, 0, 0), font=font)

        buf = io.BytesIO()
        canvas.save(buf, format=self.format.upper())
        return buf.getvalue()

    def render_gif(
        self,
        sequence: np.ndarray,
        start_time: datetime = None,
        interval_minutes: int = 10,
        duration_ms: int = 500,
    ) -> bytes:
        frames = []
        for i, frame in enumerate(sequence):
            rgb = self.colormap(frame)
            img = Image.fromarray(rgb, mode="RGB")

            draw = ImageDraw.Draw(img)
            try:
                font = ImageFont.truetype("arial.ttf", 16)
            except Exception:
                font = ImageFont.load_default()

            label = f"T+{i * interval_minutes}min"
            if start_time:
                from datetime import timedelta
                ts = start_time + timedelta(minutes=i * interval_minutes)
                label = f"{ts.strftime('%H:%M')} ({label})"
            draw.text((10, 10), label, fill=(255, 255, 255), font=font)

            frames.append(img)

        buf = io.BytesIO()
        frames[0].save(
            buf,
            format="GIF",
            save_all=True,
            append_images=frames[1:],
            duration=duration_ms,
            loop=0,
        )
        return buf.getvalue()

    def _add_colorbar(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        bar_w = 40
        gap = 15

        canvas = Image.new("RGB", (w + bar_w + gap * 2, h), (255, 255, 255))
        canvas.paste(img, (gap, 0))

        bar_h = h - 40
        bar_top = 20

        labels = get_colorbar_labels()
        n_labels = len(labels)

        for i in range(bar_h):
            value = labels[-1][0] - (i / bar_h) * (labels[-1][0] - labels[0][0])
            rgb = self.colormap(np.array([value]))[0]
            for x in range(bar_w):
                canvas.putpixel((w + gap + x, bar_top + i), tuple(rgb))

        draw = ImageDraw.Draw(canvas)
        try:
            font = ImageFont.truetype("arial.ttf", 10)
        except Exception:
            font = ImageFont.load_default()

        for i, (val, label) in enumerate(labels):
            y = bar_top + int((1 - i / (n_labels - 1)) * bar_h)
            draw.line([(w + gap - 5, y), (w + gap, y)], fill=(0, 0, 0), width=1)
            draw.text((w + gap + bar_w + 5, y - 5), label, fill=(0, 0, 0), font=font)

        return canvas

    def _add_overlay(
        self,
        img: Image.Image,
        title: str = None,
        timestamp: datetime = None,
        radar_id: str = None,
    ) -> Image.Image:
        overlay_h = 40
        w, h = img.size
        canvas = Image.new("RGB", (w, h + overlay_h), (255, 255, 255))
        canvas.paste(img, (0, overlay_h))

        draw = ImageDraw.Draw(canvas)
        try:
            font = ImageFont.truetype("arial.ttf", 14)
        except Exception:
            font = ImageFont.load_default()

        x_offset = 10
        if radar_id:
            draw.text((x_offset, 10), f"雷达: {radar_id}", fill=(0, 0, 0), font=font)
            x_offset += 150

        if timestamp:
            draw.text((x_offset, 10), timestamp.strftime("%Y-%m-%d %H:%M:%S"), fill=(0, 0, 0), font=font)
            x_offset += 200

        if title:
            draw.text((x_offset, 10), title, fill=(0, 0, 0), font=font)

        return canvas

"""
FastAPI 路由端点 - 雷达数据上传、预测、预览接口

重构要点:
  1. 所有推理请求通过 batch_scheduler.submit() 异步提交
  2. 输入张量先经 standardizer 标准化再提交，消除尺寸差异
  3. 推理结果通过 await future 获取，天然支持背压
  4. GPU 显存管理由 scheduler 内部自动处理
"""
import asyncio
import os
from datetime import datetime, timedelta
from typing import List

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Query
from fastapi.responses import Response, StreamingResponse

from .schemas import (
    RadarUploadResponse,
    SatelliteUploadResponse,
    PredictionResponse,
    PredictionFrameInfo,
    HealthResponse,
    RadarInfoResponse,
)
from .deps import get_app_state, AppState

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check(state: AppState = Depends(get_app_state)):
    scheduler_stats = state.batch_scheduler.get_stats() if state.batch_scheduler else {}
    return HealthResponse(
        status="ok",
        model_loaded=state.is_model_loaded,
        device=state.device_str,
        version="1.0.0",
    )


@router.get("/stats")
async def inference_stats(state: AppState = Depends(get_app_state)):
    if state.batch_scheduler is None:
        raise HTTPException(status_code=503, detail="Scheduler not initialized")
    return state.batch_scheduler.get_stats()


@router.post("/upload", response_model=RadarUploadResponse)
async def upload_radar_data(
    files: List[UploadFile] = File(...),
    state: AppState = Depends(get_app_state),
):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    bytes_list = []
    for file in files:
        content = await file.read()
        bytes_list.append((file.filename, content))

    frames = state.dataloader.load_from_bytes_list(bytes_list)

    if not frames:
        raise HTTPException(status_code=400, detail="Failed to parse any radar data")

    return RadarUploadResponse(
        success=True,
        message=f"Successfully loaded {len(frames)} radar frames",
        frame_count=len(frames),
        radar_id=frames[0]["radar_id"],
        start_time=frames[0]["timestamp"],
        end_time=frames[-1]["timestamp"],
    )


@router.post("/upload_satellite", response_model=SatelliteUploadResponse)
async def upload_satellite_data(
    files: List[UploadFile] = File(...),
    state: AppState = Depends(get_app_state),
):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    if state.fy4_parser is None or state.fy4_preprocessor is None:
        raise HTTPException(status_code=503, detail="Satellite processing not initialized")

    bytes_list = []
    for file in files:
        content = await file.read()
        bytes_list.append((file.filename, content))

    satellite_frames = []
    for filename, content in bytes_list:
        try:
            data = state.fy4_parser.parse_bytes(content, filename)
            satellite_frames.append(data)
        except Exception as e:
            continue

    if not satellite_frames:
        raise HTTPException(status_code=400, detail="Failed to parse any satellite data")

    preprocessed = state.fy4_preprocessor.preprocess_sequence(satellite_frames)

    return SatelliteUploadResponse(
        success=True,
        message=f"Successfully loaded {len(satellite_frames)} satellite frames",
        frame_count=len(satellite_frames),
        satellite_id=satellite_frames[0].get("satellite_id", "FY-4"),
        start_time=satellite_frames[0].get("timestamp"),
        end_time=satellite_frames[-1].get("timestamp"),
        has_attention_mask=preprocessed.get("attention_mask") is not None,
    )


@router.post("/predict", response_model=PredictionResponse)
async def predict(
    radar_files: List[UploadFile] = File(...),
    satellite_files: List[UploadFile] = File(None),
    state: AppState = Depends(get_app_state),
):
    if not radar_files:
        raise HTTPException(status_code=400, detail="No radar input files")

    radar_bytes_list = []
    for file in radar_files:
        content = await file.read()
        radar_bytes_list.append((file.filename, content))

    frames = state.dataloader.load_from_bytes_list(radar_bytes_list)

    if not frames:
        raise HTTPException(status_code=400, detail="Failed to parse radar data")

    input_tensor = state.tensor_builder.build_input_tensor(frames)
    if input_tensor is None:
        raise HTTPException(
            status_code=400,
            detail=f"Insufficient frames. Need at least {state.tensor_builder.input_seq_len}",
        )

    satellite_tensor = None
    hard_mask_tensor = None

    if satellite_files and state.use_satellite:
        satellite_bytes_list = []
        for file in satellite_files:
            content = await file.read()
            satellite_bytes_list.append((file.filename, content))

        satellite_frames = []
        for filename, content in satellite_bytes_list:
            try:
                data = state.fy4_parser.parse_bytes(content, filename)
                satellite_frames.append(data)
            except Exception:
                continue

        if satellite_frames and state.fy4_preprocessor is not None:
            preprocessed = state.fy4_preprocessor.preprocess_for_fusion(
                satellite_frames,
                target_seq_len=state.tensor_builder.input_seq_len,
                reference_times=[f["timestamp"] for f in frames],
            )
            satellite_tensor = preprocessed.get("satellite_tensor")
            hard_mask_tensor = preprocessed.get("hard_mask_tensor")

    try:
        output_tensor = await state.batch_scheduler.submit(
            input_tensor,
            satellite_tensor=satellite_tensor,
            hard_mask_tensor=hard_mask_tensor,
            meta={"radar_id": frames[0]["radar_id"]},
        )
    except RuntimeError as e:
        error_msg = str(e)
        if "OOM" in error_msg or "out of memory" in error_msg.lower() or "memory pressure" in error_msg.lower():
            raise HTTPException(status_code=503, detail=error_msg)
        raise HTTPException(status_code=500, detail=error_msg)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Inference request timed out")

    output_seq_len = state.tensor_builder.output_seq_len
    interval = state.config.get("prediction", {}).get("interval_minutes", 10)
    start_time = frames[-1]["timestamp"] + timedelta(minutes=interval)

    frame_infos = []
    for i in range(output_seq_len):
        ts = start_time + timedelta(minutes=i * interval)
        frame_infos.append(
            PredictionFrameInfo(
                index=i,
                forecast_minutes=(i + 1) * interval,
                timestamp=ts,
            )
        )

    end_time = start_time + timedelta(minutes=(output_seq_len - 1) * interval)

    fusion_info = ""
    if satellite_tensor is not None:
        fusion_info = " with satellite fusion"

    return PredictionResponse(
        success=True,
        message=f"Prediction completed{fusion_info}",
        input_frame_count=len(frames),
        output_frame_count=output_seq_len,
        start_time=start_time,
        end_time=end_time,
        interval_minutes=interval,
        frames=frame_infos,
    )


@router.get("/preview")
async def preview_single(
    file: str = Query(..., description="Path to radar binary file"),
    var_name: str = Query("Z", description="Variable to visualize: Z, ZDR, etc."),
    state: AppState = Depends(get_app_state),
):
    if not os.path.exists(file):
        raise HTTPException(status_code=404, detail="File not found")

    frame = state.dataloader.load_single_file(file)
    if frame is None or var_name not in frame["data"]:
        raise HTTPException(status_code=400, detail="Failed to load or variable not found")

    data = frame["data"][var_name]
    img_bytes = state.renderer.render_single(
        data,
        timestamp=frame["timestamp"],
        radar_id=frame["radar_id"],
        title=f"{var_name} - 观测",
    )

    return Response(content=img_bytes, media_type="image/png")


@router.get("/preview/sequence")
async def preview_prediction_sequence(
    files: List[str] = Query(..., description="List of input radar file paths"),
    satellite_files: List[str] = Query(None, description="List of input satellite file paths"),
    format: str = Query("png", description="Output format: png, gif, zip"),
    state: AppState = Depends(get_app_state),
):
    for f in files:
        if not os.path.exists(f):
            raise HTTPException(status_code=404, detail=f"File not found: {f}")

    frames = state.dataloader.load_from_files(files)
    if not frames:
        raise HTTPException(status_code=400, detail="Failed to load any frames")

    input_tensor = state.tensor_builder.build_input_tensor(frames)
    if input_tensor is None:
        raise HTTPException(status_code=400, detail="Insufficient input frames")

    satellite_tensor = None
    hard_mask_tensor = None

    if satellite_files and state.use_satellite:
        for f in satellite_files:
            if not os.path.exists(f):
                raise HTTPException(status_code=404, detail=f"Satellite file not found: {f}")

        satellite_frames = []
        for f in satellite_files:
            try:
                data = state.fy4_parser.parse_file(f)
                satellite_frames.append(data)
            except Exception:
                continue

        if satellite_frames and state.fy4_preprocessor is not None:
            preprocessed = state.fy4_preprocessor.preprocess_for_fusion(
                satellite_frames,
                target_seq_len=state.tensor_builder.input_seq_len,
                reference_times=[f["timestamp"] for f in frames],
            )
            satellite_tensor = preprocessed.get("satellite_tensor")
            hard_mask_tensor = preprocessed.get("hard_mask_tensor")

    try:
        output_tensor = await state.batch_scheduler.submit(
            input_tensor,
            satellite_tensor=satellite_tensor,
            hard_mask_tensor=hard_mask_tensor,
            meta={"radar_id": frames[0]["radar_id"]},
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    denormalized = state.tensor_builder.denormalize_output(output_tensor, var_name="Z")

    interval = state.config.get("prediction", {}).get("interval_minutes", 10)
    start_time = frames[-1]["timestamp"] + timedelta(minutes=interval)

    if format.lower() == "gif":
        gif_bytes = state.renderer.render_gif(
            denormalized,
            start_time=start_time,
            interval_minutes=interval,
        )
        return Response(content=gif_bytes, media_type="image/gif")

    images = state.renderer.render_sequence(
        denormalized,
        start_time=start_time,
        interval_minutes=interval,
        radar_id=frames[0]["radar_id"],
    )

    def image_generator():
        for i, img in enumerate(images):
            boundary = f"--frame-{i}\r\n".encode()
            yield boundary
            yield b"Content-Type: image/png\r\n\r\n"
            yield img
            yield b"\r\n"
        yield b"--end\r\n"

    return StreamingResponse(
        image_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@router.get("/preview/frame/{frame_idx}")
async def preview_prediction_frame(
    frame_idx: int,
    files: List[str] = Query(..., description="List of input radar file paths"),
    satellite_files: List[str] = Query(None, description="List of input satellite file paths"),
    state: AppState = Depends(get_app_state),
):
    for f in files:
        if not os.path.exists(f):
            raise HTTPException(status_code=404, detail=f"File not found: {f}")

    frames = state.dataloader.load_from_files(files)
    if not frames:
        raise HTTPException(status_code=400, detail="Failed to load frames")

    input_tensor = state.tensor_builder.build_input_tensor(frames)
    if input_tensor is None:
        raise HTTPException(status_code=400, detail="Insufficient input frames")

    satellite_tensor = None
    hard_mask_tensor = None

    if satellite_files and state.use_satellite:
        for f in satellite_files:
            if not os.path.exists(f):
                raise HTTPException(status_code=404, detail=f"Satellite file not found: {f}")

        satellite_frames = []
        for f in satellite_files:
            try:
                data = state.fy4_parser.parse_file(f)
                satellite_frames.append(data)
            except Exception:
                continue

        if satellite_frames and state.fy4_preprocessor is not None:
            preprocessed = state.fy4_preprocessor.preprocess_for_fusion(
                satellite_frames,
                target_seq_len=state.tensor_builder.input_seq_len,
                reference_times=[f["timestamp"] for f in frames],
            )
            satellite_tensor = preprocessed.get("satellite_tensor")
            hard_mask_tensor = preprocessed.get("hard_mask_tensor")

    try:
        output_tensor = await state.batch_scheduler.submit(
            input_tensor,
            satellite_tensor=satellite_tensor,
            hard_mask_tensor=hard_mask_tensor,
            meta={"radar_id": frames[0]["radar_id"]},
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    denormalized = state.tensor_builder.denormalize_output(output_tensor, var_name="Z")

    if frame_idx < 0 or frame_idx >= len(denormalized):
        raise HTTPException(status_code=400, detail=f"Frame index out of range (0-{len(denormalized)-1})")

    interval = state.config.get("prediction", {}).get("interval_minutes", 10)
    start_time = frames[-1]["timestamp"] + timedelta(minutes=interval)
    ts = start_time + timedelta(minutes=frame_idx * interval)

    img_bytes = state.renderer.render_single(
        denormalized[frame_idx],
        timestamp=ts,
        radar_id=frames[0]["radar_id"],
        title=f"T+{(frame_idx + 1) * interval}min - 预测",
    )

    return Response(content=img_bytes, media_type="image/png")


@router.get("/info", response_model=RadarInfoResponse)
async def get_radar_info(
    file: str = Query(..., description="Path to radar binary file"),
    state: AppState = Depends(get_app_state),
):
    if not os.path.exists(file):
        raise HTTPException(status_code=404, detail="File not found")

    try:
        volume = state.parser.parse_file(file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Parse error: {str(e)}")

    return RadarInfoResponse(
        radar_id=volume.radar_id,
        radar_name=volume.radar_name,
        latitude=volume.latitude,
        longitude=volume.longitude,
        altitude=volume.altitude,
        scan_time=volume.scan_time,
        variables=volume.list_variables(),
    )

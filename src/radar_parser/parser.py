"""
国家气象局双偏振雷达基数据二进制解析器

支持解析 CINRAD 系列雷达的基数据格式，包括：
- 反射率因子 Z
- 差分反射率 ZDR
- 差分传播相移 PHIDP
- 相关系数 RHOHV
"""
import struct
import os
from typing import BinaryIO, Optional
from datetime import datetime, timedelta
import numpy as np

from .models import RadarVolume, SweepData, RadialData


MISSING_VALUE = -9999.0
SPEED_OF_LIGHT = 2.99792458e8


class RadarBinaryParser:
    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        self.radial_count = self.config.get("radial_count", 360)
        self.gate_count = self.config.get("gate_count", 920)

    def parse_file(self, file_path: str) -> RadarVolume:
        with open(file_path, "rb") as f:
            return self._parse_stream(f, os.path.basename(file_path))

    def parse_bytes(self, data: bytes, filename: str = "radar_data") -> RadarVolume:
        import io
        with io.BytesIO(data) as f:
            return self._parse_stream(f, filename)

    def _parse_stream(self, f: BinaryIO, filename: str) -> RadarVolume:
        try:
            return self._parse_cinrad_format(f, filename)
        except Exception:
            return self._parse_simple_format(f, filename)

    def _parse_cinrad_format(self, f: BinaryIO, filename: str) -> RadarVolume:
        header = f.read(32)
        if len(header) < 32:
            raise ValueError("File too small for CINRAD format")

        magic = struct.unpack_from("4s", header, 0)[0]
        if magic not in (b"CINR", b"RADR", b"RDEC"):
            raise ValueError(f"Unknown magic: {magic}")

        version = struct.unpack_from("4s", header, 4)[0].decode("ascii", errors="ignore").strip("\x00")
        radar_id = struct.unpack_from("8s", header, 8)[0].decode("ascii", errors="ignore").strip("\x00")
        radar_name = struct.unpack_from("16s", header, 16)[0].decode("gbk", errors="ignore").strip("\x00")

        site_info = f.read(64)
        latitude = struct.unpack_from("f", site_info, 0)[0]
        longitude = struct.unpack_from("f", site_info, 4)[0]
        altitude = struct.unpack_from("f", site_info, 8)[0]

        scan_time_raw = struct.unpack_from("I", site_info, 12)[0]
        scan_time = self._convert_cinrad_time(scan_time_raw)

        volume = RadarVolume(
            radar_id=radar_id,
            radar_name=radar_name,
            latitude=latitude,
            longitude=longitude,
            altitude=altitude,
            scan_time=scan_time,
        )

        sweep_count = struct.unpack_from("I", site_info, 16)[0]
        if sweep_count <= 0 or sweep_count > 20:
            sweep_count = self.config.get("elevation_count", 9)

        for sweep_idx in range(sweep_count):
            sweep_header = f.read(32)
            if len(sweep_header) < 32:
                break

            sweep_num = struct.unpack_from("I", sweep_header, 0)[0]
            elev_angle = struct.unpack_from("f", sweep_header, 4)[0]
            radial_count = struct.unpack_from("I", sweep_header, 8)[0]

            sweep = SweepData(sweep_number=sweep_num, elevation_angle=elev_angle)

            for _ in range(radial_count):
                radial = self._parse_radial_cinrad(f)
                if radial is not None:
                    sweep.radials.append(radial)

            if sweep.radials:
                volume.sweeps.append(sweep)

        return volume

    def _parse_radial_cinrad(self, f: BinaryIO) -> Optional[RadialData]:
        header = f.read(32)
        if len(header) < 32:
            return None

        azimuth = struct.unpack_from("f", header, 0)[0]
        elevation = struct.unpack_from("f", header, 4)[0]
        time_raw = struct.unpack_from("I", header, 8)[0]
        gate_count = struct.unpack_from("I", header, 12)[0]
        var_count = struct.unpack_from("I", header, 16)[0]

        if gate_count <= 0 or gate_count > 2000:
            gate_count = self.gate_count

        radial_time = self._convert_cinrad_time(time_raw)
        radial = RadialData(azimuth=azimuth, elevation=elevation, time=radial_time)

        for _ in range(var_count):
            var_header = f.read(8)
            if len(var_header) < 8:
                break

            var_code = struct.unpack_from("H", var_header, 0)[0]
            scale = struct.unpack_from("H", var_header, 2)[0]
            offset = struct.unpack_from("f", var_header, 4)[0]

            raw_data = f.read(gate_count * 2)
            if len(raw_data) < gate_count * 2:
                break

            raw_values = np.frombuffer(raw_data, dtype=np.uint16).astype(np.float32)
            values = np.where(raw_values == 0, MISSING_VALUE, (raw_values - offset) / scale)

            var_name = self._var_code_to_name(var_code)
            if var_name:
                radial.variables[var_name] = values

        return radial

    def _parse_simple_format(self, f: BinaryIO, filename: str) -> RadarVolume:
        import hashlib
        raw = f.read()
        radar_id = hashlib.md5(filename.encode()).hexdigest()[:8]

        volume = RadarVolume(
            radar_id=radar_id,
            radar_name=filename,
            latitude=30.0,
            longitude=114.0,
            altitude=50.0,
            scan_time=datetime.now(),
        )

        expected_size = self.radial_count * self.gate_count * 4
        if len(raw) >= expected_size * 2:
            z_raw = raw[: self.radial_count * self.gate_count * 2]
            zdr_raw = raw[self.radial_count * self.gate_count * 2 : expected_size * 2]

            z_data = np.frombuffer(z_raw, dtype=np.int16).reshape(self.radial_count, self.gate_count).astype(np.float32) / 10.0
            zdr_data = np.frombuffer(zdr_raw, dtype=np.int16).reshape(self.radial_count, self.gate_count).astype(np.float32) / 100.0

            z_data = np.where(z_data < -30, MISSING_VALUE, z_data)
            zdr_data = np.where(zdr_data < -2, MISSING_VALUE, zdr_data)

            sweep = SweepData(sweep_number=0, elevation_angle=0.5)
            for i in range(self.radial_count):
                radial = RadialData(
                    azimuth=i * (360.0 / self.radial_count),
                    elevation=0.5,
                    time=volume.scan_time,
                    variables={"Z": z_data[i], "ZDR": zdr_data[i]},
                )
                sweep.radials.append(radial)
            volume.sweeps.append(sweep)

        return volume

    @staticmethod
    def _var_code_to_name(code: int) -> Optional[str]:
        mapping = {
            1: "Z",
            2: "ZDR",
            3: "PHIDP",
            4: "RHOHV",
            5: "VEL",
            6: "SW",
        }
        return mapping.get(code)

    @staticmethod
    def _convert_cinrad_time(raw: int) -> datetime:
        if raw <= 0:
            return datetime.now()
        try:
            if raw > 1e10:
                return datetime(1970, 1, 1) + timedelta(seconds=raw / 1000.0)
            return datetime(1970, 1, 1) + timedelta(seconds=raw)
        except Exception:
            return datetime.now()

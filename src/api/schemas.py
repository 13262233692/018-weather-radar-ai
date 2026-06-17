"""
API 数据模型定义
"""
from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field


class RadarUploadResponse(BaseModel):
    success: bool
    message: str
    frame_count: int = 0
    radar_id: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None


class PredictionFrameInfo(BaseModel):
    index: int
    forecast_minutes: int
    timestamp: datetime


class PredictionResponse(BaseModel):
    success: bool
    message: str
    input_frame_count: int
    output_frame_count: int
    start_time: datetime
    end_time: datetime
    interval_minutes: int
    frames: List[PredictionFrameInfo]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    version: str


class RadarInfoResponse(BaseModel):
    radar_id: str
    radar_name: str
    latitude: float
    longitude: float
    altitude: float
    scan_time: datetime
    variables: List[str]


class SatelliteUploadResponse(BaseModel):
    success: bool
    message: str
    frame_count: int = 0
    satellite_id: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    has_attention_mask: bool = False

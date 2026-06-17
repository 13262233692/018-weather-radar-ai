"""
FastAPI 路由端点 - 雷达数据上传、预测、预览接口
"""
import os
import tempfile
from datetime import datetime, timedelta
from typing import List, Optional

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Query
from fastapi.responses import Response, StreamingResponse

from .schemas import (
    RadarUploadResponse,
    PredictionResponse,
    PredictionFrameInfo,
    HealthResponse,
    RadarInfoResponse,
)
from .deps import get_app_state, AppState

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check(state: AppState = Depends(get_app_state)):
    return HealthResponse(
        status="ok",
        model_loaded=state.is_model_loaded,
        device=state.device,
        version="1.0.0",
    )


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


@router.post("/predict", response_model=PredictionResponse)
async def predict(
    files: List[UploadFile] = File(...),
    state: AppState = Depends(get_app_state),
):
    if not files:
        raise HTTPException(status_code=400, detail="No input files")

    bytes_list = []
    for file in files:
        content = await file.read()
        bytes_list.append((file.filename, content))

    frames = state.dataloader.load_from_bytes_list(bytes_list)

    if not frames:
        raise HTTPException(status_code=400, detail="Failed to parse radar data")

    input_tensor = state.tensor_builder.build_input_tensor(frames)
    if input_tensor is None:
        raise HTTPException(
            status_code=400,
            detail=f"Insufficient frames. Need at least {state.tensor_builder.input_seq_len}",
        )

    output_tensor = state.predictor.predict(input_tensor)

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

    return PredictionResponse(
        success=True,
        message="Prediction completed",
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

    output_tensor = state.predictor.predict(input_tensor)
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

    output_tensor = state.predictor.predict(input_tensor)
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

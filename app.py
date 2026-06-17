"""
FastAPI 应用主入口
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router as radar_router
from src.api.deps import get_app_state
from src.utils.config import load_config


def create_app(config_path: str = None) -> FastAPI:
    config = load_config(config_path)
    service_cfg = config.get("service", {})

    app = FastAPI(
        title="短临天气推演平台",
        description="基于雷达回波与 ConvLSTM 的短临天气预测系统",
        version="1.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.on_event("startup")
    async def startup_event():
        state = get_app_state()
        state.initialize()

    app.include_router(
        radar_router,
        prefix=service_cfg.get("api_prefix", "/api/radar"),
        tags=["radar"],
    )

    @app.get("/")
    async def root():
        return {
            "name": "短临天气推演平台",
            "version": "1.0.0",
            "endpoints": {
                "health": service_cfg.get("api_prefix", "/api/radar") + "/health",
                "upload": service_cfg.get("api_prefix", "/api/radar") + "/upload",
                "predict": service_cfg.get("api_prefix", "/api/radar") + "/predict",
                "preview": service_cfg.get("api_prefix", "/api/radar") + "/preview",
            },
        }

    return app


app = create_app()


def main():
    import uvicorn

    config = load_config()
    service_cfg = config.get("service", {})

    uvicorn.run(
        "app:app",
        host=service_cfg.get("host", "0.0.0.0"),
        port=service_cfg.get("port", 8000),
        reload=True,
        workers=1,
    )


if __name__ == "__main__":
    main()

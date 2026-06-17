"""
FastAPI 应用主入口

重构要点:
  1. 使用 lifespan 替代 on_event("startup") 管理生命周期
  2. 启动时初始化 DynamicBatchScheduler 的异步 Worker
  3. 关闭时优雅停止调度器，排空队列中剩余请求
  4. 单 Worker 模式运行，所有 GPU 推理通过队列序列化
"""
import sys
import os
import logging
from contextlib import asynccontextmanager

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router as radar_router
from src.api.deps import get_app_state
from src.utils.config import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    state = get_app_state()
    state.initialize()
    logger.info("AppState initialized")

    await state.start_scheduler()
    logger.info("BatchScheduler started")

    yield

    logger.info("Shutting down BatchScheduler...")
    await state.stop_scheduler()
    logger.info("BatchScheduler stopped")


def create_app(config_path: str = None) -> FastAPI:
    config = load_config(config_path)
    service_cfg = config.get("service", {})

    app = FastAPI(
        title="短临天气推演平台",
        description="基于雷达回波与 ConvLSTM 的短临天气预测系统",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

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
                "stats": service_cfg.get("api_prefix", "/api/radar") + "/stats",
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
        reload=False,
        workers=1,
        loop="uvloop",
        log_level="info",
    )


if __name__ == "__main__":
    main()

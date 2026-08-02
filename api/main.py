"""
FastAPI 应用入口 —— CORS 配置、路由注册、启动事件。
启动方式：uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
"""

import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.schemas import HealthResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI 数字分身 — RAG 问答 API",
    description="基于 RAG 的个人 AI 数字分身系统。支持混合检索、多 LLM Provider 切换、来源透明的智能问答。",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    from src.config import settings
    logger.info("🚀 AI 数字分身 API 启动中...")
    logger.info("   数据目录: %s", settings.DATA_RAW_PATH)
    logger.info("   向量库: %s", settings.VECTOR_DB_PATH)
    logger.info("   LLM Provider: %s", settings.ACTIVE_PROVIDER)

    bm25_path = settings.VECTOR_DB_PATH / "bm25_index.pkl"
    if not bm25_path.exists():
        logger.warning("⚠️ 索引尚未构建！请先运行: python -m src.build_index")
    else:
        logger.info("✅ 索引文件已就绪")


from api.routes.chat import router as chat_router
from api.routes.documents import router as documents_router
from api.routes.eval import router as eval_router

app.include_router(chat_router)
app.include_router(documents_router)
app.include_router(eval_router)


@app.get("/api/health", response_model=HealthResponse, summary="健康检查", tags=["system"])
async def health_check() -> HealthResponse:
    import json
    from src.config import settings

    doc_count = 0
    meta_path = settings.VECTOR_DB_PATH / "chunks_meta.json"
    if meta_path.exists():
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                doc_count = len(json.load(f))
        except Exception:
            pass

    return HealthResponse(
        status="ok",
        model_loaded=settings.VECTOR_DB_PATH.exists(),
        doc_count=doc_count,
        llm_provider=settings.ACTIVE_PROVIDER,
    )

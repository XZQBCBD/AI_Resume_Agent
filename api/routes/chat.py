"""对话路由 —— POST /api/chat"""

import logging
from fastapi import APIRouter, HTTPException
from api.schemas import ChatRequest, ChatResponse, SourceItem

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["chat"])

_engine = None


def get_engine():
    global _engine
    if _engine is None:
        from src.chat_engine import RAGChatEngine
        _engine = RAGChatEngine()
    return _engine


@router.post("/chat", response_model=ChatResponse, summary="RAG 问答")
async def chat(request: ChatRequest) -> ChatResponse:
    """基于个人知识库的检索增强生成问答。"""
    try:
        engine = get_engine()
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"RAG 引擎初始化失败: {str(e)}。请先运行 python -m src.build_index 构建索引。",
        )

    try:
        result = engine.chat(question=request.question)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.exception("对话处理异常")
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {str(e)}")

    return ChatResponse(
        answer=result["answer"],
        sources=[
            SourceItem(content=s["content"], source=s["source"], file_type=s["file_type"], score=s["score"])
            for s in result["sources"]
        ],
        used_files=result["used_files"],
    )

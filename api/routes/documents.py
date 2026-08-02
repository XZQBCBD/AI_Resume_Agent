"""文档管理路由 —— GET /api/documents + POST /api/reindex"""

import json
import logging
from collections import Counter
from fastapi import APIRouter, HTTPException
from api.schemas import DocumentsResponse, DocumentItem, ReindexResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["documents"])


@router.get("/documents", response_model=DocumentsResponse, summary="已索引文档列表")
async def list_documents() -> DocumentsResponse:
    """获取已索引文档清单。"""
    from src.config import settings

    meta_path = settings.VECTOR_DB_PATH / "chunks_meta.json"
    if not meta_path.exists():
        return DocumentsResponse(documents=[], total=0)

    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"读取索引元数据失败: {str(e)}")

    file_stats = Counter()
    file_types = {}
    for c in chunks:
        source = c["metadata"].get("source", "unknown")
        file_stats[source] += 1
        if source not in file_types:
            file_types[source] = c["metadata"].get("file_type", "其他文档")

    documents = [
        DocumentItem(name=name, type=file_types.get(name, "其他文档"), chunks=count)
        for name, count in file_stats.items()
    ]
    return DocumentsResponse(documents=documents, total=len(documents))


@router.post("/reindex", response_model=ReindexResponse, summary="重建索引")
async def reindex() -> ReindexResponse:
    """重新扫描文档目录，清空现有索引并全量重建。"""
    from src.build_index import IndexBuilder

    try:
        builder = IndexBuilder()
        builder.build(full_rebuild=True)

        meta_path = builder.settings.VECTOR_DB_PATH / "chunks_meta.json"
        doc_count = 0
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                doc_count = len(json.load(f))

        return ReindexResponse(status="done", doc_count=doc_count)
    except Exception as e:
        logger.exception("重建索引失败")
        raise HTTPException(status_code=500, detail=f"重建索引失败: {str(e)}")

"""评估路由 —— GET /api/eval/benchmark + GET /api/eval/index"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional

router = APIRouter(prefix="/api/eval", tags=["evaluation"])


class BenchmarkResponse(BaseModel):
    """基准测试响应。"""
    reports: List[dict]
    summary: dict


class IndexStatsResponse(BaseModel):
    """索引统计响应。"""
    chunk_count: int
    file_count: int
    embedding_dim: int
    avg_chunk_length: int
    max_chunk_length: int
    min_chunk_length: int
    files: List[dict]


@router.get("/benchmark", response_model=BenchmarkResponse, summary="RAG 基准测试")
async def run_benchmark():
    """对预设问题集运行检索基准测试。"""
    from src.evaluator import RAGEvaluator

    evaluator = RAGEvaluator()
    reports = evaluator.benchmark()

    summary = {
        "total_questions": len(reports),
        "avg_results": sum(r.retrieval.total_results for r in reports) / max(len(reports), 1),
        "avg_latency_ms": sum(r.retrieval.latency_ms for r in reports) / max(len(reports), 1),
    }

    return BenchmarkResponse(
        reports=[evaluator.to_dict(r) for r in reports],
        summary=summary,
    )


@router.get("/index", response_model=IndexStatsResponse, summary="索引质量统计")
async def index_stats():
    """获取索引质量统计信息。"""
    from src.evaluator import RAGEvaluator

    evaluator = RAGEvaluator()
    stats = evaluator.evaluate_index()

    if stats["chunk_count"] == 0:
        raise HTTPException(status_code=503, detail="索引尚未构建")

    return IndexStatsResponse(
        chunk_count=stats["chunk_count"],
        file_count=stats["file_count"],
        embedding_dim=stats["embedding_dim"],
        avg_chunk_length=stats["avg_chunk_length"],
        max_chunk_length=stats["max_chunk_length"],
        min_chunk_length=stats["min_chunk_length"],
        files=stats["files"],
    )

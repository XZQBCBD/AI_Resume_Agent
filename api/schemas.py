"""
Pydantic 数据模型 —— API 请求和响应的类型定义。
"""

from typing import List
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """对话请求。"""
    question: str = Field(..., min_length=1, max_length=2000, description="用户提问内容")


class SourceItem(BaseModel):
    """引用来源项。"""
    content: str = Field(..., description="原文片段")
    source: str = Field(..., description="来源文件名")
    file_type: str = Field(..., description="文档类型")
    score: float = Field(..., description="检索相关度分数")


class ChatResponse(BaseModel):
    """对话响应。"""
    answer: str = Field(..., description="AI 回答文本")
    sources: List[SourceItem] = Field(default_factory=list, description="引用来源列表")
    used_files: List[str] = Field(default_factory=list, description="使用的文件列表")


class DocumentItem(BaseModel):
    """文档项。"""
    name: str = Field(..., description="文件名")
    type: str = Field(..., description="文档类型")
    chunks: int = Field(..., description="文本块数量")


class DocumentsResponse(BaseModel):
    """文档列表响应。"""
    documents: List[DocumentItem] = Field(default_factory=list)
    total: int = Field(0, description="文档总数")


class ReindexResponse(BaseModel):
    """重建索引响应。"""
    status: str = Field(..., description="执行状态")
    doc_count: int = Field(0, description="索引的文档块数量")


class HealthResponse(BaseModel):
    """健康检查响应。"""
    status: str = Field("ok")
    model_loaded: bool = Field(True)
    doc_count: int = Field(0)
    llm_provider: str = Field("deepseek")


class ErrorResponse(BaseModel):
    """错误响应。"""
    detail: str = Field(..., description="错误描述")

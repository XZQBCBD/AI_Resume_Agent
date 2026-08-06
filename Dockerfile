# ============================================================
# AI 数字分身 RAG 系统 - Docker 镜像
# ============================================================
FROM python:3.10-slim

LABEL maintainer="xiezuoqian"
LABEL description="AI Resume Agent - RAG-based personal AI digital twin"

# 设置工作目录
WORKDIR /app

# 安装系统依赖（OCR 可选）
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件并安装
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    -i https://pypi.tuna.tsinghua.edu.cn/simple

# 预下载 embedding 模型（使用国内镜像，避免运行时下载）
# 注意：BAAI/bge-small-zh-v1.5 中文优化，512 维，效果优于 all-MiniLM-L6-v2
ENV HF_ENDPOINT=https://hf-mirror.com
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-zh-v1.5')"

# 复制项目代码
COPY . .

# 创建非 root 用户
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# 暴露端口
EXPOSE 8000 8501

# 默认启动脚本
COPY docker-entrypoint.sh /app/docker-entrypoint.sh
USER root
RUN chmod +x /app/docker-entrypoint.sh
USER appuser

ENTRYPOINT ["/app/docker-entrypoint.sh"]

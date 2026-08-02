#!/bin/bash
# ============================================================
# Docker 容器启动脚本
# 1. 检查/构建向量索引
# 2. 启动 FastAPI 后端
# 3. 启动 Streamlit 前端
# ============================================================
set -e

echo "=========================================="
echo "  AI 数字分身 RAG 系统 - 容器启动"
echo "=========================================="

# 检查索引是否存在
if [ ! -f "${VECTOR_DB_PATH:-/app/vector_db}/bm25_index.pkl" ]; then
    echo "🔨 索引不存在，开始构建..."
    python -m src.build_index --sync
    echo "✅ 索引构建完成"
else
    echo "✅ 索引文件已就绪"
fi

# 根据 SERVICE 环境变量决定启动哪个服务
SERVICE="${SERVICE:-all}"

if [ "$SERVICE" = "api" ]; then
    echo "🚀 启动 FastAPI 后端 (0.0.0.0:8000)..."
    exec python run_api.py --host 0.0.0.0 --port 8000 --no-reload
elif [ "$SERVICE" = "app" ]; then
    echo "🚀 启动 Streamlit 前端 (0.0.0.0:8501)..."
    exec python -m streamlit run app/streamlit_app.py \
        --server.port 8501 \
        --server.address 0.0.0.0 \
        --server.headless true \
        --server.enableCORS false \
        --server.enableXsrfProtection false
else
    # 同时启动两个服务（使用后台进程）
    echo "🚀 启动 FastAPI 后端 (0.0.0.0:8000)..."
    python run_api.py --host 0.0.0.0 --port 8000 --no-reload &
    API_PID=$!

    echo "🚀 启动 Streamlit 前端 (0.0.0.0:8501)..."
    python -m streamlit run app/streamlit_app.py \
        --server.port 8501 \
        --server.address 0.0.0.0 \
        --server.headless true \
        --server.enableCORS false \
        --server.enableXsrfProtection false &
    APP_PID=$!

    # 等待任意进程退出
    echo "✅ 所有服务已启动"
    echo "   API:  http://0.0.0.0:8000"
    echo "   前端: http://0.0.0.0:8501"
    wait -n $API_PID $APP_PID
fi

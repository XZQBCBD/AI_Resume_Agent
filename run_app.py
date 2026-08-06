"""前端 Streamlit 启动脚本"""
import os
import sys
import subprocess
from pathlib import Path


def main():
    project_root = Path(__file__).parent
    app_path = project_root / "app" / "streamlit_app.py"

    if not app_path.exists():
        print(f"❌ 找不到应用入口: {app_path}")
        sys.exit(1)

    server_port = os.getenv("STREAMLIT_PORT", "8501")
    server_address = os.getenv("STREAMLIT_ADDRESS", "localhost")

    print("🚀 启动 AI 数字分身 Web 界面...")
    print(f"   访问地址: http://{server_address}:{server_port}")
    api_url = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
    print(f"   后端 API: {api_url}\n")

    subprocess.run([
        sys.executable, "-m", "streamlit", "run", str(app_path),
        "--server.port", server_port,
        "--server.address", server_address,
    ])


if __name__ == "__main__":
    main()

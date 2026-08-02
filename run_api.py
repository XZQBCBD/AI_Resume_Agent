"""后端 API 启动脚本"""
import sys
import argparse


def main():
    parser = argparse.ArgumentParser(description="AI 数字分身 — API 服务")
    parser.add_argument("--host", default="127.0.0.1", help="绑定地址")
    parser.add_argument("--port", type=int, default=8000, help="绑定端口")
    parser.add_argument("--reload", action="store_true", default=True, help="开发模式热重载")
    parser.add_argument("--no-reload", action="store_false", dest="reload", help="生产模式，禁用热重载")
    args = parser.parse_args()

    import uvicorn
    uvicorn.run("api.main:app", host=args.host, port=args.port, reload=args.reload, log_level="info")


if __name__ == "__main__":
    main()

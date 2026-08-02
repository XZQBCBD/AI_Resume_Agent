"""
Streamlit 对话展示组件 —— 消息列表、引用来源折叠、异常处理。
"""

import os
import requests
import streamlit as st

# API 地址可通过环境变量配置，默认本地开发地址
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")


def render_chat_area():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pending_question" not in st.session_state:
        st.session_state.pending_question = ""

    # ==================== 历史消息展示 ====================
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            # 助手消息：展示引用来源
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander("📄 查看参考原文", expanded=False):
                    for i, src in enumerate(msg["sources"], 1):
                        badge_class = _source_badge_class(src.get("file_type", ""))
                        st.markdown(
                            f'**{i}. {src["source"]}** '
                            f'<span class="doc-type-badge {badge_class}" style="font-size:0.62rem;">{src["file_type"]}</span> '
                            f'*(相关度: {src.get("score", 0):.2f})*',
                            unsafe_allow_html=True,
                        )
                        st.text(src["content"][:500])
                        if i < len(msg["sources"]):
                            st.divider()

    # ==================== 预设问题处理 ====================
    if st.session_state.pending_question:
        question = st.session_state.pending_question
        st.session_state.pending_question = ""
        _handle_user_input(question)

    # ==================== 用户输入 ====================
    if user_input := st.chat_input('输入你的问题，例如："请做自我介绍"...'):
        _handle_user_input(user_input)


def _handle_user_input(question: str):
    """处理用户输入：发送到后端 API 并更新消息列表。"""
    st.session_state.messages.append({"role": "user", "content": question})

    api_url = f"{API_BASE_URL}/api/chat"
    try:
        with st.spinner("🤔 检索知识库并生成回答..."):
            resp = requests.post(api_url, json={"question": question}, timeout=90)

        if resp.status_code == 200:
            data = resp.json()
            st.session_state.messages.append({
                "role": "assistant",
                "content": data["answer"],
                "sources": data.get("sources", []),
                "used_files": data.get("used_files", []),
            })
        elif resp.status_code == 422:
            detail = resp.json().get("detail", "输入无效")
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"⚠️ 输入校验失败：{detail}",
                "sources": [],
            })
        elif resp.status_code == 503:
            st.session_state.messages.append({
                "role": "assistant",
                "content": "🔧 索引尚未构建。请先运行 `python -m src.build_index` 构建知识库索引。",
                "sources": [],
            })
        else:
            error_detail = resp.json().get("detail", resp.text)
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"❌ 服务异常（{resp.status_code}）：{error_detail}",
                "sources": [],
            })
    except requests.exceptions.ConnectionError:
        st.session_state.messages.append({
            "role": "assistant",
            "content": (
                "⚠️ 无法连接到后端 API 服务。\n\n"
                "**排查步骤：**\n"
                "1. 确认后端已启动：`python run_api.py`\n"
                "2. 确认端口 8000 未被占用\n"
                "3. 确认防火墙未拦截本地连接"
            ),
            "sources": [],
        })
    except requests.exceptions.Timeout:
        st.session_state.messages.append({
            "role": "assistant",
            "content": "⏱️ 请求超时（90s），LLM 响应时间过长，请稍后重试或切换更快的 Provider。",
            "sources": [],
        })
    except Exception as e:
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"❌ 请求异常：{str(e)}",
            "sources": [],
        })

    st.rerun()


def _source_badge_class(file_type: str) -> str:
    """根据文档类型返回对应的 CSS badge class。"""
    mapping = {
        "简历":  "badge-resume",
        "项目":  "badge-project",
        "博客":  "badge-blog",
        "推荐信": "badge-letter",
    }
    return mapping.get(file_type, "badge-other")

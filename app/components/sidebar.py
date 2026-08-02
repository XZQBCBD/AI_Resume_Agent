"""
Streamlit 侧边栏组件 —— LLM 切换、预设问题。
"""

import streamlit as st


def render_sidebar():
    from src.prompt_loader import PromptLoader
    from src.config import settings

    prompt_loader = PromptLoader()
    ui_config = prompt_loader.get_ui_config()

    with st.sidebar:
        st.title("🤖 AI 数字分身")
        st.caption(ui_config.get("app_slogan", "基于 RAG 的个人知识问答系统"))
        st.divider()

        # ==================== LLM Provider 切换 ====================
        st.subheader("模型设置")
        providers = settings.get_available_providers()
        current = settings.ACTIVE_PROVIDER
        selected = st.selectbox(
            "LLM Provider",
            options=providers,
            index=providers.index(current) if current in providers else 0,
            help="切换大语言模型服务商（需在 .env 中配置对应的 API Key）",
            label_visibility="collapsed",
        )
        if selected != current:
            settings.ACTIVE_PROVIDER = selected
            st.rerun()

        st.divider()

        # ==================== 预设问题 ====================
        st.subheader("快速提问")
        for i, q in enumerate(prompt_loader.get_preset_questions()):
            if st.button(q, use_container_width=True, key=f"preset_{i}"):
                st.session_state.pending_question = q
                st.rerun()

        st.divider()

        # ==================== 底部信息 ====================
        st.markdown(
            f'<div class="sidebar-footer">'
            f'<a href="{ui_config.get("github_url", "#")}" target="_blank">🔗 GitHub</a>'
            f'<br>Powered by RAG + Streamlit'
            f'</div>',
            unsafe_allow_html=True,
        )

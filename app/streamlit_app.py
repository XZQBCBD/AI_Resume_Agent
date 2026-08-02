"""
Streamlit 主页面 —— AI 数字分身 Web 界面入口。
启动方式：streamlit run app/streamlit_app.py
"""

import streamlit as st

st.set_page_config(
    page_title="面试助手",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ============================================================
# 全局 CSS 样式 —— 基于 Data-Dense Dashboard 设计系统
# 主色: #1E3A5F (Navy) / 辅色: #2563EB (Blue) / 强调: #059669 (Green)
# ============================================================
st.markdown("""
<style>
    /* ===== 全局基础 ===== */
    .stApp {
        background: linear-gradient(180deg, #F8FAFC 0%, #F1F5F9 100%);
    }

    /* ===== 侧边栏 ===== */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1E3A5F 0%, #0F2440 100%);
        border-right: 1px solid rgba(255,255,255,0.06);
    }
    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] .stCaption,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stSelectbox label {
        color: #E2E8F0 !important;
    }
    [data-testid="stSidebar"] h1 { color: #FFFFFF !important; font-size: 1.35rem !important; font-weight: 700 !important; letter-spacing: -0.01em; }
    [data-testid="stSidebar"] h2 { color: #93C5FD !important; font-size: 0.95rem !important; font-weight: 600 !important; text-transform: uppercase; letter-spacing: 0.06em; }
    [data-testid="stSidebar"] h3 { color: #CBD5E1 !important; font-size: 0.8rem !important; font-weight: 500 !important; }
    [data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.12); margin: 1rem 0; }
    [data-testid="stSidebar"] .stButton > button {
        background: rgba(255,255,255,0.08) !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
        color: #E2E8F0 !important;
        border-radius: 8px !important;
        font-size: 0.82rem !important;
        padding: 0.5rem 0.75rem !important;
        transition: all 0.2s ease !important;
        text-align: left !important;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        background: rgba(37, 99, 235, 0.25) !important;
        border-color: #2563EB !important;
        color: #FFFFFF !important;
        transform: translateX(2px);
    }
    [data-testid="stSidebar"] .stSelectbox > div > div {
        background: rgba(255,255,255,0.08) !important;
        border-color: rgba(255,255,255,0.15) !important;
        color: #E2E8F0 !important;
        border-radius: 6px !important;
    }
    [data-testid="stSidebar"] a {
        color: #93C5FD !important;
        text-decoration: none !important;
    }
    [data-testid="stSidebar"] a:hover { color: #BFDBFE !important; }

    /* ===== 侧边栏文档卡片 ===== */
    .doc-card {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 8px;
        padding: 0.5rem 0.7rem;
        margin-bottom: 0.35rem;
        font-size: 0.78rem;
        transition: all 0.2s ease;
        display: flex;
        align-items: center;
        gap: 0.45rem;
    }
    .doc-card:hover {
        background: rgba(37, 99, 235, 0.15);
        border-color: rgba(37, 99, 235, 0.35);
    }
    .doc-type-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.65rem;
        font-weight: 600;
        letter-spacing: 0.03em;
        text-transform: uppercase;
        flex-shrink: 0;
    }
    .badge-resume  { background: rgba(5, 150, 105, 0.25); color: #6EE7B7; }
    .badge-project { background: rgba(37, 99, 235, 0.25); color: #93C5FD; }
    .badge-blog    { background: rgba(217, 119, 6, 0.25);  color: #FCD34D; }
    .badge-letter  { background: rgba(147, 51, 234, 0.25); color: #C4B5FD; }
    .badge-other   { background: rgba(100, 116, 139, 0.25);color: #CBD5E1; }

    .sidebar-footer {
        font-size: 0.7rem;
        color: rgba(255,255,255,0.35) !important;
        text-align: center;
        margin-top: 0.75rem;
    }

    /* ===== 主内容区 ===== */
    .main-header {
        text-align: center;
        padding: 0.8rem 0 0.2rem 0;
    }
    .main-header h1 {
        font-size: 1.6rem;
        font-weight: 700;
        color: #1E3A5F;
        margin-bottom: 0.15rem;
    }
    .main-header p {
        color: #64748B;
        font-size: 0.9rem;
    }

    /* ===== 聊天消息优化 ===== */
    [data-testid="stChatMessage"] {
        border-radius: 12px !important;
        padding: 0.8rem 1rem !important;
        margin-bottom: 0.6rem !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06) !important;
    }
    [data-testid="stChatMessage"][data-testid="stChatMessage-role-user"] {
        background: #EFF6FF !important;
        border: 1px solid #BFDBFE !important;
    }
    [data-testid="stChatMessage"][data-testid="stChatMessage-role-assistant"] {
        background: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
    }

    /* ===== 引用来源折叠面板 ===== */
    .stExpander {
        background: #F8FAFC !important;
        border-radius: 8px !important;
        border: 1px solid #E2E8F0 !important;
        margin-top: 0.5rem !important;
    }
    .stExpander > div:first-child {
        font-size: 0.82rem !important;
        color: #475569 !important;
        font-weight: 500 !important;
    }

    /* ===== 聊天输入框 ===== */
    [data-testid="stChatInput"] textarea {
        border-radius: 12px !important;
        border: 2px solid #E2E8F0 !important;
        transition: border-color 0.2s ease !important;
    }
    [data-testid="stChatInput"] textarea:focus {
        border-color: #2563EB !important;
        box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1) !important;
    }

    /* ===== 滚动条美化 ===== */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: rgba(0,0,0,0.12); border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(0,0,0,0.22); }

    /* ===== 响应式 ===== */
    @media (max-width: 768px) {
        .main-header h1 { font-size: 1.3rem; }
    }
</style>
""", unsafe_allow_html=True)

from app.components.sidebar import render_sidebar
from app.components.chat import render_chat_area


def main():
    render_sidebar()
    # 主区域顶部标题
    st.markdown("""
    <div class="main-header">
        <h1>🤖面试助手</h1>
        <p>基于 RAG 检索增强生成的个人知识问答系统</p>
    </div>
    """, unsafe_allow_html=True)
    st.divider()
    render_chat_area()


if __name__ == "__main__":
    main()

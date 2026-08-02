"""
RAG 总控引擎 —— 检索 → 增强 → 生成。

职责：
    1. 接收用户问题
    2. 调用 HybridRetriever.search() 检索相关知识
    3. 调用 PromptLoader.build_system_prompt() 组装 Prompt
    4. 调用 LLM（OpenAI 兼容 API）生成回答
    5. 解析返回结果，附带引用来源

使用方式：
    from src.chat_engine import RAGChatEngine
    engine = RAGChatEngine()
    result = engine.chat("你最擅长什么技术？")
"""

import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class RAGChatEngine:
    """RAG 总控引擎 —— 编排检索、增强和生成。"""

    def __init__(self):
        from src.config import settings
        from src.retriever import HybridRetriever
        from src.prompt_loader import PromptLoader

        self.settings = settings
        self.retriever = HybridRetriever()
        self.prompt_loader = PromptLoader()
        self._openai_client = None

        logger.info("✅ RAG 引擎初始化完成 | LLM: %s", settings.ACTIVE_PROVIDER)

    @property
    def openai_client(self):
        """懒加载 OpenAI 兼容客户端。"""
        if self._openai_client is None:
            from openai import OpenAI
            llm_config = self.settings.get_llm_config()
            self._openai_client = OpenAI(
                base_url=llm_config["base_url"],
                api_key=llm_config["api_key"],
            )
            logger.info("🔗 LLM 客户端已连接: %s", llm_config["base_url"])
        return self._openai_client

    def chat(self, question: str, history: Optional[List[Dict]] = None) -> Dict:
        """执行一轮 RAG 对话。

        完整管道：检索 → 拼接上下文 → 填充 Prompt → 调用 LLM → 格式化返回

        Args:
            question: 用户问题
            history: 对话历史（预留）

        Returns:
            {"answer": "...", "sources": [...], "used_files": [...]}
        """
        if not question or not question.strip():
            raise ValueError("问题不能为空，请输入有效的问题。")

        question = question.strip()
        logger.info("💬 用户问题: %s", question[:80])

        # 1. 检索（Retrieve）
        retrieved = self.retriever.search(question)
        if not retrieved:
            logger.warning("⚠️ 未检索到相关文档")
            return {
                "answer": "抱歉，我的知识库中暂时没有与您问题相关的信息。请尝试换个问法，或者联系我补充相关资料。",
                "sources": [],
                "used_files": [],
            }

        # 2. 增强（Augment）—— 拼接上下文
        context = "\n\n---\n\n".join([
            f"[来源: {r['source']} ({r['file_type']})]\n{r['content']}"
            for r in retrieved
        ])

        # 3. 生成（Generate）—— 填充 Prompt + 调用 LLM
        system_prompt = self.prompt_loader.build_system_prompt(
            context=context, question=question,
        )

        llm_config = self.settings.get_llm_config()

        try:
            response = self.openai_client.chat.completions.create(
                model=llm_config["model"],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question},
                ],
                temperature=self.settings.TEMPERATURE,
                timeout=self.settings.LLM_TIMEOUT,
            )
            answer = response.choices[0].message.content
            logger.info("✅ LLM 回答长度: %d 字符", len(answer))
        except Exception as e:
            logger.error("❌ LLM 调用失败: %s", e)
            raise RuntimeError(
                f"LLM ({self.settings.ACTIVE_PROVIDER}) 调用失败: {str(e)}\n"
                f"请检查 API Key 是否正确配置，网络是否可达。"
            ) from e

        # 4. 提取引用来源
        sources = [
            {
                "content": r["content"],
                "source": r["source"],
                "file_type": r["file_type"],
                "score": r["score"],
            }
            for r in retrieved
        ]
        used_files = list(dict.fromkeys(r["source"] for r in retrieved))

        return {"answer": answer, "sources": sources, "used_files": used_files}

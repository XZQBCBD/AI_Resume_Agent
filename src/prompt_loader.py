"""
Prompt 模板加载器 —— 从 TOML 文件读取模板并填充占位符。

职责：
    1. 解析 prompts/system_prompt.toml
    2. 提取 [system] 和 [template] 配置
    3. 使用 str.format() 填充 {role} {rules} {context} {question}

使用方式：
    from src.prompt_loader import PromptLoader
    loader = PromptLoader()
    prompt = loader.build_system_prompt(context="...", question="...")
"""

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class PromptLoader:
    """TOML Prompt 模板加载器。

    从 prompts/system_prompt.toml 读取角色定义、行为规则和对话模板，
    运行时填充上下文和用户问题，输出完整的 System Prompt。

    Attributes:
        config: TOML 解析后的原始字典
        role: 角色定义文本
        rules: 行为规则文本
        template: System Prompt 模板字符串
        ui_config: UI 相关配置（标题、Slogan、预设问题等）
    """

    def __init__(self, toml_path: Optional[Path] = None):
        """初始化 Prompt 加载器。

        Args:
            toml_path: TOML 文件路径，默认使用 config.settings.PROMPT_FILE_PATH
        """
        import toml

        if toml_path is None:
            from src.config import settings
            toml_path = settings.PROMPT_FILE_PATH

        if not toml_path.exists():
            raise FileNotFoundError(
                f"Prompt 模板文件不存在: {toml_path}\n"
                f"请确保 prompts/system_prompt.toml 已正确创建。"
            )

        self.toml_path = toml_path
        self.config = toml.load(toml_path)

        # 解析 [system] 段
        system = self.config.get("system", {})
        self.role = system.get("role", "")
        self.rules = system.get("rules", "")

        # 解析 [template] 段
        template = self.config.get("template", {})
        self.template = template.get("system_prompt", "")

        # 解析 [ui] 段（供 Streamlit 前端使用）
        self.ui_config = self.config.get("ui", {})

        logger.info("✅ Prompt 模板已加载: %s", toml_path)

    def build_system_prompt(self, context: str, question: str) -> str:
        """构建完整的 System Prompt —— 填充上下文和用户问题。

        Args:
            context: 检索到的文档片段（已拼接为字符串）
            question: 用户原始问题

        Returns:
            填充完成的 System Prompt 字符串
        """
        prompt = self.template.format(
            role=self.role,
            rules=self.rules,
            context=context,
            question=question,
        )
        return prompt

    def get_preset_questions(self) -> list:
        """获取 UI 预设问题列表。"""
        return self.ui_config.get("preset_questions", [])

    def get_ui_config(self) -> dict:
        """获取完整的 UI 配置（标题、Slogan、GitHub 链接等）。"""
        return self.ui_config

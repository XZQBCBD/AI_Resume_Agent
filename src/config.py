"""
集中配置管理模块。

职责：
    1. 加载 .env 环境变量
    2. 劫持 HuggingFace/Transformers 缓存路径到非系统盘
    3. 集中管理所有路径、检索参数、LLM Provider 配置
    4. 自动创建数据目录和向量库目录

使用方式：
    from src.config import settings
    print(settings.DATA_RAW_PATH)
"""

import os
import logging
from pathlib import Path
from dataclasses import dataclass, field

from dotenv import load_dotenv

# 在模块加载时自动读取 .env 文件
load_dotenv()

# 自动设置 HuggingFace 镜像（国内用户）
if os.getenv("HF_ENDPOINT"):
    os.environ["HF_ENDPOINT"] = os.getenv("HF_ENDPOINT")

logger = logging.getLogger(__name__)


# ============================================================
# 安全类型转换辅助函数
# ============================================================

def _safe_int(env_key: str, default: int) -> int:
    """从环境变量安全读取整数值，解析失败时返回默认值并告警。"""
    raw = os.getenv(env_key, str(default))
    try:
        return int(raw)
    except ValueError:
        logger.warning(
            "⚠️ 环境变量 %s=%s 无法解析为整数，使用默认值 %d", env_key, raw, default
        )
        return default


def _safe_float(env_key: str, default: float) -> float:
    """从环境变量安全读取浮点数值，解析失败时返回默认值并告警。"""
    raw = os.getenv(env_key, str(default))
    try:
        return float(raw)
    except ValueError:
        logger.warning(
            "⚠️ 环境变量 %s=%s 无法解析为浮点数，使用默认值 %.2f", env_key, raw, default
        )
        return default


@dataclass
class Settings:
    """全局配置单例 —— 集中管理所有路径和参数。

    属性：
        PROJECT_ROOT: 项目根目录（自动推断）
        DATA_RAW_PATH: 原始文档存放路径（默认 data/raw，可通过环境变量覆盖）
        VECTOR_DB_PATH: ChromaDB 持久化路径
        MODEL_CACHE_DIR: HuggingFace 模型缓存路径（必须为非系统盘）
        PROMPT_FILE_PATH: Prompt 模板文件路径
        TOP_K_RETRIEVAL: 最终返回的文档片段数量
        VECTOR_WEIGHT: 向量检索权重
        BM25_WEIGHT: BM25 关键词检索权重
        TEMPERATURE: LLM 生成温度
        EMBEDDING_MODEL: 嵌入模型名称
        ACTIVE_PROVIDER: 当前使用的 LLM Provider
        LLM_PROVIDERS: 所有可用的 LLM Provider 配置
    """

    # ========== 项目根目录（自动推断）==========
    PROJECT_ROOT: Path = field(
        default_factory=lambda: Path(__file__).parent.parent.resolve()
    )

    # ========== 数据路径（支持环境变量覆盖）==========
    DATA_RAW_PATH: Path = field(
        default_factory=lambda: Path(
            os.getenv("DATA_RAW_PATH", "")
        )
        if os.getenv("DATA_RAW_PATH")
        else Settings._default_project_root() / "data" / "raw"
    )

    VECTOR_DB_PATH: Path = field(
        default_factory=lambda: Path(
            os.getenv("VECTOR_DB_PATH", "")
        )
        if os.getenv("VECTOR_DB_PATH")
        else Settings._default_project_root() / "vector_db"
    )

    # 模型缓存路径 —— 必须为非系统盘（使用正斜杠避免转义字符问题）
    MODEL_CACHE_DIR: Path = field(
        default_factory=lambda: Path(
            os.getenv("MODEL_CACHE_DIR", "")
        )
        if os.getenv("MODEL_CACHE_DIR")
        else Path("E:/AI_Cache/huggingface")
    )

    # Prompt 模板文件路径
    PROMPT_FILE_PATH: Path = field(
        default_factory=lambda: Path(
            os.getenv("PROMPT_FILE_PATH", "")
        )
        if os.getenv("PROMPT_FILE_PATH")
        else Settings._default_project_root() / "prompts" / "system_prompt.toml"
    )

    # ========== 检索参数 ==========
    TOP_K_RETRIEVAL: int = _safe_int("TOP_K_RETRIEVAL", 8)
    VECTOR_WEIGHT: float = _safe_float("VECTOR_WEIGHT", 0.6)
    BM25_WEIGHT: float = _safe_float("BM25_WEIGHT", 0.4)
    TOP_K_CANDIDATE: int = 10  # 各通路召回候选数
    RRF_K: int = _safe_int("RRF_K", 60)  # RRF 融合常数（越大排名差异越平滑）
    BM25_SCORE_THRESHOLD: float = 0.05  # BM25 最低分数阈值，过滤噪声

    # ========== 权重预设（A/B 测试用）==========
    WEIGHT_PRESETS = {
        "balanced":       {"vector": 0.50, "bm25": 0.50},  # 均衡模式（推荐）
        "semantic_first": {"vector": 0.60, "bm25": 0.40},  # 语义优先
        "keyword_first":  {"vector": 0.40, "bm25": 0.60},  # 关键词优先
        "dense_only":     {"vector": 1.00, "bm25": 0.00},  # 纯向量检索
        "sparse_only":    {"vector": 0.00, "bm25": 1.00},  # 纯关键词检索
    }
    ACTIVE_WEIGHT_PRESET: str = os.getenv("WEIGHT_PRESET", "balanced")

    # ========== 文档切分参数 ==========
    CHUNK_SIZE: int = _safe_int("CHUNK_SIZE", 500)
    CHUNK_OVERLAP: int = _safe_int("CHUNK_OVERLAP", 100)
    MIN_CHUNK_SIZE: int = 200  # 最小块长度（低于此值尝试合并）

    # ========== 生成参数 ==========
    TEMPERATURE: float = _safe_float("TEMPERATURE", 0.1)
    LLM_TIMEOUT: int = 60  # LLM API 超时秒数

    # ========== 嵌入模型 ==========
    EMBEDDING_MODEL: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )

    # ========== LLM Provider 配置 ==========
    ACTIVE_PROVIDER: str = os.getenv("LLM_PROVIDER", "deepseek")

    LLM_PROVIDERS: dict = field(default_factory=lambda: {
        "deepseek": {
            "base_url": "https://api.deepseek.com/v1",
            "model": "deepseek-chat",
            "api_key_env": "DEEPSEEK_API_KEY",
        },
        "openai": {
            "base_url": "https://api.openai.com/v1",
            "model": "gpt-4o",
            "api_key_env": "OPENAI_API_KEY",
        },
        "custom": {
            "base_url": os.getenv("CUSTOM_BASE_URL", ""),
            "model": os.getenv("CUSTOM_MODEL_NAME", ""),
            "api_key_env": "CUSTOM_API_KEY",
        },
    })

    def __post_init__(self):
        """自动执行环境变量劫持和目录创建。"""
        # 1. 劫持 HuggingFace 相关环境变量到非系统盘
        self._hijack_hf_env()

        # 2. 确保路径为绝对路径
        self._resolve_paths()

        # 3. 自动创建必要目录
        self._create_directories()

        # 4. 应用权重预设
        self._apply_weight_preset()

        # 5. 校验
        self._validate()

        logger.info(
            "✅ 配置加载完成 | 数据目录: %s | 向量库: %s | LLM: %s",
            self.DATA_RAW_PATH, self.VECTOR_DB_PATH, self.ACTIVE_PROVIDER,
        )

    def _hijack_hf_env(self):
        """劫持 HuggingFace 缓存环境变量，防止模型下载到 C 盘。"""
        cache_dir = str(self.MODEL_CACHE_DIR)
        os.environ["HF_HOME"] = cache_dir
        os.environ["TRANSFORMERS_CACHE"] = cache_dir
        os.environ["HUGGINGFACE_HUB_CACHE"] = cache_dir
        os.environ["SENTENCE_TRANSFORMERS_HOME"] = cache_dir

    def _resolve_paths(self):
        """将所有路径解析为绝对路径并规范化。"""
        self.PROJECT_ROOT = self.PROJECT_ROOT.resolve()
        self.DATA_RAW_PATH = self.DATA_RAW_PATH.resolve()
        self.VECTOR_DB_PATH = self.VECTOR_DB_PATH.resolve()
        self.MODEL_CACHE_DIR = self.MODEL_CACHE_DIR.resolve()
        self.PROMPT_FILE_PATH = self.PROMPT_FILE_PATH.resolve()

    def _create_directories(self):
        """自动创建数据和向量库目录（parents=True 确保父目录一并创建）。"""
        self.DATA_RAW_PATH.mkdir(parents=True, exist_ok=True)
        self.VECTOR_DB_PATH.mkdir(parents=True, exist_ok=True)
        self.MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def _validate(self):
        """验证配置合法性。"""
        # 检查权重和为 1.0
        total_weight = self.VECTOR_WEIGHT + self.BM25_WEIGHT
        if abs(total_weight - 1.0) > 0.001:
            logger.warning(
                "⚠️ 检索权重之和为 %.2f（预期 1.0），建议修正", total_weight
            )

        # 检查 Active Provider 是否在已注册列表中
        if self.ACTIVE_PROVIDER not in self.LLM_PROVIDERS:
            raise ValueError(
                f"未知的 LLM Provider: {self.ACTIVE_PROVIDER}，"
                f"可用选项: {list(self.LLM_PROVIDERS.keys())}"
            )

    # ---- 权重管理 ----

    def _apply_weight_preset(self):
        """从预设中读取并应用权重配置。"""
        if self.ACTIVE_WEIGHT_PRESET in self.WEIGHT_PRESETS:
            preset = self.WEIGHT_PRESETS[self.ACTIVE_WEIGHT_PRESET]
            # 仅当用户未通过环境变量显式覆盖时才应用预设
            if not os.getenv("VECTOR_WEIGHT"):
                self.VECTOR_WEIGHT = preset["vector"]
            if not os.getenv("BM25_WEIGHT"):
                self.BM25_WEIGHT = preset["bm25"]
            logger.info(
                "⚖️ 权重预设: %s (Dense=%.1f / Sparse=%.1f)",
                self.ACTIVE_WEIGHT_PRESET, self.VECTOR_WEIGHT, self.BM25_WEIGHT,
            )

    def set_weight_preset(self, preset_name: str):
        """运行时切换权重预设。

        Args:
            preset_name: 预设名称（balanced/semantic_first/keyword_first/dense_only/sparse_only）
        """
        if preset_name not in self.WEIGHT_PRESETS:
            raise ValueError(
                f"未知的权重预设: {preset_name}，"
                f"可用选项: {list(self.WEIGHT_PRESETS.keys())}"
            )
        preset = self.WEIGHT_PRESETS[preset_name]
        self.VECTOR_WEIGHT = preset["vector"]
        self.BM25_WEIGHT = preset["bm25"]
        self.ACTIVE_WEIGHT_PRESET = preset_name
        logger.info(
            "⚖️ 已切换权重预设: %s (Dense=%.1f / Sparse=%.1f)",
            preset_name, self.VECTOR_WEIGHT, self.BM25_WEIGHT,
        )

    # ---- 辅助方法 ----

    def get_llm_config(self) -> dict:
        """获取当前激活的 LLM Provider 的完整配置。

        Returns:
            dict: 包含 base_url, api_key, model 的字典
        """
        provider = self.LLM_PROVIDERS[self.ACTIVE_PROVIDER]
        api_key = os.getenv(provider["api_key_env"], "")

        if not api_key:
            logger.warning(
                "⚠️ 环境变量 %s 未设置，LLM 调用将失败",
                provider["api_key_env"],
            )

        return {
            "base_url": provider["base_url"],
            "api_key": api_key,
            "model": provider["model"],
        }

    def get_available_providers(self) -> list[str]:
        """返回所有已注册 Provider 的名称列表。"""
        return list(self.LLM_PROVIDERS.keys())

    @staticmethod
    def _default_project_root() -> Path:
        """用于解决 dataclass field default_factory 中无法引用 self 的问题。"""
        return Path(__file__).parent.parent.resolve()


# 模块级单例 —— 全项目唯一入口
settings = Settings()

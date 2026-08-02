"""
BM25 优化工具 —— 自定义词典、停用词过滤、分词增强。

在 retriever.py 和 build_index.py 中复用，确保索引和检索使用一致的分词策略。
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# ============================================================
# 自定义词典（简历/技术场景专用词，防止 jieba 过度切分）
# ============================================================

CUSTOM_WORDS = [
    # --- AI/ML 术语 ---
    ("大模型", 10), ("强化学习", 10), ("深度学习", 10), ("机器学习", 10),
    ("自然语言处理", 10), ("计算机视觉", 10), ("图像处理", 10),
    ("数据分析", 8), ("数据挖掘", 8), ("特征工程", 8),
    ("神经网络", 8), ("卷积神经网络", 10), ("循环神经网络", 10),
    ("目标检测", 8), ("人脸识别", 8), ("语音识别", 8),
    ("模型训练", 8), ("模型部署", 8), ("模型优化", 8),
    ("推理引擎", 8), ("向量数据库", 8), ("知识图谱", 8),

    # --- 框架/工具 ---
    ("FastAPI", 10), ("Streamlit", 10), ("Docker", 10),
    ("PyTorch", 10), ("TensorFlow", 10), ("LangChain", 10),
    ("ChromaDB", 10), ("SentenceTransformer", 10),
    ("YOLOv5", 10), ("OpenCV", 10), ("Scikit-learn", 10),
    ("GitHub Actions", 10), ("CI/CD", 10),
    ("RAG", 10), ("LLM", 10),

    # --- 编程语言 ---
    ("Python", 10), ("C++", 8), ("Java", 8), ("TypeScript", 8),
    ("JavaScript", 8), ("SQL", 8),

    # --- 数据库/中间件 ---
    ("PostgreSQL", 10), ("MySQL", 10), ("Redis", 10), ("MongoDB", 10),
    ("Elasticsearch", 10), ("RabbitMQ", 10), ("Kafka", 10),

    # --- 简历场景词 ---
    ("自我介绍", 8), ("项目经历", 8), ("工作经历", 8), ("教育背景", 8),
    ("技术栈", 8), ("团队协作", 8), ("沟通能力", 8), ("领导力", 8),
    ("问题解决", 8), ("系统架构", 8), ("微服务", 8),
    ("全栈开发", 8), ("后端开发", 8), ("前端开发", 8),
    ("数据分析师", 8), ("算法工程师", 8), ("视觉开发工程师", 10),
    ("软件工程师", 8), ("测试工程师", 8),
    ("本科", 5), ("硕士", 5), ("博士", 5), ("GPA", 8),
]

# ============================================================
# 停用词表（中文常见无意义词 + 标点）
# ============================================================

STOPWORDS = set([
    # 中文停用词
    "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一",
    "一个", "上", "也", "很", "到", "说", "要", "去", "你", "会", "着",
    "没有", "看", "好", "自己", "这", "他", "她", "它", "们", "那", "些",
    "所", "为", "所以", "因为", "但是", "然而", "虽然", "如果", "可以",
    "这个", "那个", "这些", "那些", "什么", "怎么", "怎样", "哪", "吗",
    "啊", "呢", "吧", "嘛", "哦", "嗯", "哈", "呀",
    "与", "及", "或", "等", "从", "被", "把", "让", "向", "对", "以",
    "之", "其", "中", "而", "且", "但", "却", "只", "仅", "还", "又",
    "再", "更", "最", "非常", "比较", "十分",

    # 英文停用词
    "the", "a", "an", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "can", "shall",
    "i", "me", "my", "we", "our", "you", "your", "he", "him",
    "his", "she", "her", "it", "its", "they", "them", "their",
    "this", "that", "these", "those", "in", "on", "at", "to",
    "for", "of", "with", "by", "from", "as", "into", "through",
    "during", "before", "after", "above", "below", "between",
    "and", "but", "or", "nor", "not", "so", "yet", "both",
    "each", "every", "all", "any", "few", "more", "most",
    "other", "some", "such", "no", "only", "own", "same",
    "than", "too", "very", "just", "about", "over", "also",
])


def setup_jieba():
    """配置 jieba 分词器：加载自定义词典。

    应在索引构建和检索器初始化时各调用一次。

    Returns:
        配置完成的 jieba 模块
    """
    import jieba

    # 加载自定义词典（防止过度切分技术术语）
    for word, freq in CUSTOM_WORDS:
        jieba.add_word(word, freq)

    logger.info(
        "📚 jieba 已配置: %d 个自定义词, %d 个停用词",
        len(CUSTOM_WORDS), len(STOPWORDS),
    )
    return jieba


def tokenize(text: str, jieba_instance=None, remove_stopwords: bool = True) -> list[str]:
    """增强分词：自定义词典 + 停用词过滤 + 最小长度过滤。

    Args:
        text: 待分词文本
        jieba_instance: jieba 实例（可选，若为 None 则自动初始化）
        remove_stopwords: 是否移除停用词

    Returns:
        分词后的 token 列表
    """
    if jieba_instance is None:
        jieba_instance = setup_jieba()

    # 精确模式分词
    tokens = jieba_instance.lcut(text)

    # 过滤
    filtered = []
    for token in tokens:
        token = token.strip()
        if not token:
            continue
        # 过滤纯标点/空白
        if len(token) == 1 and not token.isalnum():
            continue
        # 过滤停用词
        if remove_stopwords and token.lower() in STOPWORDS:
            continue
        filtered.append(token)

    return filtered

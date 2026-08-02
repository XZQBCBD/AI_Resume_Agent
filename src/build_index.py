"""
离线索引构建脚本 —— 遍历文档、向量化、构建 BM25、存入 ChromaDB。

职责：
    1. 遍历 DATA_RAW_PATH 下的所有文档
    2. 调用 DocumentLoader 加载并切分
    3. 使用 sentence-transformers 生成向量嵌入
    4. 存入 ChromaDB（持久化）
    5. 构建 BM25 关键词索引并序列化到磁盘
    6. 防御性编程：若目录为空，自动生成 4 个中文示例文件

使用方式：
    python -m src.build_index          # 增量模式（追加新文档）
    python -m src.build_index --full   # 全量模式（清空后重建）
"""

import sys
import json
import pickle
import logging
from pathlib import Path
from typing import List, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================
# 示例数据（防御性编程）
# ============================================================

SAMPLE_FILES = {
    # 注意：示例文件使用 .txt/.md 格式以确保开箱可用
    # 实际使用时请替换为真实的 PDF、DOCX 文件
    "01_简历_张三.txt": """张三的个人简历

教育背景
2018-2022  北京大学  计算机科学与技术  本科
GPA 3.8/4.0，校级优秀毕业生

技术技能
编程语言：Python（精通）、Java（熟练）、TypeScript（了解）
框架：FastAPI、Django、Spring Boot、React
AI/ML：LangChain、LlamaIndex、ChromaDB、Sentence-Transformers、OpenAI API
工具链：Docker、Git、Linux、CI/CD（GitHub Actions）
数据库：PostgreSQL、MySQL、Redis、MongoDB

工作经历
2022-至今  某科技有限公司  AI 应用开发工程师（Python）
· 主导公司智能客服系统后端架构设计，基于 RAG 技术实现文档问答准确率从 72% 提升至 91%
· 设计并实现混合检索引擎（向量 + BM25），日均处理 5000+ 次查询，P99 延迟 < 800ms
· 搭建公司内部 AI 开发平台，支持 Prompt 管理和模型切换，服务 3 个业务线

2021-2022  某互联网公司  后端开发实习生
· 参与用户中心微服务拆分，将单体应用按业务领域拆分为 6 个独立服务
· 编写单元测试 200+ 个，代码覆盖率从 45% 提升至 78%

项目经历
1. AI 数字分身系统（2024）
   基于 RAG 的个人知识问答系统，支持混合检索和多 LLM Provider 切换。

2. 智能客服 RAG 平台（2023）
   企业级智能客服系统，集成多种数据源，支持流式对话和反馈收集。

自我评价
热爱技术，持续学习。善于将 AI 前沿技术落地到实际业务场景中。具备良好的沟通能力和团队协作精神。""",

    "02_项目_AI数字分身.md": """# AI 数字分身系统

## 项目概述
基于 RAG（检索增强生成）的个人 AI 数字分身系统，用于求职面试场景。

## 技术架构

### 整体方案
采用 FastAPI + Streamlit 前后端分离架构：
- 后端：FastAPI 提供 REST API，封装 RAG 核心引擎
- 前端：Streamlit 提供专业 Web 界面
- 检索：Dense（向量语义）+ Sparse（BM25 关键词）混合检索

### 核心技术栈
| 层次 | 技术选型 |
|------|----------|
| 嵌入模型 | sentence-transformers/all-MiniLM-L6-v2（384维）|
| 向量存储 | ChromaDB（本地持久化，基于 SQLite）|
| 关键词检索 | rank-bm25 + jieba 分词 |
| LLM 接入 | OpenAI 兼容 API，支持多 Provider 切换 |
| Prompt 管理 | TOML 模板外部化，修改即生效 |

### 架构设计亮点
1. 混合检索融合算法：Min-Max 归一化 + 加权求和（0.6 向量 + 0.4 BM25）
2. 智能文本切分：段落边界 + 句子边界二次切分（500 字符阈值）
3. 路径全可控：所有数据/缓存/模型路径由配置文件管理

## 难点与解决方案

### 难点 1：混合检索的融合策略
单纯向量检索会漏掉精确的关键词匹配，单纯 BM25 又无法理解语义。
解决方案：双路召回各取 Top-10，Min-Max 归一化后加权融合（向量 0.6 + BM25 0.4），兼顾语义和精确匹配。

### 难点 2：文档切分的语义完整性
固定长度切分会在句子中间截断，破坏语义。
解决方案：优先按自然段落切分，超长段落（>500字符）按中英文句子边界二次切分。

## 项目成果
- 检索 Top-5 召回率达到 89%
- 单次问答响应时间 < 3 秒
- 支持 4 种文档格式，3 种 LLM Provider""",

    "03_博客_AI-Agent未来趋势.txt": """AI Agent 的未来：从工具到数字分身

最近一年，AI Agent 从概念走向落地，我认为未来有三个关键趋势：

第一，从单轮对话到自主决策。早期的 ChatGPT 只能一问一答，现在的 Agent 可以自主规划、调用工具、迭代修正。AutoGPT 和 MetaGPT 打开了多 Agent 协作的大门。

第二，RAG 成为企业级标配。纯 LLM 存在幻觉和知识滞后问题。RAG（检索增强生成）通过外部知识库提供事实支撑，在客服、法务、医疗等领域已经开始规模化落地。我们团队做的智能客服 RAG 平台就是典型案例——准确率从 72% 提升到 91%。

第三，个人 AI 数字分身将成为趋势。每个人都会拥有自己的 AI 分身——它了解你的经历、风格和偏好，可以替你参加面试、写文档、做汇报。技术上，关键是建立一个高质量的个人知识库和人格化的生成策略。

我认为最有价值的不是模型本身，而是如何将模型与具体业务场景深度结合。这需要同时理解技术和业务，这正是我在努力的方向。""",

    "04_推荐信_技术总监.txt": """推荐信

被推荐人：张三

我非常荣幸地为张三撰写这封推荐信。张三于 2022 年至 2024 年在某科技有限公司担任 AI 应用开发工程师，由我直接管理。

技术能力方面，张三是团队中最出色的 Python 工程师之一。他主导的智能客服 RAG 平台项目，独立完成了从架构设计到上线的全过程。他提出的混合检索引擎方案（向量 + BM25）将问答准确率从 72% 显著提升至 91%，这一成果获得了公司年度技术创新奖。

问题解决能力方面，张三展现了出色的系统性思考能力。面对检索延迟过高的问题，他不仅定位到向量维度瓶颈，还通过引入结果缓存和批量处理将 P99 延迟控制在 800ms 以内。

团队协作方面，张三积极推动代码审查文化，主动编写技术文档，帮助 3 名新人快速上手项目。他的技术分享《RAG 实践：从 0 到 1》在公司内部获得了最高评分。

我毫无保留地推荐张三，相信他能在任何技术团队中发挥关键作用。任何公司雇佣他都将是明智的投资。

某科技有限公司
技术总监：李四
2024 年 1 月""",
}


def generate_sample_files(data_dir: Path):
    """在指定目录下生成 4 个中文示例文件。"""
    data_dir.mkdir(parents=True, exist_ok=True)

    for filename, content in SAMPLE_FILES.items():
        filepath = data_dir / filename
        if filepath.exists():
            logger.info("  ⏭️ 跳过（已存在）: %s", filename)
            continue

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        logger.info("  ✅ 已生成: %s", filename)


# ============================================================
# 索引构建器
# ============================================================

class IndexBuilder:
    """离线索引构建器 —— 将文档转化为可检索的向量和关键词索引。"""

    def __init__(self):
        from src.config import settings
        from src.loader import DocumentLoader
        from sentence_transformers import SentenceTransformer
        from rank_bm25 import BM25Okapi
        from src.bm25_utils import setup_jieba, tokenize

        self.settings = settings
        self.loader = DocumentLoader(
            max_chunk_chars=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
        )

        # 初始化嵌入模型
        logger.info("🔧 加载嵌入模型: %s", settings.EMBEDDING_MODEL)
        self.embedding_model = SentenceTransformer(
            settings.EMBEDDING_MODEL,
            cache_folder=str(settings.MODEL_CACHE_DIR),
        )

        self.jieba = setup_jieba()
        self.tokenize = tokenize
        self.BM25Okapi = BM25Okapi

    def build(self, full_rebuild: bool = False):
        """执行索引构建。

        Args:
            full_rebuild: True=清空后全量重建, False=增量追加
        """
        data_dir = self.settings.DATA_RAW_PATH

        # 0. 防御性编程：目录为空时自动生成示例文件
        supported = {".pdf", ".docx", ".md", ".txt"}
        existing = [f for f in data_dir.iterdir() if f.suffix.lower() in supported]
        if not existing:
            logger.info("📝 数据目录为空，自动生成示例文件...")
            generate_sample_files(data_dir)

        # 1. 加载文档并切分
        logger.info("📄 开始加载文档...")
        chunks = self.loader.load_all(data_dir)
        if not chunks:
            logger.error("❌ 未加载到任何有效文档，索引构建终止。")
            return

        # 2. 生成向量嵌入
        logger.info("🧮 生成向量嵌入 (%d 个文本块)...", len(chunks))
        texts = [c.content for c in chunks]
        embeddings = self.embedding_model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True,
        )

        # 3. 保存向量嵌入到磁盘（numpy 格式，避免 ChromaDB 兼容性问题）
        logger.info("💾 保存向量嵌入...")
        import numpy as np

        embeddings_path = self.settings.VECTOR_DB_PATH / "embeddings.npy"
        ids_path = self.settings.VECTOR_DB_PATH / "embedding_ids.json"

        # 生成唯一 ID
        ids = [
            f"{c.metadata['source']}_chunk_{c.metadata['chunk_index']}"
            for c in chunks
        ]

        # 全量模式直接覆盖，增量模式合并
        if not full_rebuild and embeddings_path.exists():
            existing_embs = np.load(embeddings_path)
            with open(ids_path, "r", encoding="utf-8") as f:
                existing_ids = json.load(f)
            # 去重：只添加新的 id
            existing_set = set(existing_ids)
            new_mask = [i for i, id_ in enumerate(ids) if id_ not in existing_set]
            if new_mask:
                new_embs = embeddings[new_mask]
                all_embs = np.vstack([existing_embs, new_embs])
                all_ids = existing_ids + [ids[i] for i in new_mask]
                logger.info("  ✅ 新增 %d 条向量记录（总计 %d）", len(new_mask), len(all_ids))
            else:
                all_embs = existing_embs
                all_ids = existing_ids
                logger.info("  ⏭️ 无新增文档，跳过向量入库")
        else:
            all_embs = embeddings
            all_ids = ids
            logger.info("  ✅ 保存 %d 条向量记录", len(all_ids))

        np.save(embeddings_path, all_embs)
        with open(ids_path, "w", encoding="utf-8") as f:
            json.dump(all_ids, f, ensure_ascii=False)

        logger.info("  📁 向量索引: %s", embeddings_path)

        # 4. 构建 BM25 索引并序列化（增强分词：自定义词典 + 停用词过滤）
        logger.info("🔤 构建 BM25 关键词索引（增强分词）...")
        tokenized_corpus = [self.tokenize(text, self.jieba) for text in texts]
        bm25_index = self.BM25Okapi(tokenized_corpus)

        bm25_path = self.settings.VECTOR_DB_PATH / "bm25_index.pkl"
        bm25_data = {
            "tokenized_corpus": tokenized_corpus,
            "bm25_index": bm25_index,
            "chunks": [(c.content, c.metadata) for c in chunks],
        }
        with open(bm25_path, "wb") as f:
            pickle.dump(bm25_data, f)
        logger.info("  ✅ BM25 索引已保存: %s", bm25_path)

        # 5. 保存 chunk 元数据
        meta_path = self.settings.VECTOR_DB_PATH / "chunks_meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(
                [{"content": c.content, "metadata": c.metadata} for c in chunks],
                f, ensure_ascii=False, indent=2,
            )

        logger.info("🎉 索引构建完成！共 %d 个文本块", len(chunks))



# ============================================================
# 变更检测与智能同步
# ============================================================

import hashlib
from datetime import datetime


def compute_file_hash(filepath: Path) -> str:
    """计算文件 MD5 哈希，用于检测文件变更。"""
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def scan_files(data_dir: Path) -> dict[str, str]:
    """扫描目录，返回 {文件名: MD5哈希} 映射。"""
    supported = {".pdf", ".docx", ".md", ".txt"}
    files = {}
    for ext in supported:
        for f in data_dir.glob(f"*{ext}"):
            files[f.name] = compute_file_hash(f)
    return files


def detect_changes(data_dir: Path, hash_file: Path) -> dict:
    """检测文档变更：新增、修改、删除。

    Args:
        data_dir: 文档目录
        hash_file: 存储上次哈希的文件路径

    Returns:
        {
            "added": [文件名],
            "modified": [文件名],
            "deleted": [文件名],
            "unchanged": [文件名],
            "has_changes": bool,
        }
    """
    current_files = scan_files(data_dir)

    # 加载上次哈希记录
    previous_files = {}
    if hash_file.exists():
        with open(hash_file, "r", encoding="utf-8") as f:
            previous_files = json.load(f)

    current_names = set(current_files.keys())
    previous_names = set(previous_files.keys())

    added = sorted(current_names - previous_names)
    deleted = sorted(previous_names - current_names)
    modified = sorted(
        name for name in (current_names & previous_names)
        if current_files[name] != previous_files[name]
    )
    unchanged = sorted(
        name for name in (current_names & previous_names)
        if current_files[name] == previous_files[name]
    )

    return {
        "added": added,
        "modified": modified,
        "deleted": deleted,
        "unchanged": unchanged,
        "has_changes": bool(added or modified or deleted),
    }


def save_file_hashes(data_dir: Path, hash_file: Path):
    """保存当前文件哈希快照。"""
    current_files = scan_files(data_dir)
    hash_file.parent.mkdir(parents=True, exist_ok=True)
    with open(hash_file, "w", encoding="utf-8") as f:
        json.dump(current_files, f, ensure_ascii=False, indent=2)
    logger.info("📸 文件哈希快照已保存: %s", hash_file)


def print_change_report(changes: dict):
    """打印变更检测报告。"""
    logger.info("=" * 50)
    logger.info("🔍 文档变更检测报告")
    logger.info("-" * 50)

    if changes["added"]:
        logger.info("  ➕ 新增 (%d): %s", len(changes["added"]), changes["added"])
    if changes["modified"]:
        logger.info("  ✏️  修改 (%d): %s", len(changes["modified"]), changes["modified"])
    if changes["deleted"]:
        logger.info("  ➖ 删除 (%d): %s", len(changes["deleted"]), changes["deleted"])
    if changes["unchanged"]:
        logger.info("  ✅ 未变 (%d): %s", len(changes["unchanged"]), changes["unchanged"])

    if not changes["has_changes"]:
        logger.info("  ✨ 无任何变更，索引已是最新。")
    logger.info("=" * 50)


# ============================================================
# CLI 入口
# ============================================================

if __name__ == "__main__":
    from src.config import settings

    hash_file = settings.VECTOR_DB_PATH / "file_hashes.json"
    dry_run = "--dry-run" in sys.argv

    if "--sync" in sys.argv:
        # 智能同步模式：检测变更 → 报告 → 按需重建
        logger.info("🔍 检测文档变更...")
        changes = detect_changes(settings.DATA_RAW_PATH, hash_file)
        print_change_report(changes)

        if dry_run:
            logger.info("💡 --dry-run 模式，不执行实际操作。")
            sys.exit(0)

        if changes["has_changes"]:
            # 有变更则全量重建（最干净的方式，避免 ID 偏移问题）
            logger.info("🔄 检测到变更，执行全量重建...")
            builder = IndexBuilder()
            builder.build(full_rebuild=True)
            save_file_hashes(settings.DATA_RAW_PATH, hash_file)
        else:
            logger.info("✨ 无变更，跳过索引构建。")

    elif "--full" in sys.argv or "--rebuild" in sys.argv:
        # 全量重建模式
        builder = IndexBuilder()
        builder.build(full_rebuild=True)
        save_file_hashes(settings.DATA_RAW_PATH, hash_file)

    elif "--status" in sys.argv:
        # 仅检查状态
        changes = detect_changes(settings.DATA_RAW_PATH, hash_file)
        print_change_report(changes)

    else:
        # 默认增量模式
        builder = IndexBuilder()
        builder.build(full_rebuild=False)
        save_file_hashes(settings.DATA_RAW_PATH, hash_file)

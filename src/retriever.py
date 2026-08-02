"""
混合检索器 —— Dense（向量语义）+ Sparse（BM25 关键词）+ 融合排序。

职责：
    1. Dense 通路：用户问题 → embedding → numpy 余弦相似度（Top-10）
    2. Sparse 通路：用户问题 → jieba 分词 → BM25.get_scores（Top-10）
    3. 融合排序：Min-Max 归一化 → 加权求和（0.6 + 0.4）→ Top-K

使用方式：
    from src.retriever import HybridRetriever
    retriever = HybridRetriever()
    results = retriever.search("你最擅长什么技术？")
"""

import json
import os
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Dense + Sparse 混合检索器。

    采用双路召回 + 加权融合策略：
    - Dense 通路：通过句子嵌入捕获语义相似性
    - Sparse 通路：通过 BM25 关键词匹配保证精确召回
    - 融合：Min-Max 归一化后加权求和
    """

    def __init__(self):
        from src.config import settings
        from sentence_transformers import SentenceTransformer
        from src.bm25_utils import setup_jieba, tokenize

        self.settings = settings
        self.jieba = setup_jieba()
        self.tokenize = tokenize

        # 初始化嵌入模型
        logger.info("🔧 加载嵌入模型: %s", settings.EMBEDDING_MODEL)
        self.embedding_model = SentenceTransformer(
            settings.EMBEDDING_MODEL,
            cache_folder=str(settings.MODEL_CACHE_DIR),
        )

        # 加载向量索引
        self._load_embeddings()

        # 加载 BM25 索引
        self._load_bm25_index()

        # 加载 chunks 元数据
        self._load_chunks_meta()

        # ── 数据一致性校验 ──
        self._validate_index_consistency()

        logger.info(
            "✅ 混合检索器初始化完成 | 文档块: %d | 向量维度: %d",
            len(self.all_chunks),
            self.embeddings.shape[1] if self.embeddings is not None else 0,
        )

    def search(self, query: str, top_k: Optional[int] = None) -> List[Dict]:
        """执行混合检索。

        Args:
            query: 用户查询问题
            top_k: 返回结果数量（默认使用 config 中的 TOP_K_RETRIEVAL）

        Returns:
            结果列表，每项包含 content, source, score, file_type, chunk_index
        """
        if top_k is None:
            top_k = self.settings.TOP_K_RETRIEVAL

        # RRF 融合需要更大的候选池以保证召回覆盖
        n_candidates = max(self.settings.TOP_K_CANDIDATE, top_k * 4)

        # 1. Dense 通路：向量语义检索
        dense_results = self._dense_search(query, n_candidates=n_candidates)

        # 2. Sparse 通路：BM25 关键词检索
        sparse_results = self._sparse_search(query, n_candidates=n_candidates)

        # 3. RRF 融合排序
        fused = self._fuse_results_rrf(dense_results, sparse_results, top_k)

        return fused

    def _dense_search(self, query: str, n_candidates: int | None = None) -> List[Dict]:
        """Dense 通路：向量语义检索（余弦相似度）。

        Args:
            query: 用户查询
            n_candidates: 返回候选数量（默认 TOP_K_CANDIDATE）

        Returns:
            [{content, source, score}, ...] 相似度分数 ∈ [0,1]
        """
        if n_candidates is None:
            n_candidates = self.settings.TOP_K_CANDIDATE

        if self.embeddings is None or len(self.all_chunks) == 0:
            logger.warning("⚠️ 向量索引为空，跳过 Dense 检索")
            return []

        # 生成查询向量
        query_embedding = self.embedding_model.encode(
            query, convert_to_numpy=True
        )

        # 计算余弦相似度
        # cosine_sim = dot(A, B) / (||A|| * ||B||)
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        emb_norms = self.embeddings / (np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-8)
        similarities = np.dot(emb_norms, query_norm)

        # 取 Top-N（上限受 chunks 数量约束，防止数据不一致时越界）
        max_idx = min(len(similarities), len(self.all_chunks))
        n_candidate = min(n_candidates, max_idx)
        top_indices = np.argsort(similarities)[::-1][:n_candidate]

        output = []
        for idx in top_indices:
            if int(idx) >= len(self.all_chunks):
                continue
            sim = similarities[idx]
            if sim <= 0:
                continue
            chunk = self.all_chunks[int(idx)]
            output.append({
                "content": chunk["content"],
                "source": chunk["metadata"].get("source", "unknown"),
                "file_type": chunk["metadata"].get("file_type", "其他文档"),
                "chunk_index": chunk["metadata"].get("chunk_index", 0),
                "score": float(sim),
                "score_type": "dense",
            })

        return output

    def _sparse_search(self, query: str, n_candidates: int | None = None) -> List[Dict]:
        """Sparse 通路：BM25 关键词检索。

        优化：
            - 自定义词典防止过度切分技术术语
            - 停用词过滤去除噪声
            - 分数阈值过滤低质量命中
        """
        if n_candidates is None:
            n_candidates = self.settings.TOP_K_CANDIDATE

        if self.bm25_index is None:
            logger.warning("⚠️ BM25 索引未加载，跳过 Sparse 检索")
            return []

        # 增强分词：自定义词典 + 停用词过滤
        tokenized_query = self.tokenize(query, self.jieba, remove_stopwords=True)

        if not tokenized_query:
            logger.debug("  BM25 分词后为空: %s", query)
            return []

        scores = self.bm25_index.get_scores(tokenized_query)

        # 分数阈值过滤（上限受 chunks 数量约束）
        score_threshold = self.settings.BM25_SCORE_THRESHOLD
        max_idx = min(len(scores), len(self.all_chunks))
        n_candidate = min(n_candidates, max_idx)
        top_indices = np.argsort(scores)[::-1][:n_candidate]

        output = []
        for idx in top_indices:
            if int(idx) >= len(self.all_chunks):
                continue
            if scores[idx] <= score_threshold:
                continue
            chunk = self.all_chunks[int(idx)]
            output.append({
                "content": chunk["content"],
                "source": chunk["metadata"].get("source", "unknown"),
                "file_type": chunk["metadata"].get("file_type", "其他文档"),
                "chunk_index": chunk["metadata"].get("chunk_index", 0),
                "score": float(scores[idx]),
                "score_type": "sparse",
            })

        return output

    def _fuse_results_rrf(self, dense: List[Dict], sparse: List[Dict], top_k: int) -> List[Dict]:
        """RRF（Reciprocal Rank Fusion）融合 Dense 和 Sparse 检索结果。

        算法：对每个通路的排名取倒数加权，无需分数归一化。
              RRF_score(d) = Σ 1/(k + rank_i(d))

        优势：
            - 无需 Min-Max 归一化，天然处理异构分数量纲
            - 对异常高分不敏感，更鲁棒
            - 双路命中自动获得更高权重（两个通路都有排名）
            - 工业界标准方案（Elasticsearch 8.x+ 内置）

        Args:
            dense: Dense 通路结果（已按分数降序排列）
            sparse: Sparse 通路结果（已按分数降序排列）
            top_k: 最终返回数量

        Returns:
            按 RRF 分数降序排列的 top_k 结果
        """
        k = self.settings.RRF_K

        merged: Dict[str, Dict] = {}

        # Dense 通路：按排名计算 RRF 分数
        for rank, item in enumerate(dense, 1):
            key = f"{item['source']}_{item['chunk_index']}"
            rrf = 1.0 / (k + rank)
            merged[key] = {
                **item,
                "rrf_score": rrf,
                "score_type": "dense",
                "dense_rank": rank,
                "dense_score": item.get("score", 0),
                "sparse_score": 0,
            }

        # Sparse 通路：按排名计算 RRF 分数
        for rank, item in enumerate(sparse, 1):
            key = f"{item['source']}_{item['chunk_index']}"
            rrf = 1.0 / (k + rank)
            if key in merged:
                merged[key]["rrf_score"] += rrf
                merged[key]["score_type"] = "both"
                merged[key]["sparse_rank"] = rank
                merged[key]["sparse_score"] = item.get("score", 0)
            else:
                merged[key] = {
                    **item,
                    "rrf_score": rrf,
                    "score_type": "sparse",
                    "sparse_rank": rank,
                    "dense_score": 0,
                    "sparse_score": item.get("score", 0),
                }

        # 按 RRF 分数降序排列（RRF 仅负责排序，不产生最终展示分数）
        sorted_results = sorted(
            merged.values(), key=lambda x: x["rrf_score"], reverse=True
        )
        for item in sorted_results:
            item["rrf_raw"] = round(item.pop("rrf_score"), 6)

        top_results = sorted_results[:top_k]

        # ── 分数归一化：对原始 Dense/BM25 分数做 Min-Max，加权求和生成可读分数 ──
        if top_results:
            dense_raw = [item.get("dense_score", 0) for item in top_results]
            sparse_raw = [item.get("sparse_score", 0) for item in top_results]

            # Min-Max 归一化（分别处理，解决分数量纲不一致问题）
            def _norm(values):
                mn, mx = min(values), max(values)
                if mx == mn:
                    return [1.0] * len(values)
                return [(v - mn) / (mx - mn) for v in values]

            d_norm = _norm(dense_raw)
            s_norm = _norm(sparse_raw)

            w_d = self.settings.VECTOR_WEIGHT
            w_s = self.settings.BM25_WEIGHT

            for i, item in enumerate(top_results):
                item["score"] = round(w_d * d_norm[i] + w_s * s_norm[i], 4)

        # ── DEBUG 日志 ──
        if os.getenv("DEBUG_RETRIEVER", "").lower() in ("1", "true", "yes"):
            logger.info("  🔍 RRF 融合详情 (Top-%d):", min(3, len(top_results)))
            for i, item in enumerate(top_results):
                logger.info(
                    "    #%d [%.4f / RRF=%.6f] type=%s src=%s chunk=%s",
                    i + 1, item["score"], item["rrf_raw"],
                    item.get("score_type", "?"),
                    item.get("source", "?")[:30],
                    item.get("chunk_index", "?"),
                )
                logger.info("       %s", item["content"][:80].replace("\n", " "))

        return top_results

    def _fuse_results(self, dense: List[Dict], sparse: List[Dict], top_k: int) -> List[Dict]:
        """[已弃用] Min-Max 归一化 + 加权求和，保留用于 A/B 对比。

        新代码请使用 _fuse_results_rrf。
        """
        dense_norm = self._minmax_normalize(dense)
        sparse_norm = self._minmax_normalize(sparse)

        w_vec = self.settings.VECTOR_WEIGHT
        w_bm25 = self.settings.BM25_WEIGHT

        merged: Dict[str, Dict] = {}

        for item in dense_norm:
            key = f"{item['source']}_{item['chunk_index']}"
            item["fused_score"] = w_vec * item["score"]
            merged[key] = item

        for item in sparse_norm:
            key = f"{item['source']}_{item['chunk_index']}"
            score_contrib = w_bm25 * item["score"]
            if key in merged:
                merged[key]["fused_score"] += score_contrib
                merged[key]["score_type"] = "both"
            else:
                item["fused_score"] = score_contrib
                merged[key] = item

        sorted_results = sorted(
            merged.values(), key=lambda x: x["fused_score"], reverse=True
        )

        top_results = sorted_results[:top_k]
        for item in top_results:
            item["score"] = round(item.pop("fused_score"), 4)

        return top_results

    def _minmax_normalize(self, items: List[Dict]) -> List[Dict]:
        """Min-Max 归一化到 [0, 1]。"""
        if not items:
            return []
        scores = [item["score"] for item in items]
        min_s, max_s = min(scores), max(scores)
        if max_s == min_s:
            return [{**item, "score": 1.0} for item in items]
        return [
            {**item, "score": (item["score"] - min_s) / (max_s - min_s)}
            for item in items
        ]

    def _load_embeddings(self):
        """从磁盘加载向量嵌入。"""
        emb_path = self.settings.VECTOR_DB_PATH / "embeddings.npy"
        ids_path = self.settings.VECTOR_DB_PATH / "embedding_ids.json"

        if not emb_path.exists():
            logger.warning("⚠️ 向量索引文件不存在，请先运行 python -m src.build_index")
            self.embeddings = None
            self.embedding_ids = []
            return

        self.embeddings = np.load(emb_path)
        with open(ids_path, "r", encoding="utf-8") as f:
            self.embedding_ids = json.load(f)
        logger.info("📂 向量索引已加载: %d 条", len(self.embedding_ids))

    def _load_bm25_index(self):
        """从磁盘加载 BM25 索引。"""
        bm25_path = self.settings.VECTOR_DB_PATH / "bm25_index.pkl"
        if not bm25_path.exists():
            logger.warning("⚠️ BM25 索引文件不存在，请先运行 python -m src.build_index")
            self.bm25_index = None
            self.tokenized_corpus = []
            return

        with open(bm25_path, "rb") as f:
            data = pickle.load(f)
        self.bm25_index = data["bm25_index"]
        self.tokenized_corpus = data["tokenized_corpus"]
        logger.info("📂 BM25 索引已加载: %d 条", len(self.tokenized_corpus))

    def _load_chunks_meta(self):
        """加载 chunk 元数据列表。"""
        meta_path = self.settings.VECTOR_DB_PATH / "chunks_meta.json"
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                self.all_chunks = json.load(f)
        else:
            self.all_chunks = []

    def _validate_index_consistency(self):
        """校验向量索引、BM25 索引与 chunks 元数据的一致性。

        检测数据文件不同步的情况（多见于索引构建中断或手动修改数据文件），
        打印告警日志并自动裁剪以保障运行时不崩溃。
        """
        emb_count = self.embeddings.shape[0] if self.embeddings is not None else 0
        bm25_count = len(self.tokenized_corpus)
        chunks_count = len(self.all_chunks)

        counts = {
            "embeddings": emb_count,
            "embedding_ids": len(self.embedding_ids),
            "bm25_corpus": bm25_count,
            "chunks_meta": chunks_count,
        }
        # 找出基准值（多数一致的值）
        from collections import Counter
        cnt = Counter(counts.values())
        majority = cnt.most_common(1)[0][0]

        mismatches = {k: v for k, v in counts.items() if v != majority}
        if mismatches:
            logger.warning(
                "⚠️ 索引数据不一致！各文件记录数: %s",
                {k: v for k, v in counts.items()},
            )
            logger.warning(
                "   多数值=%d，不一致项=%s。建议运行 python -m src.build_index --full 重建。",
                majority, mismatches,
            )
            # 以最小值为安全上限，自动裁剪防止越界
            safe_limit = min(v for v in counts.values() if v > 0)
            if emb_count > safe_limit:
                logger.warning("   🔧 自动裁剪 embeddings (%d → %d)", emb_count, safe_limit)
                self.embeddings = self.embeddings[:safe_limit]
                self.embedding_ids = self.embedding_ids[:safe_limit]
            if bm25_count > safe_limit:
                logger.warning("   🔧 自动裁剪 BM25 corpus (%d → %d)", bm25_count, safe_limit)
                self.tokenized_corpus = self.tokenized_corpus[:safe_limit]
            if chunks_count > safe_limit:
                logger.warning("   🔧 自动裁剪 chunks_meta (%d → %d)", chunks_count, safe_limit)
                self.all_chunks = self.all_chunks[:safe_limit]
        else:
            logger.info("✅ 索引一致性校验通过 (%d 条记录)", chunks_count)


# ============================================================
# 模块自测
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    retriever = HybridRetriever()
    results = retriever.search("你最擅长什么技术？")
    print(f"\n查询: 你最擅长什么技术？")
    print(f"返回 {len(results)} 条结果:\n")
    for i, r in enumerate(results):
        print(f"{i+1}. [{r['score']}] [{r['file_type']}] {r['source']}")
        print(f"   {r['content'][:100]}...\n")

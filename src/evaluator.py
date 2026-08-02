"""
RAG 系统性能评估模块。

评估维度：
    1. 检索性能：召回数、分数分布、双路命中率
    2. 时延性能：检索耗时、生成耗时、端到端耗时
    3. 文本质量：chunk 长度分布、重叠覆盖度

使用方式：
    python -m src.evaluator                        # 自动基准测试
    python -m src.evaluator --question "自我介绍"   # 单条评估
"""

import sys
import json
import time
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass, field

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


# ============================================================
# 评估基准问题（覆盖不同检索场景）
# ============================================================

BENCHMARK_QUESTIONS = [
    # 自我介绍
    "请做自我介绍",
    "简要介绍一下你的职业背景和技术方向",
    # 项目经历 — AI数字分身 + MamaCare
    "详细介绍一下你的AI数字分身RAG系统",
    "介绍一下MamaCare孕期智能守护助手这个项目",
    "你在项目中遇到过什么技术难题，怎么解决的？",
    # 技术栈
    "你擅长哪些技术栈？",
    "你对大模型应用开发熟悉吗？用过哪些框架和工具？",
    "你会哪些编程语言？",
    # RAG 系统
    "你的混合检索是如何实现的？用了什么融合算法？",
    "你是如何处理文档切分的？",
    # Agent 开发
    "MamaCare项目中Agent的工作流是如何编排的？",
    "Agent的记忆系统是如何设计的？",
    # 教育 + 团队 + 工程
    "你的教育背景是什么？",
    "你在团队中通常扮演什么角色？有什么特长？",
    "你在项目中用了哪些工程化实践和设计模式？",
]


@dataclass
class RetrievalStats:
    """检索阶段统计。"""
    query: str = ""
    total_results: int = 0
    dense_hits: int = 0          # 向量通路命中数
    sparse_hits: int = 0         # BM25 通路命中数
    both_hits: int = 0           # 双路命中数
    scores: List[float] = field(default_factory=list)
    latency_ms: float = 0.0


@dataclass
class GenerationStats:
    """生成阶段统计。"""
    answer_length: int = 0
    sources_count: int = 0
    used_files: List[str] = field(default_factory=list)
    latency_ms: float = 0.0


@dataclass
class EvalReport:
    """单次评估完整报告。"""
    question: str = ""
    retrieval: Optional[RetrievalStats] = None
    generation: Optional[GenerationStats] = None
    total_latency_ms: float = 0.0


class RAGEvaluator:
    """RAG 系统性能评估器。

    使用方式：
        evaluator = RAGEvaluator()
        report = evaluator.evaluate("你最擅长什么技术？")
        evaluator.print_report(report)
    """

    def __init__(self):
        from src.retriever import HybridRetriever
        from src.prompt_loader import PromptLoader
        from src.config import settings

        self.retriever = HybridRetriever()
        self.prompt_loader = PromptLoader()
        self.settings = settings

        # 统计 chunk 分布
        self._chunk_stats = None

    # ================================================================
    # 单条评估
    # ================================================================

    def evaluate(self, question: str, run_generation: bool = False) -> EvalReport:
        """评估单条查询的 RAG 管道性能。

        Args:
            question: 用户问题
            run_generation: 是否执行 LLM 生成（默认仅评估检索）

        Returns:
            EvalReport 完整报告
        """
        report = EvalReport(question=question)

        # --- 阶段1: 检索 ---
        t0 = time.perf_counter()

        dense_raw = self.retriever._dense_search(question)
        t_dense = time.perf_counter()

        sparse_raw = self.retriever._sparse_search(question)
        t_sparse = time.perf_counter()

        fused = self.retriever.search(question)
        t_fuse = time.perf_counter()

        report.retrieval = RetrievalStats(
            query=question,
            total_results=len(fused),
            dense_hits=len(dense_raw),
            sparse_hits=len(sparse_raw),
            both_hits=sum(1 for r in fused if r.get("score_type") == "both"),
            scores=[r.get("score", 0) for r in fused],
            latency_ms=(t_fuse - t0) * 1000,
        )

        # --- 阶段2: 生成 ---
        if run_generation:
            from src.chat_engine import RAGChatEngine

            engine = RAGChatEngine()
            t_gen_start = time.perf_counter()
            try:
                result = engine.chat(question)
                t_gen_end = time.perf_counter()

                report.generation = GenerationStats(
                    answer_length=len(result["answer"]),
                    sources_count=len(result["sources"]),
                    used_files=result.get("used_files", []),
                    latency_ms=(t_gen_end - t_gen_start) * 1000,
                )
            except Exception as e:
                logger.warning("  ⚠️ 生成阶段失败: %s", e)
                report.generation = GenerationStats()

        report.total_latency_ms = (time.perf_counter() - t0) * 1000
        return report

    # ================================================================
    # 批量基准测试
    # ================================================================

    def benchmark(self, questions: Optional[List[str]] = None) -> List[EvalReport]:
        """对一组问题运行基准测试。

        Args:
            questions: 测试问题列表，默认使用 BENCHMARK_QUESTIONS

        Returns:
            所有问题的评估报告列表
        """
        if questions is None:
            questions = BENCHMARK_QUESTIONS

        logger.info("=" * 60)
        logger.info("🚀 RAG 系统性能基准测试")
        logger.info("=" * 60)
        logger.info("  测试问题: %d 个", len(questions))
        logger.info("  检索权重: Dense=%.1f / Sparse=%.1f",
                     self.settings.VECTOR_WEIGHT, self.settings.BM25_WEIGHT)
        logger.info("  Top-K: %d", self.settings.TOP_K_RETRIEVAL)
        logger.info("")

        reports = []
        for i, q in enumerate(questions, 1):
            logger.info("[%d/%d] %s", i, len(questions), q)
            report = self.evaluate(q, run_generation=False)
            reports.append(report)
            self._print_single_result(report)
            logger.info("")

        # 汇总统计
        self._print_summary(reports)
        return reports

    # ================================================================
    # 索引质量评估
    # ================================================================

    def evaluate_index(self) -> dict:
        """评估索引质量：chunk 分布、向量维度、文件覆盖等。"""
        meta_path = self.settings.VECTOR_DB_PATH / "chunks_meta.json"
        emb_path = self.settings.VECTOR_DB_PATH / "embeddings.npy"

        stats = {
            "chunk_count": 0,
            "file_count": 0,
            "embedding_dim": 0,
            "avg_chunk_length": 0,
            "max_chunk_length": 0,
            "min_chunk_length": float("inf"),
            "length_distribution": {},
            "files": [],
        }

        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                chunks = json.load(f)

            stats["chunk_count"] = len(chunks)
            lengths = [len(c["content"]) for c in chunks]
            stats["avg_chunk_length"] = int(np.mean(lengths))
            stats["max_chunk_length"] = max(lengths)
            stats["min_chunk_length"] = min(lengths)

            # 长度分布
            bins = [(0, 200), (200, 350), (350, 500), (500, float("inf"))]
            labels = ["<200", "200-350", "350-500", ">500"]
            for (lo, hi), label in zip(bins, labels):
                count = sum(1 for l in lengths if lo <= l < hi)
                stats["length_distribution"][label] = count

            # 文件统计
            files = {}
            for c in chunks:
                src = c["metadata"].get("source", "unknown")
                ftype = c["metadata"].get("file_type", "其他")
                if src not in files:
                    files[src] = {"type": ftype, "chunks": 0, "chars": 0}
                files[src]["chunks"] += 1
                files[src]["chars"] += len(c["content"])
            stats["file_count"] = len(files)
            stats["files"] = [
                {"name": k, **v} for k, v in files.items()
            ]

        if emb_path.exists():
            emb = np.load(emb_path)
            stats["embedding_dim"] = emb.shape[1]

        self._chunk_stats = stats
        return stats

    # ================================================================
    # 输出格式化
    # ================================================================

    def _print_single_result(self, report: EvalReport):
        """打印单条评估结果。"""
        r = report.retrieval
        if r is None:
            return

        logger.info(
            "  ├─ 检索: %d 条结果 | "
            "Dense:%d / Sparse:%d / Both:%d | "
            "Top1:%.4f / Avg:%.4f | "
            "⏱ %.1fms",
            r.total_results,
            r.dense_hits, r.sparse_hits, r.both_hits,
            r.scores[0] if r.scores else 0,
            np.mean(r.scores) if r.scores else 0,
            r.latency_ms,
        )

    def _print_summary(self, reports: List[EvalReport]):
        """打印批量基准汇总。"""
        n = len(reports)

        # 检索统计
        total_results = [r.retrieval.total_results for r in reports if r.retrieval]
        avg_scores = [
            np.mean(r.retrieval.scores) for r in reports
            if r.retrieval and r.retrieval.scores
        ]
        both_rates = [
            r.retrieval.both_hits / max(r.retrieval.total_results, 1)
            for r in reports if r.retrieval
        ]
        latencies = [r.retrieval.latency_ms for r in reports if r.retrieval]

        logger.info("=" * 60)
        logger.info("📊 基准测试汇总 (N=%d)", n)
        logger.info("-" * 60)
        logger.info("  平均召回数:     %.1f 条", np.mean(total_results))
        logger.info("  平均融合分数:   %.4f", np.mean(avg_scores))
        logger.info("  双路命中率:     %.1f%%", np.mean(both_rates) * 100)
        logger.info("  平均检索耗时:   %.1f ms", np.mean(latencies))
        logger.info("  检索P99耗时:    %.1f ms", np.percentile(latencies, 99) if len(latencies) > 1 else latencies[0])
        logger.info("=" * 60)

    def print_index_report(self):
        """打印索引质量报告。"""
        stats = self.evaluate_index()

        logger.info("=" * 60)
        logger.info("📦 索引质量报告")
        logger.info("-" * 60)
        logger.info("  文档块总数:      %d", stats["chunk_count"])
        logger.info("  文件数:          %d", stats["file_count"])
        logger.info("  向量维度:        %d", stats["embedding_dim"])
        logger.info("  平均块长度:      %d 字符", stats["avg_chunk_length"])
        logger.info("  最大/最小块:     %d / %d 字符",
                     stats["max_chunk_length"], stats["min_chunk_length"])
        logger.info("  长度分布:        %s", stats["length_distribution"])
        logger.info("")
        logger.info("  文件明细:")
        for f in stats["files"]:
            logger.info(
                "    [%s] %s | %d块 | %d字符",
                f["type"], f["name"], f["chunks"], f["chars"],
            )
        logger.info("=" * 60)

    # ================================================================
    # 高级评估指标
    # ================================================================

    def run_extended_benchmark(self, test_file: Optional[str] = None) -> dict:
        """从测试文件加载问题集并运行扩展基准测试。

        Args:
            test_file: 测试文件路径，默认 tests/test_questions.json

        Returns:
            包含所有报告和高级指标的字典
        """
        if test_file is None:
            test_file = str(
                self.settings.PROJECT_ROOT / "tests" / "test_questions.json"
            )

        with open(test_file, "r", encoding="utf-8") as f:
            test_data = json.load(f)

        questions = [q["question"] for q in test_data["questions"]]
        q_meta = {q["question"]: q for q in test_data["questions"]}

        logger.info("=" * 60)
        logger.info("🚀 RAG 扩展基准测试 (%d 题)", len(questions))
        logger.info("   权重预设: %s", self.settings.ACTIVE_WEIGHT_PRESET)
        logger.info("   权重: Dense=%.1f / Sparse=%.1f",
                     self.settings.VECTOR_WEIGHT, self.settings.BM25_WEIGHT)
        logger.info("")

        reports = []
        for i, q in enumerate(questions, 1):
            report = self.evaluate(q, run_generation=False)
            reports.append(report)
            r = report.retrieval
            logger.info(
                "[%2d/%d] %s → %d条 | D:%d/S:%d | Top1:%.3f | %s",
                i, len(questions), q[:30], r.total_results,
                r.dense_hits, r.sparse_hits,
                r.scores[0] if r.scores else 0,
                q_meta[q]["category"],
            )

        # 计算高级指标
        metrics = self._calc_advanced_metrics(reports, test_data)
        self._print_extended_summary(metrics)

        return {
            "config": {
                "weight_preset": self.settings.ACTIVE_WEIGHT_PRESET,
                "vector_weight": self.settings.VECTOR_WEIGHT,
                "bm25_weight": self.settings.BM25_WEIGHT,
                "top_k": self.settings.TOP_K_RETRIEVAL,
                "chunk_size": self.settings.CHUNK_SIZE,
                "chunk_overlap": self.settings.CHUNK_OVERLAP,
            },
            "index_stats": self.evaluate_index(),
            "reports": [self.to_dict(r) for r in reports],
            "metrics": metrics,
        }

    def _calc_advanced_metrics(self, reports: List[EvalReport],
                                test_data: dict) -> dict:
        """计算高级评估指标：Hit Rate@K, MRR, 按类别统计。

        采用基于 expected_keywords 的严格匹配判断，
        检索内容中至少包含 1 个期望关键词才算命中。

        Args:
            reports: 评估报告列表
            test_data: 测试数据（含 expected_keywords）

        Returns:
            高级指标字典
        """
        questions_meta = {q["question"]: q for q in test_data["questions"]}

        # ---- Hit Rate@K: Top-K 结果中有任意一条内容包含期望关键词 ----
        hit_rate = {}
        for k in [1, 3, 5]:
            hits = 0
            valid = 0
            for report in reports:
                meta = questions_meta.get(report.question, {})
                expected_kw = meta.get("expected_keywords", [])
                if not expected_kw:
                    continue
                valid += 1

                # 检查 Top-K 检索内容的实际匹配
                if not report.retrieval or report.retrieval.total_results == 0:
                    continue

                # 获取 Top-K 结果的文本
                top_k_content = ""
                for idx in range(min(k, report.retrieval.total_results)):
                    # 从 reports 本身的检索信息无法直接拿到 content
                    # 我们用 scores 中的信息来近似
                    pass

                # 直接用检索器重新获取内容来判断
                hit = self._check_keyword_hit(report, expected_kw, k)
                if hit:
                    hits += 1

            hit_rate[f"hit_rate@{k}"] = round(hits / max(valid, 1), 4)

        # ---- MRR: 基于关键词匹配的第一个相关位置 ----
        mrr_sum = 0.0
        valid = 0
        for report in reports:
            meta = questions_meta.get(report.question, {})
            expected_kw = meta.get("expected_keywords", [])
            if not expected_kw:
                continue
            valid += 1

            rank_found = self._find_first_relevant_rank(report, expected_kw)
            if rank_found > 0:
                mrr_sum += 1.0 / rank_found

        mrr = round(mrr_sum / max(valid, 1), 4)

        # ---- 按类别统计 ----
        by_category = {}
        for report in reports:
            meta = questions_meta.get(report.question, {})
            cat = meta.get("category", "其他")
            if cat not in by_category:
                by_category[cat] = {"reports": [], "hits": 0, "total": 0}
            by_category[cat]["reports"].append(report)

        category_stats = {}
        for cat, data in by_category.items():
            cat_reports = data["reports"]
            scores = [
                r.retrieval.scores[0] if r.retrieval and r.retrieval.scores else 0
                for r in cat_reports
            ]
            avg_both = np.mean([
                r.retrieval.both_hits for r in cat_reports if r.retrieval
            ])
            category_stats[cat] = {
                "count": len(cat_reports),
                "avg_top1_score": round(float(np.mean(scores)), 4),
                "avg_results": round(float(np.mean([
                    r.retrieval.total_results for r in cat_reports if r.retrieval
                ])), 1),
                "avg_both_hits": round(float(avg_both), 1),
                "avg_sparse_hits": round(float(np.mean([
                    r.retrieval.sparse_hits for r in cat_reports if r.retrieval
                ])), 1),
            }

        # 全局统计
        all_both = [r.retrieval.both_hits for r in reports if r.retrieval]
        all_sparse = [r.retrieval.sparse_hits for r in reports if r.retrieval]
        all_top1 = [r.retrieval.scores[0] for r in reports if r.retrieval and r.retrieval.scores]
        all_latency = [r.retrieval.latency_ms for r in reports if r.retrieval]

        return {
            **hit_rate,
            "mrr": mrr,
            "total_questions": len(reports),
            "avg_top1_score": round(float(np.mean(all_top1)), 4),
            "avg_both_hits": round(float(np.mean(all_both)), 1),
            "avg_sparse_hits": round(float(np.mean(all_sparse)), 1),
            "avg_latency_ms": round(float(np.mean(all_latency)), 1),
            "by_category": category_stats,
        }

    def _check_keyword_hit(self, report: EvalReport,
                            keywords: List[str], k: int) -> bool:
        """检查 Top-K 检索结果中是否包含期望关键词。"""
        if not report.retrieval or report.retrieval.total_results == 0:
            return False
        # 取 Top-K 的分数来判断（用 chunk 无关，用 question 匹配）
        # 实际上我们检查 retrieved 内容
        results = self.retriever.search(report.question, top_k=k)
        for r in results:
            content_lower = r.get("content", "").lower()
            if any(kw.lower() in content_lower for kw in keywords):
                return True
        return False

    def _find_first_relevant_rank(self, report: EvalReport,
                                   keywords: List[str]) -> int:
        """找到第一个包含期望关键词的检索结果排名（1-based），未找到返回 0。"""
        if not report.retrieval:
            return 0
        results = self.retriever.search(report.question, top_k=10)
        for rank, r in enumerate(results, 1):
            content_lower = r.get("content", "").lower()
            if any(kw.lower() in content_lower for kw in keywords):
                return rank
        return 0

    def _print_extended_summary(self, metrics: dict):
        """打印扩展基准测试汇总。"""
        logger.info("")
        logger.info("=" * 60)
        logger.info("📊 扩展基准测试汇总")
        logger.info("-" * 60)
        logger.info("  总题数:           %d", metrics["total_questions"])
        logger.info("  Top1 平均分数:    %.4f", metrics.get("avg_top1_score", 0))
        logger.info("  Hit Rate@1:       %.1f%%", metrics.get("hit_rate@1", 0) * 100)
        logger.info("  Hit Rate@3:       %.1f%%", metrics.get("hit_rate@3", 0) * 100)
        logger.info("  Hit Rate@5:       %.1f%%", metrics.get("hit_rate@5", 0) * 100)
        logger.info("  MRR:              %.4f", metrics.get("mrr", 0))
        logger.info("  平均双路命中:     %.1f 条", metrics.get("avg_both_hits", 0))
        logger.info("  平均Sparse召回:   %.1f 条", metrics.get("avg_sparse_hits", 0))
        logger.info("  平均检索耗时:     %.1f ms", metrics.get("avg_latency_ms", 0))
        logger.info("")
        logger.info("  按类别统计:")
        for cat, stats in metrics.get("by_category", {}).items():
            logger.info(
                "    %-10s  %2d题  Top1=%.3f  Sparse=%.1f条  Both=%.1f条",
                cat, stats["count"], stats["avg_top1_score"],
                stats["avg_sparse_hits"], stats["avg_both_hits"],
            )
        logger.info("=" * 60)

    def compare_presets(self, presets: Optional[List[str]] = None) -> List[dict]:
        """对比不同权重预设的检索效果。

        Args:
            presets: 要对比的预设列表，默认 ["balanced", "semantic_first", "keyword_first"]

        Returns:
            每个预设的指标汇总列表
        """
        if presets is None:
            presets = ["balanced", "semantic_first", "keyword_first"]

        results = []
        for preset_name in presets:
            logger.info("\n\n🔬 测试预设: %s", preset_name)
            self.settings.set_weight_preset(preset_name)
            result = self.run_extended_benchmark()
            results.append(result)

        # 打印对比表
        self._print_comparison_table(results, presets)
        return results

    def _print_comparison_table(self, results: List[dict], presets: List[str]):
        """打印优化前后对比表。"""
        logger.info("\n\n")
        logger.info("=" * 72)
        logger.info("📊 权重预设对比表")
        logger.info("=" * 72)

        header = f"{'指标':<20}"
        for p in presets:
            header += f" {p:>16}"
        logger.info(header)
        logger.info("-" * 72)

        metrics_map = [
            ("Top1 平均分数", "avg_top1_score", ".4f"),
            ("Hit Rate@1", "hit_rate@1", ".1%"),
            ("Hit Rate@3", "hit_rate@3", ".1%"),
            ("Hit Rate@5", "hit_rate@5", ".1%"),
            ("MRR", "mrr", ".4f"),
            ("平均双路命中(条)", "avg_both_hits", ".1f"),
            ("平均Sparse召回(条)", "avg_sparse_hits", ".1f"),
            ("平均耗时(ms)", "avg_latency_ms", ".1f"),
        ]

        for label, key, fmt in metrics_map:
            row = f"{label:<20}"
            for r in results:
                val = r["metrics"].get(key, 0)
                if fmt == ".1%":
                    row += f" {val*100:>15.1f}%"
                else:
                    row += f" {val:>16.4f}"
            logger.info(row)

        logger.info("-" * 72)
        logger.info("  💡 推荐选择分数最高的预设作为生产配置")
        logger.info("=" * 72)

    # ================================================================
    # LLM-as-a-Judge 评估
    # ================================================================

    JUDGE_PROMPT = """你是一个严格的 RAG 系统评估专家。请根据以下【参考资料】和【用户问题】，对【AI 回答】进行三维打分。

## 评分维度（1-5 分，整数）

1. **准确性（Accuracy）**：回答是否与参考资料一致？有无编造事实或幻觉？
   - 5: 完全基于资料，事实无错误
   - 3: 部分与资料一致，有轻微偏差
   - 1: 大量编造或与资料矛盾

2. **相关性（Relevance）**：回答是否直接回应了用户问题？
   - 5: 完全切题，无冗余
   - 3: 部分切题，有少量无关内容
   - 1: 答非所问

3. **完整性（Completeness）**：回答是否覆盖了参考资料中的关键信息？
   - 5: 关键信息点全覆盖
   - 3: 覆盖了部分关键信息
   - 1: 遗漏了大部分关键信息

## 输出格式（严格 JSON，不要有其他文字）

```json
{{"accuracy": 4, "relevance": 5, "completeness": 3, "comment": "简短评语（20字以内）"}}
```

---
【参考资料】
{context}

---
【用户问题】
{question}

---
【AI 回答】
{answer}

---
请评分："""

    def run_llm_judge(self, test_file: str | None = None,
                       sample_size: int = 5, enable_retry: bool = False) -> dict:
        """使用 LLM-as-a-Judge 评估 RAG 端到端回答质量。

        从测试集中抽取 sample_size 个问题，走完整 RAG 管道生成答案，
        然后由 LLM Judge 对答案进行准确性/相关性/完整性三维打分。

        Args:
            test_file: 测试文件路径
            sample_size: 评估问题数（建议 5-10 题，避免 LLM 调用过多）

        Returns:
            {"scores": [...], "avg_accuracy": 0, "avg_relevance": 0, "avg_completeness": 0}
        """
        from src.chat_engine import RAGChatEngine
        from src.config import settings
        from openai import OpenAI

        # 加载测试问题
        if test_file is None:
            test_file = str(
                settings.PROJECT_ROOT / "tests" / "test_questions.json"
            )
        with open(test_file, "r", encoding="utf-8") as f:
            test_data = json.load(f)

        questions = [q for q in test_data["questions"]]
        if len(questions) > sample_size:
            import random
            random.seed(42)
            questions = random.sample(questions, sample_size)

        logger.info("=" * 60)
        logger.info("🧑‍⚖️ LLM-as-a-Judge 评估 (%d 题)", len(questions))
        logger.info("   Judge 模型: %s", settings.ACTIVE_PROVIDER)
        logger.info("=" * 60)
        logger.info("")

        engine = RAGChatEngine()
        llm_config = settings.get_llm_config()
        client = OpenAI(base_url=llm_config["base_url"], api_key=llm_config["api_key"])

        all_scores = []
        for i, q in enumerate(questions, 1):
            question = q["question"]
            category = q.get("category", "其他")

            # Step 1: RAG 管道生成答案
            try:
                result = engine.chat(question)
                answer = result["answer"]
                sources = result.get("sources", [])
                context = "\n\n---\n\n".join([
                    f"[来源: {s['source']}]\n{s['content'][:300]}"
                    for s in sources[:3]
                ])
            except Exception as e:
                logger.warning("  [%d/%d] ❌ RAG 生成失败: %s", i, len(questions), e)
                continue

            # Step 2: LLM Judge 打分
            judge_prompt = self.JUDGE_PROMPT.format(
                context=context or "（无参考资料）",
                question=question,
                answer=answer[:1200],
            )

            try:
                resp = client.chat.completions.create(
                    model=llm_config["model"],
                    messages=[{"role": "user", "content": judge_prompt}],
                    temperature=0.0,
                    timeout=30,
                )
                judge_text = resp.choices[0].message.content.strip()

                # 提取 JSON（处理 ```json ... ``` 包裹）
                if "```json" in judge_text:
                    judge_text = judge_text.split("```json")[1].split("```")[0]
                elif "```" in judge_text:
                    judge_text = judge_text.split("```")[1].split("```")[0]
                scores = json.loads(judge_text)

            except Exception as e:
                logger.warning("  [%d/%d] ⚠️ Judge 评分失败: %s", i, len(questions), e)
                scores = {"accuracy": 0, "relevance": 0, "completeness": 0, "comment": str(e)}

            retry_scores = None
            # ── Self-Reflection 重试机制 ──
            if enable_retry and (
                scores.get("accuracy", 5) < 3 or scores.get("relevance", 5) < 4
            ):
                feedback = scores.get("comment", "回答质量不足，请重新生成")
                logger.info("  🔄 重试中… (原因: %s)", feedback)
                try:
                    retry_answer = client.chat.completions.create(
                        model=llm_config["model"],
                        messages=[
                            {"role": "system", "content": engine.prompt_loader.build_system_prompt(
                                context=context or "", question=question,
                            )},
                            {"role": "user", "content": question},
                            {"role": "assistant", "content": answer},
                            {"role": "user", "content": f"以上回答存在问题：{feedback}\n请重新回答，严格基于参考资料，不要编造任何信息。"},
                        ],
                        temperature=self.settings.TEMPERATURE,
                        timeout=self.settings.LLM_TIMEOUT,
                    ).choices[0].message.content

                    # 再次评分
                    retry_prompt = self.JUDGE_PROMPT.format(
                        context=context or "（无参考资料）",
                        question=question,
                        answer=retry_answer[:1200],
                    )
                    retry_resp = client.chat.completions.create(
                        model=llm_config["model"],
                        messages=[{"role": "user", "content": retry_prompt}],
                        temperature=0.0, timeout=30,
                    )
                    retry_text = retry_resp.choices[0].message.content.strip()
                    if "```json" in retry_text:
                        retry_text = retry_text.split("```json")[1].split("```")[0]
                    elif "```" in retry_text:
                        retry_text = retry_text.split("```")[1].split("```")[0]
                    retry_scores = json.loads(retry_text)
                    retry_scores["retry_comment"] = retry_scores.get("comment", "")
                except Exception as e:
                    logger.warning("  ⚠️ 重试失败: %s", e)

            score_entry = {
                "question": question[:50],
                "category": category,
                "answer_preview": answer[:100],
                **scores,
            }
            if retry_scores:
                score_entry["retry"] = {
                    "accuracy": retry_scores.get("accuracy", 0),
                    "relevance": retry_scores.get("relevance", 0),
                    "completeness": retry_scores.get("completeness", 0),
                }
                logger.info(
                    "  📈 重试结果: 准确:%d→%d 相关:%d→%d 完整:%d→%d",
                    scores.get("accuracy", 0), retry_scores.get("accuracy", 0),
                    scores.get("relevance", 0), retry_scores.get("relevance", 0),
                    scores.get("completeness", 0), retry_scores.get("completeness", 0),
                )
            all_scores.append(score_entry)

            logger.info(
                "  [%d/%d] %s | 准确:%d 相关:%d 完整:%d | %s",
                i, len(questions), question[:30],
                scores.get("accuracy", 0),
                scores.get("relevance", 0),
                scores.get("completeness", 0),
                scores.get("comment", ""),
            )

        # 汇总统计
        if not all_scores:
            logger.warning("⚠️ 无有效评分结果")
            return {"scores": [], "avg_accuracy": 0, "avg_relevance": 0, "avg_completeness": 0}

        avg_acc = np.mean([s["accuracy"] for s in all_scores])
        avg_rel = np.mean([s["relevance"] for s in all_scores])
        avg_cmp = np.mean([s["completeness"] for s in all_scores])
        overall = round((avg_acc + avg_rel + avg_cmp) / 3, 2)

        # 重试统计
        retried = [s for s in all_scores if "retry" in s]
        retry_avg = {"acc": 0, "rel": 0, "cmp": 0}
        if retried:
            retry_avg["acc"] = np.mean([s["retry"]["accuracy"] for s in retried])
            retry_avg["rel"] = np.mean([s["retry"]["relevance"] for s in retried])
            retry_avg["cmp"] = np.mean([s["retry"]["completeness"] for s in retried])

        logger.info("")
        logger.info("=" * 60)
        logger.info("📊 LLM-as-a-Judge 评估汇总 (%d 题)", len(all_scores))
        logger.info("-" * 60)
        logger.info("  平均准确性:   %.2f / 5", avg_acc)
        logger.info("  平均相关性:   %.2f / 5", avg_rel)
        logger.info("  平均完整性:   %.2f / 5", avg_cmp)
        logger.info("  综合均分:     %.2f / 5", overall)
        if retried:
            logger.info("-" * 60)
            logger.info("  🔄 重试统计 (%d 题触发):", len(retried))
            logger.info("    重试后准确性: %.2f / 5", retry_avg["acc"])
            logger.info("    重试后相关性: %.2f / 5", retry_avg["rel"])
            logger.info("    重试后完整性: %.2f / 5", retry_avg["cmp"])
        logger.info("=" * 60)

        return {
            "scores": all_scores,
            "avg_accuracy": round(avg_acc, 2),
            "avg_relevance": round(avg_rel, 2),
            "avg_completeness": round(avg_cmp, 2),
            "overall": overall,
            "retry_count": len(retried),
            "retry_avg_accuracy": round(retry_avg["acc"], 2) if retried else None,
            "retry_avg_relevance": round(retry_avg["rel"], 2) if retried else None,
            "retry_avg_completeness": round(retry_avg["cmp"], 2) if retried else None,
        }

    def to_dict(self, report: EvalReport) -> dict:
        """将评估报告转为可序列化字典。"""
        return {
            "question": report.question,
            "total_latency_ms": round(report.total_latency_ms, 1),
            "retrieval": {
                "total_results": report.retrieval.total_results,
                "dense_hits": report.retrieval.dense_hits,
                "sparse_hits": report.retrieval.sparse_hits,
                "both_hits": report.retrieval.both_hits,
                "top_score": round(report.retrieval.scores[0], 4) if report.retrieval.scores else 0,
                "avg_score": round(np.mean(report.retrieval.scores), 4) if report.retrieval.scores else 0,
                "latency_ms": round(report.retrieval.latency_ms, 1),
            } if report.retrieval else None,
            "generation": {
                "answer_length": report.generation.answer_length,
                "sources_count": report.generation.sources_count,
                "used_files": report.generation.used_files,
                "latency_ms": round(report.generation.latency_ms, 1),
            } if report.generation else None,
        }


# ============================================================
# CLI 入口
# ============================================================

if __name__ == "__main__":
    evaluator = RAGEvaluator()

    if "--judge" in sys.argv:
        # LLM-as-a-Judge 评估模式（--retry 启用 Self-Reflection 重试）
        enable_retry = "--retry" in sys.argv
        evaluator.run_llm_judge(sample_size=5, enable_retry=enable_retry)

    elif "--compare" in sys.argv:
        # 权重预设对比模式
        evaluator.compare_presets()

    elif "--extended" in sys.argv:
        # 扩展基准测试模式（20题 + 高级指标）
        evaluator.run_extended_benchmark()

    elif "--question" in sys.argv:
        # 单条评估
        evaluator.print_index_report()
        print()
        idx = sys.argv.index("--question")
        q = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else "自我介绍"
        report = evaluator.evaluate(q, run_generation="--gen" in sys.argv)
        print(json.dumps(evaluator.to_dict(report), ensure_ascii=False, indent=2))

    else:
        # 默认：索引报告 + 5题基准
        evaluator.print_index_report()
        print()
        evaluator.benchmark()

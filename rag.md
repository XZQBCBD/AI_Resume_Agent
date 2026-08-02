# RAG 系统技术栈详解

> 本文档详细阐述 AI 简历分身项目的 RAG（Retrieval-Augmented Generation）系统的技术架构、技术选型及设计决策。

---

## 目录

1. [总体架构概览](#1-总体架构概览)
2. [嵌入模型](#2-嵌入模型)
3. [向量存储](#3-向量存储)
4. [文档加载与切分](#4-文档加载与切分)
5. [检索策略](#5-检索策略)
6. [LLM 生成层](#6-llm-生成层)
7. [RAG 管道编排](#7-rag-管道编排)
8. [缓存与优化层](#8-缓存与优化层)
9. [技术选型总结](#9-技术选型总结)

---

## 1. 总体架构概览

```
┌──────────────────────────────────────────────────────────────┐
│                        用户提问                               │
└─────────────────────┬────────────────────────────────────────┘
                      ▼
┌──────────────────────────────────────────────────────────────┐
│                    HybridRetriever                            │
│  ┌─────────────────────┐   ┌─────────────────────┐           │
│  │  Dense 通路          │   │  Sparse 通路         │           │
│  │  SentenceTransformer │   │  jieba 分词 + BM25   │           │
│  │  余弦相似度          │   │  Okapi 评分          │           │
│  │  → Top-10 候选       │   │  → Top-10 候选       │           │
│  └─────────┬───────────┘   └─────────┬───────────┘           │
│            └──────────┬──────────────┘                       │
│                       ▼                                      │
│              RRF 倒数排名融合                                  │
│              → Top-8 最终结果                                  │
└─────────────────────┬────────────────────────────────────────┘
                      ▼
┌──────────────────────────────────────────────────────────────┐
│                     Prompt 增强                               │
│  TOML 模板 + {context} + {question} → 完整 System Prompt       │
└─────────────────────┬────────────────────────────────────────┘
                      ▼
┌──────────────────────────────────────────────────────────────┐
│                     LLM 生成                                  │
│  DeepSeek Chat (默认) / GPT-4o / 自定义 OpenAI 兼容 API       │
│  temperature=0.1                                              │
└─────────────────────┬────────────────────────────────────────┘
                      ▼
                  答案 + 来源
```

本项目没有使用 LangChain 或 LlamaIndex 等 RAG 框架，而是基于底层库完全从零构建了 RAG 管道。这是本项目最核心的设计决策——**全栈自研**，以获得对每个环节的完全可控性。

---

## 2. 嵌入模型

### 2.1 模型选型

**当前使用：`sentence-transformers/all-MiniLM-L6-v2`**

配置位置：`src/config.py:147-149`

```python
EMBEDDING_MODEL: str = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
```

加载位置：`src/retriever.py:46-50`、`src/build_index.py:181-185`

### 2.2 模型特性

| 属性 | 值 |
|------|-----|
| 模型架构 | MiniLM（基于 BERT 的知识蒸馏轻量模型） |
| 向量维度 | 384 |
| 最大序列长度 | 256 tokens |
| 模型大小 | ~80 MB |
| 中文支持 | 通过多语言预训练支持中文 |
| 本地运行 | ✅ 完全本地，无需 API 调用 |

### 2.3 选型理由

1. **轻量高效**：80MB 的模型大小意味着启动快、内存占用低，适合本地部署场景。384 维的向量比主流 768/1024 维模型缩减了一半以上的存储和计算开销。

2. **本地部署**：无需调用外部 Embedding API（如 OpenAI text-embedding-ada-002），零网络延迟，零 API 费用，数据隐私完全可控。

3. **中英文兼顾**：all-MiniLM-L6-v2 虽然以英文为主，但在多语言场景下表现尚可。配合同项目的 BM25 关键词检索（jieba 中文分词），可以有效弥补纯语义检索在中文精确匹配上的不足。

4. **生态成熟**：sentence-transformers 库对 HuggingFace 生态兼容良好，模型加载、编码、缓存都开箱即用。

### 2.4 模型缓存

配置位置：`src/config.py:102-108, 194-200`

项目强制将 HuggingFace 缓存路径劫持到 `E:/AI_Cache/huggingface`，通过设置以下环境变量：

```
HF_HOME = E:/AI_Cache/huggingface
TRANSFORMERS_CACHE = E:/AI_Cache/huggingface
HUGGINGFACE_HUB_CACHE = E:/AI_Cache/huggingface
SENTENCE_TRANSFORMERS_HOME = E:/AI_Cache/huggingface
```

**原因**：避免模型文件下载到系统盘（C 盘），防止磁盘空间耗尽，同时便于模型复用和迁移。

---

## 3. 向量存储

### 3.1 当前方案：NumPy 本地文件存储

**不使用任何外部向量数据库。**

核心实现：
- `src/build_index.py:223-261` → 向量写入 `vector_db/embeddings.npy`
- `src/retriever.py:349-363` → 向量通过 `numpy.load()` 直接加载到内存

存储文件清单：

| 文件 | 内容 |
|------|------|
| `vector_db/embeddings.npy` | 原始 NumPy 嵌入矩阵（float32） |
| `vector_db/embedding_ids.json` | 嵌入 ID 列表（与 embeddings.npy 行对应） |
| `vector_db/chunks_meta.json` | Chunk 完整元数据（source、file_type、section_type 等） |
| `vector_db/file_hashes.json` | 源文件 MD5 哈希（变更检测用） |
| `vector_db/bm25_index.pkl` | BM25 索引（pickle 序列化） |

### 3.2 相似度计算

`src/retriever.py:119-123`，直接使用 NumPy 内积计算余弦相似度：

```python
similarities = np.dot(normalized_query, normalized_embeddings.T)
```

由于向量在索引构建时已经 L2 归一化，点积即等价于余弦相似度。

### 3.3 为何不使用 ChromaDB / FAISS / Milvus

ChromaDB 原在 `requirements.txt` 中列出（`chromadb>=0.4.18`）但**实际代码未使用**。

**原因**（来源：项目文档中的踩坑记录）：
- ChromaDB 依赖的 **SQLite3 在 Windows 上存在兼容性问题**（需要特定版本的 `sqlite3.dll`，Python 自带的版本不满足要求）
- 对于本项目的规模（数十份个人文档，数百个 chunk），**无需专用向量数据库的开销**
- NumPy 矩阵内积检索在几千个向量的规模下完全够用（毫秒级）
- 减少了依赖复杂度，降低了部署门槛

这是一个务实的工程决策：**当数据规模不需要分布式/持久化向量数据库时，NumPy 内存方案是最简单可靠的。**

---

## 4. 文档加载与切分

### 4.1 文档加载

实现文件：`src/loader.py`

采用**策略模式**按文件后缀选择解析器：

| 格式 | 解析器 | 底层库 | 备注 |
|------|--------|--------|------|
| `.pdf` | `PDFParser` | pdfplumber → pypdf（回退） → pytesseract OCR（最终回退） | 三级降级策略保证鲁棒性 |
| `.docx` | `DocxParser` | python-docx | 直接提取段落文本 |
| `.md` | `PlainTextParser` | 原生文件读取 | — |
| `.txt` | `PlainTextParser` | 原生文件读取 | — |

**三级 PDF 解析降级**（`src/loader.py:113-285`）：
1. **pdfplumber**（首选）：对中文 PDF 兼容性最好，能正确提取中文字符
2. **pypdf**（回退）：当 pdfplumber 解析失败时使用
3. **pytesseract OCR**（最终回退）：针对扫描件/图片型 PDF，使用 OCR 提取文字

### 4.2 分块策略

实现类：`TextChunker`（`src/loader.py:406-574`）

**核心参数**（`src/config.py:138-140`）：

| 参数 | 值 | 说明 |
|------|-----|------|
| `CHUNK_SIZE` | 500 字符 | 最大 chunk 大小 |
| `CHUNK_OVERLAP` | 100 字符 | 相邻 chunk 重叠区域 |
| `MIN_CHUNK_SIZE` | 200 字符 | 低于此值的 chunk 合并处理 |

**分块管道**（五阶段）：

```
原始文档
  │
  ▼
[1. 结构切分] _split_by_structure()
  识别列表项、标题行 → 在语义边界处切分
  再按空行（段落）二次切分
  │
  ▼
[2. 长段落切分] _split_long_paragraph()
  超过 500 字符的段落，在句子边界（。！？.!?\n）处切分
  中英文句子边界正则均支持
  │
  ▼
[3. 过滤]
  移除纯空白 chunk
  │
  ▼
[4. 短 chunk 合并] _merge_short_chunks()
  < 200 字符的 chunk 与相邻 chunk 合并或借取文本
  │
  ▼
[5. 重叠添加] _add_overlap_between_chunks()
  前一个 chunk 末尾 100 字符添加到下一个 chunk 开头
  │
  ▼
最终 Chunk 列表
```

### 4.3 元数据注解

每个 chunk 携带丰富的元数据（`src/loader.py:669-691`）：

| 字段 | 来源 | 示例 |
|------|------|------|
| `source` | 文件名 | `02_项目_AI数字分身.md` |
| `file_type` | 文件名前缀推断 | `项目` (00=个人总结, 01=简历, 02=项目, 03=博客) |
| `section_type` | 内容检测 | `project`, `skills`, `education`, `experience`, `self_intro`, `awards`, `contact` |
| `chunk_index` | 顺序编号 | `3` |
| `total_chunks` | 该文档 chunk 总数 | `15` |

`section_type` 通过 `detect_section_type()` 函数（`src/loader.py:334-354`）基于关键词匹配检测，支持 9 种语义章节类型。

### 4.4 选型理由

1. **500 字符的 chunk 大小**是经过实验的平衡点：太小则语义不完整（尤其中文简历中一个项目描述往往 200-500 字），太大则检索精度下降。
2. **基于结构的切分**优先于固定大小切分：尊重文档的自然结构（段落、标题、列表），避免在句子中间截断。
3. **100 字符重叠**确保跨 chunk 边界的上下文不被丢失。
4. **丰富元数据**为后续可能的过滤、排序、来源标注提供了基础。

---

## 5. 检索策略

### 5.1 双路混合检索架构

实现文件：`src/retriever.py`

```
用户查询
    │
    ├──────────────────────────┐
    ▼                          ▼
┌───────────────┐       ┌──────────────┐
│  Dense 通路    │       │  Sparse 通路  │
│  (语义检索)    │       │  (关键词检索)  │
└───────┬───────┘       └──────┬───────┘
        ▼                      ▼
  向量编码 +              jieba 分词 +
  余弦相似度              BM25 Okapi
        │                      │
        ▼                      ▼
   Top-10 候选             Top-10 候选
        │                      │
        └──────────┬───────────┘
                   ▼
           RRF 倒数排名融合
           k=60
                   │
                   ▼
           Top-8 最终结果
        (含归一化混合分数)
```

### 5.2 Dense 通路（语义检索）

`src/retriever.py:97-147`

- 使用 SentenceTransformer 将查询编码为 384 维向量
- 与预计算的全量 chunk 嵌入矩阵做点积（余弦相似度）
- 返回 Top-10 候选项（由 `TOP_K_CANDIDATE` 配置）

### 5.3 Sparse 通路（BM25 关键词检索）

`src/retriever.py:149-195`

**分词**：jieba 分词 + 自定义词典 + 停用词过滤

**自定义词典**（`src/bm25_utils.py:16-49`）：包含 90+ 条简历领域专业术语，如：
- 技术栈：`Spring Boot`、`React`、`Kubernetes`、`TensorFlow`
- 中文术语：`微服务`、`高并发`、`系统架构`、`性能优化`

**停用词**（`src/bm25_utils.py:56-81`）：约 130 个中英文停用词（的、了、the、a、is 等），减少噪声。

**低分过滤**：`BM25_SCORE_THRESHOLD = 0.05`（`src/config.py:125`），过滤相关性极低的候选项。

### 5.4 RRF 倒数排名融合

`src/retriever.py:197-296`

融合公式：
```
RRF_score(d) = Σ 1/(k + rank_i(d))
其中 k = 60，rank_i 表示文档 d 在第 i 个通路中的排名
```

**最终公开分数**（`src/retriever.py:263-281`）：
1. 对原始 Dense 分数和 BM25 分数分别做 Min-Max 归一化
2. 按权重加权求和：

```
final_score = VECTOR_WEIGHT × dense_norm + BM25_WEIGHT × bm25_norm
```

默认权重：`VECTOR_WEIGHT = 0.6`，`BM25_WEIGHT = 0.4`

### 5.5 权重预设

`src/config.py:128-135, 248-266`

支持运行时切换五种检索策略：

| 预设名称 | Dense 权重 | Sparse 权重 | 适用场景 |
|----------|------------|-------------|----------|
| `balanced` | 0.5 | 0.5 | 通用场景 |
| `semantic_first` | 0.6 | 0.4 | 语义理解优先（默认） |
| `keyword_first` | 0.4 | 0.6 | 精确关键词匹配优先 |
| `dense_only` | 1.0 | 0.0 | 纯语义检索 |
| `sparse_only` | 0.0 | 1.0 | 纯关键词检索 |

### 5.6 索引一致性校验

`src/retriever.py:389-433`

每次检索前自动验证 embeddings、BM25 和 chunks_meta 的记录数是否一致。发现不一致时裁剪到安全的最小值并输出警告，建议通过 `python -m src.build_index --full` 重建。

### 5.7 选型理由

1. **混合检索是本项目的核心竞争力之一**：纯语义检索（Dense）擅长捕获同义词和语义相近的内容，但对精确术语匹配（如"Spring Cloud Gateway"）容易漏检；纯关键词检索（BM25）对于技术术语精确匹配优秀，但无法理解"项目经验"和"工作经历"之间的语义关联。双路融合互补。
2. **RRF 融合优于简单的分数加权**：不同通路的分数分布差异很大（余弦相似度 vs BM25 分数），直接加权无意义。RRF 基于排名融合，消除了分数尺度差异。
3. **jieba 分词 + 自定义词典**：中文分词对技术术语的处理是薄弱环节（如"SpringBoot"可能被切为"spring"和"boot"），通过自定义词典保证了简历领域关键词的完整识别。
4. **权重预设系统**提供了灵活的 A/B 测试能力，可以通过评估器（`src/evaluator.py`）对比不同策略的检索质量。

---

## 6. LLM 生成层

### 6.1 模型配置

配置位置：`src/config.py:151-170`

| Provider | 模型 | API 地址 | 环境变量 |
|----------|------|----------|----------|
| **deepseek**（默认） | `deepseek-chat` | `https://api.deepseek.com/v1` | `DEEPSEEK_API_KEY` |
| openai | `gpt-4o` | `https://api.openai.com/v1` | `OPENAI_API_KEY` |
| custom | 自定义 | `CUSTOM_BASE_URL` | `CUSTOM_API_KEY` |

生成参数：`TEMPERATURE = 0.1`（`src/config.py:143`）

### 6.2 调用方式

`src/chat_engine.py:39-49`：使用 `openai.OpenAI` Python SDK 通过 OpenAI 兼容 API 调用。所有 provider 均使用同样的接口格式，通过切换 `base_url` 和 `api_key` 实现。

运行时可在 Streamlit 侧边栏（`app/components/sidebar.py:20-33`）动态切换 LLM provider。

### 6.3 选型理由

1. **DeepSeek 作为默认 provider**：
   - 中文理解和生成能力业界领先
   - API 价格远低于 GPT-4o（约 1/10）
   - 支持 OpenAI 兼容 API 格式，迁移成本极低
   - 上下文窗口 128K，足够容纳大量检索结果

2. **OpenAI 兼容 API 统一接口**：不依赖特定 LLM provider 的 SDK，未来切换模型只需修改配置，无需改代码。

3. **temperature=0.1**：作为简历问答系统，需要输出稳定、一致、减少随机性。极低的 temperature 确保同一问题反复询问可获得相似回答。

4. **为何不用本地模型**：虽然嵌入模型选择了本地部署，但生成模型需要较强的推理和语言组织能力（如按 STAR 法则组织回答），本地可运行的开源模型（如 Qwen、ChatGLM）在当时的能力尚不足以稳定满足需求。

---

## 7. RAG 管道编排

### 7.1 核心类：`RAGChatEngine`

实现文件：`src/chat_engine.py`，核心方法 `chat()`（第 51-123 行）

```
chat(question)
    │
    ▼
[1. 检索] HybridRetriever.search(question)
    混合检索 → RRF 融合 → Top-8 Chunk
    │
    ▼
[2. 增强] 构建上下文文本
    格式: "[来源: {filename} ({file_type})]\n{content}"
    以 "---" 分隔多个 chunk
    │
    ▼
[3. Prompt 组装] PromptLoader.build_system_prompt(context, question)
    填充 TOML 模板中的 {role}, {rules}, {context}, {question}
    │
    ▼
[4. LLM 调用] client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": filled_prompt},
            {"role": "user", "content": question}
        ],
        temperature=0.1
    )
    │
    ▼
[5. 返回] {answer, sources, used_files}
```

### 7.2 Prompt 模板系统

实现文件：`src/prompt_loader.py`，模板文件：`prompts/system_prompt.toml`

模板采用 TOML 格式，包含：

- **角色定义**：AI 分身 = 被面试者的数字化身，以第一人称回答问题
- **行为规则**：零幻觉（只基于提供的上下文回答）、STAR 法则组织项目经验、不知道就说不知道
- **上下文占位符**：`{role}`、`{rules}`、`{context}`、`{question}`

### 7.3 API 层

`api/routes/chat.py:21-49`：将 `RAGChatEngine` 封装为 FastAPI 端点。

`RAGChatEngine` 使用单例模式（`get_engine()`），避免每次请求都重新加载向量索引。

### 7.4 索引构建

实现文件：`src/build_index.py`，类 `IndexBuilder`

独立的离线管道，支持三种模式：

| 模式 | CLI 命令 | 说明 |
|------|----------|------|
| 增量构建 | `python -m src.build_index` | 默认：检查已有 embeddings，仅追加新 chunk |
| 全量重建 | `python -m src.build_index --full` | 清空全部索引重建 |
| 同步模式 | `python -m src.build_index --sync` | 基于文件哈希变更检测，仅重建变化的文件 |
| 状态检查 | `python -m src.build_index --status` | 只读检查，输出索引状态 |

---

## 8. 缓存与优化层

### 8.1 HuggingFace 模型缓存

`src/config.py:194-200`

强制将 HuggingFace 全家桶的缓存目录指向 `E:/AI_Cache/huggingface`，避免：
- 模型重复下载
- C 盘空间耗尽（Windows 下 HuggingFace 默认缓存路径在 `C:\Users\<user>\.cache\huggingface`）

### 8.2 文件哈希变更检测

`src/build_index.py:298-370`

- 使用 MD5 计算每个源文件的哈希值
- 存储于 `vector_db/file_hashes.json`
- `--sync` 模式对比新旧哈希，仅重建增/改/删的文件
- 避免每次都要全量重建索引（当文档数量增多时尤为重要）

### 8.3 增量索引

`src/build_index.py:235-251`

默认模式下，按 `{source, chunk_index}` 键去重，仅追加新 chunk 到已有索引。大幅减少重复计算。

### 8.4 索引一致性自动修复

`src/retriever.py:389-433`

检测 embeddings、BM25 和 chunks_meta 三者的记录数是否一致：
- 不一致时自动裁剪到最小值并输出警告
- 建议用户执行全量重建
- 防止因不一致导致索引越界崩溃

### 8.5 懒加载

- `src/chat_engine.py:38-49` → OpenAI 客户端通过 `@property` 懒加载
- `api/routes/chat.py:13-18` → `RAGChatEngine` 以模块级单例懒加载

避免启动时一次性加载所有重量级组件。

### 8.6 未实现的优化

| 未实现 | 原因 |
|--------|------|
| 跨查询结果缓存（Redis/内存字典） | 简历问答场景中查询重复率极低，缓存命中率难以保证 |
| 交叉编码器重排序（Cross-Encoder Reranker） | RRF 融合 + 多权重预设已满足当前精度需求；重排序会引入额外延迟和模型依赖 |
| 查询改写（Query Rewriting） | 用户的自然语言查询通常已足够精准，暂不需要 |
| 多轮对话上下文管理 | 当前产品形态为单轮问答，未引入对话历史管理 |

### 8.7 Self-Reflection 重试（评估模式）

`src/evaluator.py:734-793`

在 LLM-as-a-Judge 评估模式下，如果评分不理想（准确率 < 3 或相关性 < 4），系统会进行反思重试：用已评分 judge 的评语作为反馈提示 LLM 重新生成答案，再次评分。这是一种基于 LLM 自我反思的质量提升机制。

### 8.8 权重预设 A/B 测试

`src/config.py:128-135, 248-266` + `src/evaluator.py:541-562`

五种检索权重预设可通过 `set_weight_preset()` 在运行时切换，并结合 `compare_presets()` 进行对比基准测试，用数据驱动选择最优策略。

---

## 9. 技术选型总结

### 9.1 技术栈全景图

| 层面 | 技术选型 | 核心考量 |
|------|----------|----------|
| **嵌入模型** | `all-MiniLM-L6-v2` (384d) | 轻量本地部署，平衡性能与资源 |
| **向量存储** | NumPy `.npy` 内存加载 | 小规模数据，最简单可靠 |
| **文档解析** | pdfplumber / pypdf / pytesseract / python-docx | 中文 PDF 兼容性 + 三级降级 |
| **分块策略** | 结构感知 + 句子边界 + 500字符/100重叠 | 尊重文档自然结构 |
| **Dense 检索** | SentenceTransformer 余弦相似度 | 语义理解 |
| **Sparse 检索** | jieba 分词 + BM25Okapi (rank-bm25) | 中文关键词精确匹配 |
| **融合策略** | RRF (k=60) + 加权归一化分数 | 消除分数尺度差异 |
| **中文分词** | jieba + 90+ 条自定义词典 + 130 停用词 | 简历领域术语识别 |
| **生成模型** | DeepSeek Chat（默认）/ 可切换至 GPT-4o | 中文能力强 + 性价比高 |
| **LLM 接口** | OpenAI 兼容 API（`openai` SDK） | Provider 无关，切换零成本 |
| **Prompt 管理** | TOML 模板 + `str.format()` | 结构化、可维护 |
| **Web 后端** | FastAPI + uvicorn | 高性能异步 |
| **Web 前端** | Streamlit | 快速原型，AI 场景友好 |
| **配置管理** | python-dotenv + dataclass 单例 | 集中管理 + 环境变量覆盖 |
| **RAG 框架** | **无框架，全栈自研** | 完全可控，理解每一行 |

### 9.2 核心设计哲学

1. **全栈自研优于框架依赖**：LangChain/LlamaIndex 虽然功能丰富，但引入了大量间接依赖和黑盒抽象。对于本项目的规模，自研管道的代码量约 1500 行，带来的是每个环节的完全可控性和透明性。

2. **本地优先**：嵌入模型本地运行，零 API 延迟，零费用。唯一的外部 API 调用是 LLM 生成（DeepSeek），这是当前技术条件下无法本地化的环节。

3. **务实主义**：ChromaDB 在 Windows 上有兼容性问题就直接用 NumPy；数据量不大就内存加载；不需要重排序就不加。不追求技术的"先进性"，只追求解决当前问题的"最适合方案"。

4. **中文场景深度优化**：从 jieba 自定义词典到 pdfplumber 中文 PDF 解析，从 STAR 法则 prompt 到中文句子边界正则，每个环节都针对中文简历问答场景做了定制化处理。

5. **可观测性与可调试性**：丰富的元数据（source、file_type、section_type）、索引一致性校验、哈希变更检测、评估器评分体系——这些都是为了确保系统行为的可理解和可调试。

---

> 最后更新：2026-07-31

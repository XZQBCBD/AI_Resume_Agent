# AI 数字分身系统（RAG 架构）

> 基于 RAG（检索增强生成）的个人 AI 数字分身，用于求职面试场景。
> 支持混合检索（RRF 排序 + MMR 多样性 + 跨文档去重）、查询意图感知 section_type 加权、章节类型标注、LLM-as-a-Judge 评估（含 Self-Reflection 重试）、增强 BM25、Markdown 表格保护、扫描版 PDF OCR。

---

## ⚠️ 首次使用

### 1. 修改配置

通过 `.env` 覆盖默认配置（推荐）：

```ini
MODEL_CACHE_DIR=E:/AI_Cache/huggingface   # 模型缓存，必须非系统盘
DATA_RAW_PATH=E:/MyDocs/resume_data       # 文档目录（可选）
VECTOR_DB_PATH=D:/VectorDB                # 向量库目录（可选）
```

### 2. 配置 API Key

```bash
cp .env.example .env
# 编辑 .env，至少填一个 LLM API Key
```

### 3. 放置文档到 `data/raw/`

> ⚠️ **重要**：只放面试相关的知识文档（项目总结、技能清单、个人背景），不要放 README/部署指南等运维文档。超大文档（>15KB）建议按模块拆分为多个文件，提升检索粒度。

| 前缀 | 自动分类 | 建议内容 | 示例 |
|------|----------|---------|------|
| `00_` | 个人总结 | 教育背景 + 项目摘要 + 求职意向 | `00_个人总结_谢作乾.md` |
| `01_` | 简历 | PDF 简历 | `01_谢作乾简历.pdf` |
| `02_` | 项目 | 按模块细分的项目总结（每个 5-15KB） | `02_AI数字分身_核心功能.md` |
| `03_` | 博客 | 技术博客文章 | `03_技术博客.md` |
| `04_` | 推荐信 | 推荐信 | `04_推荐信.pdf` |
| `05_` | 技能全景 | 技术栈唯一权威来源（避免多文件重复） | `05_技能.md` |

> 不含前缀的文件名将通过关键词模糊匹配分类。`.gitkeep` 确保空目录被 Git 跟踪。

---

## 🚀 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 构建索引（首次自动生成示例文档，后续用 --sync 同步更新）
python -m src.build_index --sync

# 3. 启动后端
python run_api.py
# → API 文档 http://localhost:8000/docs

# 4. 新终端，启动前端
python run_app.py
# → 界面 http://localhost:8501
```

---

## 📁 项目结构

```
AI_Resume_Agent/
├── api/                         # FastAPI 后端
│   ├── main.py                  # 应用入口 + CORS
│   ├── schemas.py               # Pydantic 请求/响应模型
│   └── routes/
│       ├── chat.py              # POST /api/chat           RAG 问答
│       ├── documents.py         # GET /api/documents       文档列表
│       │                        # POST /api/reindex       重建索引
│       └── eval.py              # GET /api/eval/benchmark  性能基准
│                                # GET /api/eval/index      索引统计
│
├── app/                         # Streamlit 前端
│   ├── streamlit_app.py         # 主页面 + 全局 CSS
│   └── components/
│       ├── sidebar.py           # LLM 切换 + 预设问题
│       └── chat.py              # 对话展示 + 引用来源折叠
│
├── src/                         # RAG 核心引擎（零 Web 依赖）
│   ├── config.py                # 集中配置 + 权重预设 + RRF/MMR参数 + HF 缓存劫持
│   ├── loader.py                # 文档加载 + 智能切分 + 表格保护 + 章节标注 + 项目归属 + OCR
│   ├── bm25_utils.py            # BM25 增强：自定义词典 + 停用词过滤
│   ├── prompt_loader.py         # TOML Prompt 模板加载
│   ├── build_index.py           # 离线索引构建 + 变更检测 + 示例数据
│   ├── retriever.py             # 混合检索（Dense+Sparse+RRF融合+QueryIntent加权+MMR去重+一致性校验）
│   ├── chat_engine.py           # RAG 总控引擎（含 project + section_type 元信息注入）
│   └── evaluator.py             # 性能评估 + Hit@K/MRR + A/B对比 + LLM-as-Judge
│
├── tests/
│   └── test_questions.json      # 30题/8类意图测试集
├── prompts/
│   └── system_prompt.toml       # Prompt 模板（外部化，修改即生效）
├── data/raw/                    # 原始文档（面试知识库，仅面试相关内容）
│   ├── 00_个人总结_谢作乾.md     # 个人背景 + 教育 + 项目摘要 + 求职意向
│   ├── 01_谢作乾简历.pdf         # PDF 简历
│   ├── 02_AI数字分身_设计思路.md  # 项目架构 + Agent 理念 + 痛点规避
│   ├── 02_AI数字分身_技术选型.md  # LLM/Embedding/向量存储/BM25/后端/前端选型
│   ├── 02_AI数字分身_核心功能.md  # PDF解析→切分→检索→Prompt→Provider→索引
│   ├── 02_AI数字分身_优化思路.md  # 检索质量 + 系统性能 + 工程架构优化
│   ├── 02_AI数字分身_评估与亮点.md # 评估指标 + A/B对比 + 10大亮点 + JD映射
│   ├── 02_MamaCare_设计思路.md    # 项目概览 + 系统架构 + LangGraph 工作流
│   ├── 02_MamaCare_技术选型.md    # 技术栈总览 + 关键依赖
│   ├── 02_MamaCare_核心功能.md    # 目录结构 + 多意图/RAG/记忆/推送/安全/工具/API/DB
│   ├── 02_MamaCare_优化思路.md    # v2.2优化 + 8大技术亮点
│   ├── 02_MamaCare_评估与亮点.md  # 四维评估 + 技术亮点 + 核心能力表
│   └── 05_技能.md                # 技术全景（技能唯一权威来源）
├── vector_db/                   # 向量 + BM25 索引 + 文件哈希快照
├── run_api.py / run_app.py      # 启动脚本
├── requirements.txt
└── README.md
```

---

## 🔧 核心特性

### 1. PDF 解析：三级级联引擎

```
PDF 文件
  ├─ 1️⃣ pdfplumber    ← 中文兼容性最优（默认生效）
  ├─ 2️⃣ pypdf         ← 回退方案
  └─ 3️⃣ OCR/pytesseract ← 乱码率 > 30% 自动触发
```

扫描版 PDF 需额外安装：
```bash
pip install pdf2image pytesseract
# + Tesseract-OCR: https://github.com/UB-Mannheim/tesseract/wiki
```

### 2. 文本切分：语义感知 + 表格保护 + 章节类型标注

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `CHUNK_SIZE` | 500 | 单块最大字符数 |
| `CHUNK_OVERLAP` | 100 | 相邻块重叠（20%） |
| `MIN_CHUNK_SIZE` | 200 | 最小块长度，低于此值合并 |

六级切分策略：
1. **表格保护**：检测 Markdown 表格行列边界（`|---|` 分隔行），完整表格块作为不可分割单元，防止技术对比表被拦腰截断
2. **结构感知**：识别列表项（`1.`、`-`、`第X章`）和标题（`##`）作为硬边界
3. **段落切分**：按空行自然分段
4. **超长拆分**：> 500 字符按句子边界二次切分
5. **短块合并**：< 200 字符与相邻块合并，保持语义完整
6. **章节标注**：自动检测标题和正文内容，为每个 chunk 标注 `section_type`（education / project / skills / experience / self_intro / awards / contact / general），提升下游检索精度

### 3. BM25 增强分词

| 优化项 | 说明 |
|--------|------|
| 自定义词典 | 73 个技术术语（YOLOv5、PyTorch、RAG、LLM 等） |
| 停用词过滤 | 180 个中英文无意义词（的、了、the、a 等） |
| 分数阈值 | `BM25_SCORE_THRESHOLD=0.05`，过滤噪声命中 |

### 4. 混合检索：RRF（倒数排名融合）

| 通路 | 技术 | 说明 |
|------|------|------|
| Dense | numpy 余弦相似度 | 语义理解 |
| Sparse | BM25 + 增强 jieba | 关键词精准匹配 |

**融合算法：RRF 排序 + 原始分数归一化**

```
Step 1: RRF 排序
  RRF_score(d) = Σ 1/(k + rank_i(d)), k = 60
  解决两路分数量纲不一致问题，确定最终排名

Step 2: 分数归一化（展示用）
  对 Top-K 结果的 Dense 余弦相似度和 BM25 原始得分
  分别做 Min-Max 归一化到 [0, 1]
  加权求和: display_score = w_dense × norm_dense + w_bm25 × norm_bm25
```

**权重预设**（`.env` 中 `WEIGHT_PRESET` 切换，用于 Min-Max 备用模式）：

| 预设 | Dense | Sparse | 适用场景 |
|------|-------|--------|----------|
| `balanced` | 0.5 | 0.5 | 通用均衡 |
| `semantic_first` | 0.6 | 0.4 | 语义理解需求高 |
| `keyword_first` | 0.4 | 0.6 | 精确关键词匹配优先 |
| `dense_only` | 1.0 | 0.0 | 纯向量检索 |
| `sparse_only` | 0.0 | 1.0 | 纯关键词检索 |

### 5. 查询意图感知 + MMR 多样性 + 跨文档去重

**查询意图识别**：根据用户问题中的关键词自动识别目标章节类型，对匹配的 chunk 做 RRF 分数加权（×1.2）。

| 意图 | 触发词示例 | 加权 section_type |
|------|-----------|-------------------|
| 技能 | 技能、框架、语言、掌握、精通 | skills |
| 教育 | 学校、毕业、学历、专业 | education |
| 项目 | 项目、开发、架构、RAG | project |
| 自我介绍 | 自我介绍、背景、你是谁 | self_intro |

**MMR（Maximal Marginal Relevance）多样性选取**：

```
MMR(d) = λ × RRF_score(d) - (1-λ) × max_sim(d, selected)
λ = 0.7, 同文档最多入选 2 条
```

- 在 RRF 排序后，贪心地逐条选取结果
- 每次选取时平衡「相关性」与「已选结果的差异性」
- 同一文档的 chunk 最多 2 条，防止单文档垄断 Top-K

**跨文档去重**：对 MMR 结果做最终安全检查——不同文件间余弦相似度 > 0.85 的 chunk，保留排名靠前的，去除冗余。

**项目过滤（v2.1）**：查询中检测项目关键词（"AI数字分身"/"MamaCare"）→ 非目标项目的 chunk 从候选池分离至后备池 → MMR 仅从目标项目选取。后备池仅在目标项目候选不足时补位，彻底消除跨项目张冠李戴。

### 6. 性能评估 + A/B 对比 + LLM-as-Judge

```bash
# 扩展基准测试（30 题 + Hit Rate@K + MRR + 分类统计）
python -m src.evaluator --extended

# LLM-as-a-Judge 评估（准确性/相关性/完整性三维打分）
python -m src.evaluator --judge

# LLM-as-a-Judge + Self-Reflection 重试（自动纠错低质量回答）
python -m src.evaluator --judge --retry

# 权重预设 A/B 对比（balanced / semantic_first / keyword_first）
python -m src.evaluator --compare

# 默认 5 题基准 + 索引质量报告
python -m src.evaluator

# 单条评估
python -m src.evaluator --question "自我介绍"
```

**评估指标**：

| 指标 | 说明 |
|------|------|
| Hit Rate@K | Top-K 结果中至少 1 条命中期望关键词的比例 |
| MRR | 第一个相关文档排名的倒数均值 |
| Top1 分数 | 排名第一的融合分数 |
| 双路命中 | Dense + Sparse 同时命中的文档数 |
| LLM Judge | 准确性 / 相关性 / 完整性三维 LLM 打分（1-5） |

**测试集**：`tests/test_questions.json` — 30 题 / 8 类别（自我介绍、项目经历、技术栈、RAG系统、Agent开发、教育背景、团队角色、工程实践），每题含 `expected_keywords` 用于自动评估。

### 6. LLM Provider 可切换

| Provider | 配置方式 |
|----------|----------|
| DeepSeek | `.env` 中 `DEEPSEEK_API_KEY=sk-xxx` |
| OpenAI | `.env` 中 `OPENAI_API_KEY=sk-xxx` |
| 自定义 | `.env` 中 `CUSTOM_BASE_URL` + `CUSTOM_API_KEY` |

前端侧边栏下拉框切换，或 `.env` 中 `LLM_PROVIDER=deepseek|openai|custom`

---

## 📦 向量数据库更新（增 / 删 / 改）

系统通过 **MD5 文件哈希** 追踪文档变更，支持智能增量同步。

### 四种模式

| 命令 | 行为 | 适用场景 |
|------|------|----------|
| `python -m src.build_index --sync` | 检测变更 → 有变化则自动重建 | **日常使用首选** |
| `python -m src.build_index --status` | 仅检测变更，不执行操作 | 查看哪些文档有变化 |
| `python -m src.build_index --sync --dry-run` | 检测变更并报告，不重建 | 预览将要更新的内容 |
| `python -m src.build_index --full` | 强制全量重建 | 修改了切分/检索参数后 |
| `python -m src.build_index` | 增量追加（默认） | 仅新增文档，快速追加 |

### 变更检测原理

```
data/raw/ 文档 ──MD5哈希──▶ file_hashes.json（上次快照）
                                  │
                                  ▼ 对比
                   ┌────────┬──────────┬──────────┐
                   │  新增  │   修改   │   删除   │
                   │ hash新 │ hash不同 │ hash消失 │
                   └────────┴──────────┴──────────┘
                                  │
                                  ▼
                          任一变更 → 全量重建
                          无变更   → 跳过
```

### 典型工作流

```bash
# 场景1：添加新简历 PDF
cp ~/新简历.pdf data/raw/
python -m src.build_index --sync      # 自动检测新增 → 重建

# 场景2：修改了项目文档
vim data/raw/02_项目_xxx.md
python -m src.build_index --status    # 先看看变化
python -m src.build_index --sync      # 确认后同步

# 场景3：删除了过期文档
rm data/raw/旧简历.pdf
python -m src.build_index --status    # 显示「删除: 旧简历.pdf」
python -m src.build_index --sync      # 自动清理 + 重建

# 场景4：不确定有没有改过
python -m src.build_index --status    # 仅检测，零开销

# 场景5：调整了切分参数（CHUNK_SIZE 等）
python -m src.build_index --full      # 强制全量重建
```

### 通过 API 重建

```bash
# 远程/自动化场景
curl -X POST http://localhost:8000/api/reindex
```

---

## 🌐 API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/api/chat` | RAG 问答 |
| `GET` | `/api/documents` | 已索引文档清单 |
| `POST` | `/api/reindex` | 全量重建索引 |
| `GET` | `/api/health` | 健康检查 |
| `GET` | `/api/eval/benchmark` | 检索性能基准（5题） |
| `GET` | `/api/eval/index` | 索引质量统计 |

---

## ⚙️ 配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `TOP_K_RETRIEVAL` | 8 | 最终返回结果数 |
| `TOP_K_CANDIDATE` | 10 | 各通路召回候选数 |
| `RRF_K` | 60 | RRF 融合常数（越大排名差异越平滑） |
| `MMR_LAMBDA` | 0.7 | MMR 相关性/多样性平衡（越大越偏相关性） |
| `DEDUP_SIM_THRESHOLD` | 0.85 | 跨文档去重相似度阈值 |
| `SECTION_BOOST_FACTOR` | 1.2 | section_type 匹配加权系数 |
| `DEBUG_RETRIEVER` | false | 启用检索融合调试日志（打印 Top-3 详情） |
| `WEIGHT_PRESET` | balanced | 权重预设（Min-Max 备用模式） |
| `VECTOR_WEIGHT` | 0.5 | 向量检索权重（Min-Max 备用） |
| `BM25_WEIGHT` | 0.5 | BM25 检索权重（Min-Max 备用） |
| `BM25_SCORE_THRESHOLD` | 0.05 | BM25 最低分数阈值 |
| `CHUNK_SIZE` | 500 | 文本块最大字符数 |
| `CHUNK_OVERLAP` | 100 | 相邻块重叠字符数 |
| `MIN_CHUNK_SIZE` | 200 | 最小块长度 |
| `TEMPERATURE` | 0.1 | LLM 生成温度 |
| `LLM_TIMEOUT` | 60 | LLM API 超时秒数 |

所有参数均可通过 `.env` 环境变量覆盖。

---

## 🛠️ 技术栈

| 层次 | 技术 |
|------|------|
| 检索融合 | RRF（Reciprocal Rank Fusion, k=60）+ MMR 多样性 + 跨文档去重 |
| 检索加权 | 查询意图识别 → section_type 加权（×1.2） |
| 检索 (Dense) | numpy 余弦相似度，矩阵运算加速 |
| 检索 (Sparse) | rank-bm25 + jieba（73 自定义词 + 180 停用词） |
| 嵌入模型 | BAAI/bge-small-zh-v1.5（中文优化，512维） |
| LLM 接入 | OpenAI 兼容 API（DeepSeek/OpenAI/自定义） |
| 后端 | FastAPI + uvicorn |
| 前端 | Streamlit |
| PDF 解析 | pdfplumber → pypdf → OCR 三级级联 |
| 文本切分 | 表格保护 + 结构感知 + 段落优先 + 句子兜底 + 短块合并 + 章节标注 |
| 评估 | src/evaluator.py（Hit@K + MRR + A/B 对比 + LLM-as-Judge） |

---

## 📝 自定义 Prompt

编辑 `prompts/system_prompt.toml`，修改角色定义、行为规则和预设问题，重启后端即可生效。

```toml
[system]
role = "你是「谢XX」的 AI 数字分身..."
rules = "请严格遵守以下规则：..."

[template]
system_prompt = "{role}\n{rules}\n...\n{context}\n...\n{question}"

[ui]
app_title = "AI面试助手"
preset_questions = ["请做自我介绍", ...]
```

---

## 🔧 常见问题

**Q: HuggingFace 下载模型失败？**
A: `.env` 已配置 `HF_ENDPOINT=https://hf-mirror.com`（国内镜像），首次构建自动使用。

**Q: 如何更新向量数据库（增/删/改文档）？**
A: 修改 `data/raw/` 目录后运行 `python -m src.build_index --sync`，系统自动检测变更并按需重建。详见上方「📦 向量数据库更新」章节。

**Q: 如何切换 LLM？**
A: 前端侧边栏下拉框切换，或 `.env` 中修改 `LLM_PROVIDER`。

**Q: 扫描版 PDF 没有文本？**
A: `pip install pdf2image pytesseract` + 安装 Tesseract-OCR。系统自动检测乱码率并触发 OCR。

**Q: RRF 分数为什么很小（0.01-0.03）？**
A: RRF 公式 `1/(60+rank)` 天然输出小数值，这是预期行为。系统在 RRF 排序后会自动对原始 Dense/BM25 分数做 Min-Max 归一化生成 0~1 的可读分数。启用 `DEBUG_RETRIEVER=true` 可查看每步分数详情：
```bash
DEBUG_RETRIEVER=true python -c "from src.retriever import HybridRetriever; r=HybridRetriever(); print(r.search('自我介绍')[:2])"
```

**Q: 检索结果不够准确？**
A:
- 调整 `RRF_K` 参数（`.env` 中 `RRF_K=30` 增加排名差异敏感度，`RRF_K=120` 更平滑）
- 增大召回：`TOP_K_RETRIEVAL=8`
- 调整 MMR 参数：`MMR_LAMBDA=0.5` 更偏多样性，`MMR_LAMBDA=0.9` 更偏相关性
- 运行 `python -m src.evaluator --extended` 查看各类别 Hit Rate

**Q: 什么是 MMR？如何调整多样性？**
A: MMR（Maximal Marginal Relevance）在保证检索相关性的同时，让 Top-K 结果覆盖不同文档和不同维度的信息。`MMR_LAMBDA=0.7` 是默认平衡值——设为 0.5 会更多样化（不同文档的结果更多），设为 0.9 更偏向纯相关性。同文档最多入选 2 条的规则在 `retriever.py` 的 `_apply_mmr()` 中可调整。

**Q: 如何评估检索质量？**
A:
```bash
python -m src.evaluator --extended   # 30 题 + Hit@K + MRR + 分类统计
python -m src.evaluator --judge      # LLM 三维打分（准确性/相关性/完整性）
python -m src.evaluator --compare    # A/B 权重对比
```

**Q: BM25 分词不准（技术术语被拆开）？**
A: 编辑 `src/bm25_utils.py` 中的 `CUSTOM_WORDS` 列表，添加自定义词后重建索引。

"""
多格式文档加载与智能文本分块模块。

职责：
    1. 根据文件后缀自动选择解析器（策略模式）
    2. 按自然段落切分，保持语义完整性
    3. 段落超长时按句子边界二次切分
    4. 每个文本块附带元数据（来源、类型、序号）

支持的格式：.pdf / .docx / .md / .txt

使用方式：
    from src.loader import DocumentLoader
    loader = DocumentLoader()
    chunks = loader.load_all(data_path)
"""

import re
import logging
from pathlib import Path
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


# ============================================================
# 数据模型
# ============================================================

class Chunk:
    """文本块 —— RAG 管道中最小的检索单元。

    Attributes:
        content: 文本内容
        metadata: 元数据 (source, file_type, chunk_index, total_chunks)
    """

    def __init__(self, content: str, metadata: Optional[Dict] = None):
        self.content = content.strip()
        self.metadata = metadata or {}

    def __repr__(self):
        src = self.metadata.get("source", "unknown")
        idx = self.metadata.get("chunk_index", "?")
        preview = self.content[:50].replace("\n", " ")
        return f"<Chunk [{src}#{idx}] {preview}...>"


# ============================================================
# 文件类型映射
# ============================================================

# 文件名前缀 → 文档类型映射（精确匹配）
PREFIX_TYPE_MAP = {
    "00_": "个人总结",
    "01_": "简历",
    "02_": "项目",
    "03_": "博客",
    "04_": "推荐信",
    "05_": "技能全景",
}

# 文件名关键词 → 文档类型映射（模糊匹配，优先级低于前缀匹配）
KEYWORD_TYPE_MAP = {
    "简历": "简历",
    "项目": "项目",
    "博客": "博客",
    "推荐信": "推荐信",
    "resume": "简历",
    "cv": "简历",
    "个人总结": "个人总结",
    "技能": "技能全景",
    "技术全景": "技能全景",
}


def infer_file_type(filename: str) -> str:
    """根据文件名推断文档类型。

    优先按前缀精确匹配（01_, 02_...），其次按关键词模糊匹配。

    Args:
        filename: 文件名（如 01_简历_张三.pdf 或 谢作乾简历（new）.pdf）

    Returns:
        文档类型中文名（简历/项目/博客/推荐信/其他文档）
    """
    # 1. 前缀精确匹配
    for prefix, ftype in PREFIX_TYPE_MAP.items():
        if filename.startswith(prefix):
            return ftype

    # 2. 关键词模糊匹配（大小写不敏感）
    lower_name = filename.lower()
    for keyword, ftype in KEYWORD_TYPE_MAP.items():
        if keyword.lower() in lower_name:
            return ftype

    return "其他文档"


# ============================================================
# 文档解析器（策略模式）
# ============================================================

class BaseParser:
    """解析器基类 —— 所有格式解析器必须实现 parse() 方法。"""

    def parse(self, file_path: Path) -> str:
        raise NotImplementedError


class PDFParser(BaseParser):
    """PDF 解析器 —— 多引擎级联：pdfplumber → pypdf → OCR。

    策略：
        1. 优先使用 pdfplumber（中文编码兼容性最好）
        2. 若为空则回退到 pypdf
        3. 文本质量检测（乱码率 > 30% 触发 OCR）
        4. OCR 使用 pytesseract（需安装 Tesseract-OCR + 中文语言包）
    """

    # 乱码检测：连续非ASCII可打印字符占比过高视为乱码
    GARBAGE_THRESHOLD = 0.30
    # 最低有效文本长度（低于此值认为解析失败）
    MIN_TEXT_LENGTH = 50

    def parse(self, file_path: Path) -> str:
        full_text = ""

        # 策略1: pdfplumber（中文支持最优）
        try:
            full_text = self._extract_with_pdfplumber(file_path)
            if len(full_text.strip()) >= self.MIN_TEXT_LENGTH:
                logger.info("  📄 pdfplumber 解析成功: %s", file_path.name)
            else:
                full_text = ""
        except Exception as e:
            logger.debug("  pdfplumber 解析失败: %s", e)

        # 策略2: pypdf（回退方案）
        if not full_text.strip():
            try:
                full_text = self._extract_with_pypdf(file_path)
                if len(full_text.strip()) >= self.MIN_TEXT_LENGTH:
                    logger.info("  📄 pypdf 解析成功: %s", file_path.name)
                else:
                    full_text = ""
            except Exception as e:
                logger.debug("  pypdf 解析失败: %s", e)

        # 策略3: 质量检测 → OCR
        if full_text.strip():
            garbage_ratio = self._calc_garbage_ratio(full_text)
            if garbage_ratio > self.GARBAGE_THRESHOLD:
                logger.warning(
                    "  ⚠️ 文本乱码率过高(%.0f%%)，尝试OCR: %s",
                    garbage_ratio * 100, file_path.name,
                )
                ocr_text = self._extract_with_ocr(file_path)
                if ocr_text.strip():
                    full_text = ocr_text
        else:
            logger.warning("  ⚠️ 未提取到文本，尝试OCR: %s", file_path.name)
            ocr_text = self._extract_with_ocr(file_path)
            if ocr_text.strip():
                full_text = ocr_text

        # 后处理：清洗页眉页脚数字、多余空白
        full_text = self._clean_text(full_text)

        if not full_text.strip():
            logger.warning("⚠️ PDF 所有解析方式均失败: %s", file_path.name)

        return full_text

    # ---- 解析引擎 ----

    def _extract_with_pdfplumber(self, file_path: Path) -> str:
        """使用 pdfplumber 提取文本（中文兼容性最好）。"""
        import pdfplumber
        texts = []
        with pdfplumber.open(str(file_path)) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    texts.append(text)
        return "\n".join(texts)

    def _extract_with_pypdf(self, file_path: Path) -> str:
        """使用 pypdf 提取文本（回退方案）。"""
        from pypdf import PdfReader
        reader = PdfReader(str(file_path))
        texts = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                texts.append(text)
        return "\n".join(texts)

    def _extract_with_ocr(self, file_path: Path) -> str:
        """使用 OCR 识别扫描版 PDF。

        先将 PDF 每页转为图片，再用 pytesseract 识别中文。
        需要安装: Tesseract-OCR + chi_sim 语言包
        """
        try:
            from pdf2image import convert_from_path
            import pytesseract
        except ImportError:
            logger.warning(
                "  ⚠️ OCR 依赖未安装。请执行:\n"
                "    pip install pdf2image pytesseract\n"
                "    并安装 Tesseract-OCR: https://github.com/UB-Mannheim/tesseract/wiki"
            )
            return ""

        try:
            images = convert_from_path(str(file_path), dpi=200)
            texts = []
            for i, img in enumerate(images):
                text = pytesseract.image_to_string(img, lang="chi_sim+eng")
                if text.strip():
                    texts.append(text)
            logger.info("  🔍 OCR 识别完成: %s (%d 页)", file_path.name, len(images))
            return "\n".join(texts)
        except Exception as e:
            logger.error("  ❌ OCR 识别失败: %s", e)
            return ""

    # ---- 文本质量检测 ----

    def _calc_garbage_ratio(self, text: str) -> float:
        """计算文本乱码率。

        检测连续的非ASCII可打印字符比例，
        和无效 Unicode 字符（如替换字符 U+FFFD）。
        """
        if not text:
            return 1.0

        garbage_count = 0
        total = len(text)

        for ch in text:
            code = ord(ch)
            # U+FFFD 替换字符 = 明确乱码
            if code == 0xFFFD:
                garbage_count += 1
            # 控制字符（除了常见空白）
            elif code < 32 and code not in (9, 10, 13):
                garbage_count += 1

        return garbage_count / max(total, 1)

    # ---- 文本清洗 ----

    def _clean_text(self, text: str) -> str:
        """清洗提取的文本。

        1. 移除独立数字行（页眉页脚页码残留）
        2. 规范化空白
        """
        import re

        lines = text.split("\n")
        cleaned = []

        for line in lines:
            stripped = line.strip()
            # 跳过纯数字行（页码残留）或单字符行
            if re.match(r'^\d{1,3}$', stripped):
                continue
            # 跳过短无意义行（如 ".06"、单独的标点）
            if len(stripped) <= 2 and re.match(r'^[\d\.\,\;\:\-\—]+$', stripped):
                continue
            if stripped:
                cleaned.append(line)

        text = "\n".join(cleaned)

        # 合并多个连续空行为单个空行
        text = re.sub(r'\n{3,}', '\n\n', text)

        return text


class DocxParser(BaseParser):
    """Word 解析器 —— 使用 python-docx 提取段落文本。"""

    def parse(self, file_path: Path) -> str:
        from docx import Document
        doc = Document(str(file_path))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        return "\n".join(paragraphs)


class PlainTextParser(BaseParser):
    """纯文本解析器 —— 用于 .md 和 .txt 文件。"""

    def parse(self, file_path: Path) -> str:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()


# 后缀 → 解析器映射
PARSER_MAP = {
    ".pdf": PDFParser(),
    ".docx": DocxParser(),
    ".md": PlainTextParser(),
    ".txt": PlainTextParser(),
}

# ============================================================
# 章节类型检测 —— 自动标注 chunk 所属语义章节
# ============================================================

# (正则, 章节类型) 列表 — 按优先级从高到低匹配
SECTION_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r'教育背景|学历|毕业院校|主修|专业|课程|学习经历|在校'), "education"),
    (re.compile(r'项目经历|项目经验|项目概述|项目成果|核心项目|主要项目|项目描述'), "project"),
    (re.compile(r'技术(栈|能力|技能|特长)|编程语言|开发能力|熟练掌握|精通|了解.*框架'), "skills"),
    (re.compile(r'工作经历|实习经历|工作经验|任职|在职|就职'), "experience"),
    (re.compile(r'自我(介绍|评价|描述)|个人(简介|概述|特点)|关于我|基本信息'), "self_intro"),
    (re.compile(r'获得荣誉|获奖|证书|竞赛|奖学金|荣誉称号'), "awards"),
    (re.compile(r'联系方式|电话|邮箱|地址|GitHub|博客|主页'), "contact"),
    (re.compile(r'推荐信|推荐人|推荐理由'), "recommendation"),
]

# 默认章节类型
DEFAULT_SECTION_TYPE = "general"


def detect_section_type(text: str, headings: list[str] | None = None) -> str:
    """根据文本内容和所在章节标题推断 chunk 的语义类型。

    优先匹配当前 chunk 内容，其次匹配所属章节标题。

    Args:
        text: chunk 文本内容（取前 200 字符用于匹配）
        headings: 该 chunk 所属的章节标题列表（从父级到当前）

    Returns:
        章节类型: education/project/skills/experience/self_intro/awards/contact/recommendation/general
    """
    # 合并检测文本：标题（权重高）+ 正文
    head_text = " ".join(headings) if headings else ""
    detect_text = (head_text + " " + text[:200]).lower()

    for pattern, sec_type in SECTION_PATTERNS:
        if pattern.search(detect_text):
            return sec_type

    return DEFAULT_SECTION_TYPE


def detect_headings_in_text(text: str) -> list[tuple[int, str]]:
    """从 Markdown/纯文本中提取章节标题及其位置。

    支持格式：
        - Markdown: # 标题 / ## 二级 / ### 三级
        - 纯文本: 以【】「」包裹的标题行
        - 数字编号: 一、/ 1. / 1.1 开头的行

    Args:
        text: 原始文本

    Returns:
        [(行号, 标题文本), ...] 按出现位置排序
    """
    headings: list[tuple[int, str]] = []
    lines = text.split("\n")

    md_heading = re.compile(r'^#{1,4}\s+(.+)$')
    bracket_heading = re.compile(r'^[【「](.+?)[】」]$')
    numbered_heading = re.compile(
        r'^(?:[一二三四五六七八九十]+[、．.]|(?:\d+[.、．])|(?:\d+\.\d+))\s*(.+)$'
    )

    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue

        m = md_heading.match(stripped)
        if m:
            headings.append((i, m.group(1).strip()))
            continue

        m = bracket_heading.match(stripped)
        if m:
            headings.append((i, m.group(1).strip()))
            continue

        m = numbered_heading.match(stripped)
        if m and len(stripped) > 5:  # 避免误匹配短数字行
            headings.append((i, m.group(1).strip()))

    return headings


# ============================================================
# 文本切分器
# ============================================================

class TextChunker:
    """智能文本切分器 —— 按段落切分，超长时按句子边界二次切分，块之间可重叠。

    切分规则（优先级从高到低）：
        1. 按列表项/标题行作为硬边界（保持结构完整性）
        2. 以空行（连续两个换行）为段落边界
        3. 段落超过 max_chunk_chars(500) 时，按句子边界二次切分
        4. 合并过短块（< min_chunk_size），保持 200-500 字符的理想区间
        5. 相邻块之间保留 chunk_overlap 字符的重叠区域
        6. 过滤纯空白块
    """

    # 列表/标题行检测正则
    LIST_PATTERN = re.compile(
        r'(?:^|\n)(?:\d+[.、．)]\s*|[-•·]\s*|[（(]\d+[)）]\s*|第[一二三四五六七八九十\d]+[章节条款]|#+\s)',
        re.MULTILINE,
    )

    def __init__(self, max_chunk_chars: int = 500, chunk_overlap: int = 100,
                 min_chunk_size: int = 200):
        """
        Args:
            max_chunk_chars: 每个文本块的最大字符数
            chunk_overlap: 相邻块之间的重叠字符数
            min_chunk_size: 最小块长度（低于此值尝试合并相邻块）
        """
        self.max_chunk_chars = max_chunk_chars
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        # 句子边界正则：匹配中文和英文句子结束符
        self.sentence_boundary = re.compile(
            r'(?<=[。！？.!?\n])\s*'
        )

    def split(self, text: str) -> List[str]:
        """将长文本切分为语义块，平衡长度分布。

        Args:
            text: 原始文本

        Returns:
            文本块列表（长度集中在 min_chunk_size ~ max_chunk_chars）
        """
        # 第一步：按列表/标题硬边界切分，再按段落切分
        paragraphs = self._split_by_structure(text)

        # 第二步：超长段落按句子边界二次切分
        chunks = []
        for para in paragraphs:
            if len(para) <= self.max_chunk_chars:
                chunks.append(para)
            else:
                sub_chunks = self._split_long_paragraph(para)
                chunks.extend(sub_chunks)

        # 第三步：过滤纯空白块
        chunks = [c.strip() for c in chunks if c.strip()]

        # 第四步：合并过短块（< min_chunk_size）
        chunks = self._merge_short_chunks(chunks)

        # 第五步：构建重叠（仅在句边界切分产生的块之间添加）
        if self.chunk_overlap > 0 and len(chunks) > 1:
            chunks = self._add_overlap_between_chunks(chunks)

        return chunks

    def _split_by_structure(self, text: str) -> List[str]:
        """按列表/标题结构切分，然后按段落切分。

        先识别列表项和标题行作为硬边界，
        再在每个段内按空行切分。
        """
        # 在列表项/标题前插入分隔符
        marked = self.LIST_PATTERN.sub(lambda m: "\n__SPLIT__\n" + m.group().lstrip(), text)
        # 按分隔符切分
        sections = marked.split("__SPLIT__")
        # 每段再按段落切分
        paragraphs = []
        for sec in sections:
            sec = sec.strip()
            if sec:
                paragraphs.extend(self._split_by_paragraphs(sec))
        return paragraphs

    def _merge_short_chunks(self, chunks: List[str]) -> List[str]:
        """合并过短的相邻块，使每块尽量达到 min_chunk_size。

        策略：从左到右扫描，将 < min_chunk_size 的块与后一个块合并。
        """
        if len(chunks) <= 1:
            return chunks

        merged = []
        i = 0
        while i < len(chunks):
            current = chunks[i]

            # 当前块过短，尝试与下一块合并
            while len(current) < self.min_chunk_size and i + 1 < len(chunks):
                # 检查合并后是否超限
                next_len = len(chunks[i + 1])
                if len(current) + next_len <= self.max_chunk_chars:
                    current = current + "\n" + chunks[i + 1]
                    i += 1
                else:
                    # 下一块很大，只从它借一部分补齐到 min_chunk_size
                    need = self.min_chunk_size - len(current)
                    borrow = chunks[i + 1][:need]
                    current = current + "\n" + borrow
                    # 剩余部分放回 chunks
                    chunks[i + 1] = chunks[i + 1][need:]
                    break

            merged.append(current.strip())
            i += 1

        return merged

    def _add_overlap_between_chunks(self, chunks: List[str]) -> List[str]:
        """在相邻块之间添加重叠区域。

        将前一个块的末尾 chunk_overlap 个字符拼接
        到下一个块的开头，确保跨块边界的信息不丢失。

        Args:
            chunks: 原始文本块列表

        Returns:
            带重叠的文本块列表
        """
        overlapped = []
        for i, chunk in enumerate(chunks):
            if i > 0 and len(chunks[i - 1]) > self.chunk_overlap:
                # 从前一个块取末尾 overlap 字符作为本块的前缀上下文
                prefix = chunks[i - 1][-self.chunk_overlap:]
                overlapped.append(prefix + "\n" + chunk)
            else:
                overlapped.append(chunk)
        return overlapped

    def _split_by_paragraphs(self, text: str) -> List[str]:
        """按空行切分段落。"""
        # 以连续换行符为段落分隔
        paragraphs = re.split(r'\n\s*\n', text)
        return [p.strip() for p in paragraphs if p.strip()]

    def _split_long_paragraph(self, paragraph: str) -> List[str]:
        """将超长段落按句子边界切分，确保每块不超过 max_chunk_chars。"""
        sentences = self.sentence_boundary.split(paragraph)
        chunks = []
        current = ""

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            if len(current) + len(sentence) <= self.max_chunk_chars:
                current += sentence
            else:
                if current.strip():
                    chunks.append(current.strip())
                current = sentence

        if current.strip():
            chunks.append(current.strip())

        return chunks


# ============================================================
# 文档加载器（主入口）
# ============================================================

class DocumentLoader:
    """多格式文档加载器 —— 遍历目录、解析文件、切分文本。

    使用方式：
        loader = DocumentLoader()
        chunks = loader.load_all(Path("data/raw"))
        for chunk in chunks:
            print(chunk.metadata["source"], len(chunk.content))
    """

    def __init__(self, max_chunk_chars: int = 500, chunk_overlap: int = 100):
        self.chunker = TextChunker(
            max_chunk_chars=max_chunk_chars,
            chunk_overlap=chunk_overlap,
        )
        self.supported_extensions = set(PARSER_MAP.keys())

    def load_all(self, data_dir: Path) -> List[Chunk]:
        """加载指定目录下的所有文档，切分为 Chunk 列表。

        Args:
            data_dir: 原始文档目录路径

        Returns:
            所有文档的 Chunk 列表，每个 Chunk 附带元数据
        """
        all_chunks = []
        files = self._collect_files(data_dir)

        if not files:
            logger.warning("⚠️ 目录 %s 中没有支持的文档文件", data_dir)
            return all_chunks

        logger.info("📄 发现 %d 个文档文件，开始加载...", len(files))

        for file_path in files:
            try:
                file_chunks = self._load_single_file(file_path)
                all_chunks.extend(file_chunks)
                logger.info(
                    "  ✅ %s → %d 个文本块", file_path.name, len(file_chunks)
                )
            except Exception as e:
                # 单文件失败不影响整体
                logger.error("  ❌ 加载失败 %s: %s", file_path.name, e)

        logger.info("📊 共加载 %d 个文档，切分为 %d 个文本块",
                     len(files), len(all_chunks))
        return all_chunks

    def _collect_files(self, data_dir: Path) -> List[Path]:
        """收集目录下所有支持格式的文件（按文件名排序）。"""
        files = []
        for ext in self.supported_extensions:
            files.extend(data_dir.glob(f"*{ext}"))
        files.sort(key=lambda p: p.name)
        return files

    def _load_single_file(self, file_path: Path) -> List[Chunk]:
        """加载单个文件并切分为 Chunk 列表。

        Args:
            file_path: 文件路径

        Returns:
            Chunk 列表
        """
        # 1. 选择解析器
        suffix = file_path.suffix.lower()
        parser = PARSER_MAP.get(suffix)
        if parser is None:
            raise ValueError(f"不支持的文件格式: {suffix}")

        # 2. 解析文本
        raw_text = parser.parse(file_path)
        if not raw_text.strip():
            logger.warning("⚠️ 文件 %s 无有效文本内容", file_path.name)
            return []

        # 3. 推断文件类型
        file_type = infer_file_type(file_path.name)

        # 4. 检测章节标题（用于标注 section_type）
        headings = detect_headings_in_text(raw_text)

        # 5. 切分为文本块
        chunk_texts = self.chunker.split(raw_text)

        # 6. 为每个块附加元数据（含 section_type）
        chunks = []
        total = len(chunk_texts)
        for i, text in enumerate(chunk_texts):
            # 找到该 chunk 所属的章节标题（chunk 开头之前最近的标题）
            chunk_start = raw_text.find(text) if text in raw_text else -1
            chunk_headings = []
            if chunk_start >= 0:
                chunk_line = raw_text[:chunk_start].count("\n")
                chunk_headings = [
                    h_text for h_line, h_text in headings if h_line <= chunk_line
                ]

            section_type = detect_section_type(text, chunk_headings)

            metadata = {
                "source": file_path.name,
                "file_type": file_type,
                "section_type": section_type,
                "chunk_index": i,
                "total_chunks": total,
            }
            chunks.append(Chunk(content=text, metadata=metadata))

        return chunks


# ============================================================
# 模块自测
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from src.config import settings

    loader = DocumentLoader()
    chunks = loader.load_all(settings.DATA_RAW_PATH)
    print(f"\n总计 {len(chunks)} 个文本块")
    for c in chunks[:3]:
        print(f"  [{c.metadata['file_type']}] {c.metadata['source']} "
              f"#{c.metadata['chunk_index']}: {c.content[:80]}...")

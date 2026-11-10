# ============================================================
# 【文件作用总览】
# 这是本地 RAG 系统的“核心引擎”，不包含任何 UI 代码，只负责数据链路：
#   1) 配置：读取 app_settings.json，统一 Web / CLI / MCP 三个入口的路径与模型；
#   2) 文档解析：把 pdf/docx/pptx/xlsx/txt/md 统一转成带页码/章节的纯文本；
#   3) 分块：按固定大小切 chunk，并附带 source/domain/page/section 元数据；
#   4) 向量化 + 入库：用本地 SentenceTransformer 生成向量，写入 ChromaDB；
#   5) 索引维护：全量重建 rebuild_index + 增量更新 update_index；
#   6) 检索：query_collection 支持按领域过滤，返回片段与元数据。
#
# 【为什么把配置也放在这里】
# 旧版本 app.py、build_index.py、mcp_server.py 各自写死了一套
# docs_dir / embedding_model / chroma_dir，导致 CLI 和 MCP 经常读到
# 另一个空向量库。现在三个入口都从这里读取同一份 app_settings.json，
# 并用 chroma_dir_for_model() 推导同一个向量库目录，从根源上消除漂移。
#
# 【设计原则】
# - 全部本地化、离线化：不依赖云服务，不依赖网络。
# - 解析差异收敛在最外层：所有格式先转成 [{text,page,section}]，
#   后面的切块/入库/检索只处理一种结构，新增格式时不用改核心逻辑。
# - ID 可复现：chunk id = 文件路径 MD5 + 序号，重复解析同一文件得到相同 ID，
#   增量更新时可以安全地先按 source 删除旧片段，再重新写入。
# ============================================================

# ---------- 标准库导入 ----------
import os            # 路径处理、目录遍历、环境变量
import re            # 正则：从 PDF 文本里识别“第N章 / N.N”章节标题
import json          # app_settings.json 与 file_hashes.json 的读写
import hashlib       # MD5：chunk id 前缀 + 文件内容指纹
import threading     # 保护进程级 collection 缓存，避免多线程重复加载模型
from typing import Any

# 【关键】离线环境变量必须在 import sentence_transformers 之前设置。
# 原因：SentenceTransformer 在导入/初始化阶段可能检查 HuggingFace 缓存，
# 企业内网/断网环境下会卡住甚至报错。放到 import 之后虽然有时也能生效，
# 但属于“碰运气”；放到之前才是确定性的做法。
# 使用 setdefault 而不是直接赋值，是为了允许部署方通过外部环境变量覆盖，
# 例如调试时可以临时 HF_HUB_OFFLINE=0 走在线模型。
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

# ---------- 第三方依赖 ----------
import chromadb                                       # 本地持久化向量数据库
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from sentence_transformers import SentenceTransformer  # 本地 embedding 模型
from pypdf import PdfReader                            # PDF 解析
from docx import Document                              # Word 解析
from pptx import Presentation                          # PPT 解析
from openpyxl import load_workbook                     # Excel 解析


# ==============================================================================
# 零、项目级常量与配置
# ==============================================================================

# 以“当前文件所在目录”作为项目根目录，而不是依赖 os.getcwd()。
# 原因：streamlit、MCP 客户端、cron 任务的工作目录可能完全不同；
# 如果所有相对路径都相对 CWD，换一个启动方式就会找不到 docs 或向量库。
# __file__ 是 rag_core.py 的绝对路径，dirname 后就是项目根目录，最稳定。
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# 三个入口共享的配置文件路径。
SETTINGS_FILE = os.path.join(PROJECT_ROOT, "app_settings.json")

# 增量索引的“文件内容指纹表”。放在项目根目录，方便人工排查。
# 注意：这里不用相对路径，避免 CLI 从其它目录启动时写到别处。
DEFAULT_HASH_MAP = os.path.join(PROJECT_ROOT, "file_hashes.json")

# Chroma 集合名固定为 local_kb。一个 chroma_dir 下只放一个集合，
# 目录本身已经按 embedding 模型隔离，不需要再在集合名上做区分。
COLLECTION_NAME = "local_kb"

# 支持的文档扩展名。集中定义可以避免 load_single_file、update_index、
# get_domains 三处各写一份，未来新增格式只改这里（再加一个 read_xxx）。
SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".docx", ".pptx", ".xlsx"}

# 固定切块大小。这里刻意不改成语义切块：
# 1) 现有索引就是按 1000 字切的，改切块策略会让旧索引与新查询不兼容；
# 2) 定长切块简单稳定，chunk 大小可控，对“检索后交给 LLM”场景够用；
# 3) 真正的语义切块需要额外的句子分割依赖，离线环境不一定装得上。
# 代价是可能从句子中间切断，注释里保留这个已知取舍。
CHUNK_SIZE = 1000

# 批量写入 Chroma 的批大小。一次把几万个 chunk 塞进 collection.add()
# 会让 embedding 阶段占用大量内存；分批写入可以显著降低峰值内存。
BATCH_SIZE = 256

# Web 设置页、CLI、MCP 在没有 app_settings.json 时使用的兜底配置。
# 注意：这些值必须和 app_settings.json 的字段名完全一致，
# 否则 merge 后会多出/缺少字段，导致 UI 读取 KeyError。
DEFAULT_SETTINGS: dict[str, Any] = {
    "docs_dir": "./docs",
    "embedding_model": "./models/bge-small-zh-v1.5",
    "use_llm": True,
    "llm_model": "qwen2.5:7b",
    "ollama_url": "http://localhost:11434",
    "selected_domain": "全部",
    "top_k": 4,
}


def load_app_settings(path: str = SETTINGS_FILE) -> dict[str, Any]:
    """读取 app_settings.json，并把缺失字段补齐为 DEFAULT_SETTINGS。

    为什么要 merge 而不是直接返回文件内容：
    - 老版本的配置文件可能没有 top_k / selected_domain 等新字段；
    - 用户手动删掉某个字段时，UI 不应该直接 KeyError；
    - 统一 merge 后，所有入口拿到的配置结构永远完整。

    文件不存在、JSON 损坏、权限不足时都回退到默认值，保证应用能启动。
    这是一个“可恢复配置”策略：宁可让用户看到默认设置，也不要白屏崩溃。
    """
    settings = dict(DEFAULT_SETTINGS)
    try:
        with open(path, "r", encoding="utf-8") as file:
            loaded = json.load(file)
        if isinstance(loaded, dict):
            settings.update(loaded)
    except (OSError, json.JSONDecodeError):
        # OSError：文件不存在/无权限；JSONDecodeError：文件被手动改坏。
        # 两种情况都退回默认值，不抛异常。
        pass
    return settings


def save_app_settings(settings: dict[str, Any], path: str = SETTINGS_FILE) -> None:
    """原子写入 app_settings.json。

    为什么用“临时文件 + os.replace”：
    - 直接 open(path, "w") 写到一半进程被杀，会留下半截 JSON；
    - 下次启动 load_app_settings 解析失败，用户的配置就全丢了；
    - 先写 path.tmp，再 os.replace 原子替换，可以保证文件要么是旧内容，
      要么是完整的新内容，不会出现“半截 JSON”。
    """
    merged = {**DEFAULT_SETTINGS, **settings}
    temp_path = f"{path}.tmp"
    with open(temp_path, "w", encoding="utf-8") as file:
        json.dump(merged, file, ensure_ascii=False, indent=2)
    os.replace(temp_path, path)


def resolve_project_path(path: str) -> str:
    """把配置里的相对路径解析为“相对项目根目录”的绝对路径。

    为什么需要它：
    - 用户可以在设置页填 "./docs"，也可以填 "~/knowledge"；
    - Streamlit / MCP / CLI 的 CWD 可能不同；
    - 统一在入口处解析一次，后续函数只处理绝对路径，逻辑最简单。
    """
    if not path:
        return PROJECT_ROOT
    expanded = os.path.expanduser(str(path))
    if os.path.isabs(expanded):
        return os.path.normpath(expanded)
    return os.path.normpath(os.path.join(PROJECT_ROOT, expanded))


def chroma_dir_for_model(embedding_model: str) -> str:
    """根据 embedding 模型推导向量库目录：./chromadb/{模型目录名}。

    为什么按模型分目录：
    - 不同 embedding 模型输出维度不同（如 384 / 1024），
      混在同一个 Chroma 集合里会直接报维度不匹配；
    - 分目录后，切换模型只是切换一个路径，互不影响；
    - 目录名取 basename，既短又直观，例如 bge-small-zh-v1.5。
    """
    model_short_name = os.path.basename(os.path.normpath(str(embedding_model)))
    if not model_short_name:
        model_short_name = "default-embedding"
    return os.path.join(PROJECT_ROOT, "chromadb", model_short_name)


# ==============================================================================
# 一、文档解析模块
# 目标：把不同格式的文件统一转成 [{text, page, section}]，上层只处理一种结构。
# ==============================================================================

def read_pdf(file_path: str) -> list[dict[str, Any]]:
    """按页读取 PDF，并尽量识别当前页所属章节。

    返回结构：
        [{"text": str, "page": int, "section": str}, ...]

    为什么章节状态要跨页保持：
    PDF 里“第5章”通常只出现在章节首页，后续页面不会重复。
    如果每页独立识别，大部分页面会变成“未知章节”；
    用一个 current_section 变量跨页继承，元数据才有实用价值。

    为什么异常只打印不抛出：
    知识库可能有几百个文件，单个损坏/加密/扫描版 PDF 不应该让整次
    建库失败。跳过坏文件、保留已解析内容，是更符合运维习惯的策略。
    """
    pages_data: list[dict[str, Any]] = []
    current_section = "未知章节"  # 跨页保存当前章节
    try:
        reader = PdfReader(file_path)
        for page_index, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""
            if not page_text.strip():
                # 扫描版 PDF 没有文本层，extract_text() 返回空。
                # 这里没有接 OCR，直接跳过空白页，避免写入无意义 chunk。
                continue

            # 只在前 5 行找章节标题：
            # 章节标题通常出现在页首；全文搜索容易把正文里的“5.1”误判成标题。
            lines = page_text.split("\n")
            for line in lines[:5]:
                match = re.match(r"^(第\s*\d+\s*章|\d+\.\d+)[\.\s]*(.*)", line.strip())
                if match:
                    # 截断到 30 字：元数据只需要“可识别的章节名”，
                    # 不需要存整行，避免 metadata 过大。
                    current_section = line.strip()[:30]
                    break

            pages_data.append({
                "text": page_text,
                "page": page_index + 1,  # 页码从 1 开始，方便用户核对原文
                "section": current_section,
            })
    except Exception as exc:
        # 单个 PDF 解析失败不应该让整个索引流程崩溃。
        print(f"读取 PDF 失败 {file_path}: {exc}")
    return pages_data


def read_docx(file_path: str) -> str:
    """读取 Word，把段落和表格都拼成纯文本。

    为什么表格不能漏：
    Word 里的关键数据（预算、人员、指标）经常只存在于表格中；
    如果只读 paragraphs，会丢掉大量业务信息。
    单元格之间用 " | " 分隔，保留行列感，方便 LLM 理解。
    """
    text = ""
    try:
        doc = Document(file_path)
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                text += paragraph.text + "\n"
        for table in doc.tables:
            for row in table.rows:
                row_text = " | ".join(cell.text.strip() for cell in row.cells)
                if row_text.strip():
                    text += row_text + "\n"
    except Exception as exc:
        print(f"读取 DOCX 失败 {file_path}: {exc}")
    return text


def read_pptx(file_path: str) -> str:
    """读取 PPT，把每页的文本框和表格抽成纯文本。

    PPT 的信息密度低、页面多，按 slide 顺序拼接即可；
    每页之间空一行，作为天然的段落边界，后续切块时更容易保持页内语义。
    """
    text = ""
    try:
        presentation = Presentation(file_path)
        for slide in presentation.slides:
            for shape in slide.shapes:
                if shape.has_text_frame:
                    for paragraph in shape.text_frame.paragraphs:
                        if paragraph.text.strip():
                            text += paragraph.text + "\n"
                elif shape.has_table:
                    for row in shape.table.rows:
                        row_text = " | ".join(cell.text.strip() for cell in row.cells)
                        if row_text.strip():
                            text += row_text + "\n"
            text += "\n"
    except Exception as exc:
        print(f"读取 PPTX 失败 {file_path}: {exc}")
    return text


def read_xlsx(file_path: str) -> str:
    """读取 Excel，按工作表导出为文本。

    关键参数：
    - read_only=True：流式读取，避免大表格把内存打满；
    - data_only=True：读公式的计算结果，而不是 "=SUM(...)" 字符串，
      否则 LLM 看到的是公式而不是业务数字。
    显式标注工作表名，避免多张表的内容混在一起无法区分。
    """
    text = ""
    try:
        workbook = load_workbook(file_path, read_only=True, data_only=True)
        for sheet_name in workbook.sheetnames:
            worksheet = workbook[sheet_name]
            text += f"--- 工作表: {sheet_name} ---\n"
            for row in worksheet.iter_rows(values_only=True):
                if any(cell is not None and str(cell).strip() for cell in row):
                    row_text = " | ".join(
                        str(cell).strip() if cell is not None else "" for cell in row
                    )
                    text += row_text + "\n"
            text += "\n"
        workbook.close()
    except Exception as exc:
        print(f"读取 XLSX 失败 {file_path}: {exc}")
    return text


def read_text_file(file_path: str) -> str:
    """读取 txt/md，按常见编码依次尝试。

    为什么不能只用 utf-8：
    中文知识库里大量旧文件是 GBK/GB18030 编码，直接 utf-8 读取会
    UnicodeDecodeError，导致整个 txt 被跳过。依次尝试：
    1) utf-8-sig：兼容 Windows 记事本带 BOM 的 UTF-8；
    2) utf-8：标准 UTF-8；
    3) gb18030：覆盖 GBK/GB2312 的中文超集；
    4) 最后用 errors="replace" 兜底，宁可出现少量乱码，也不丢整个文件。
    """
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            with open(file_path, "r", encoding=encoding) as file:
                return file.read()
        except UnicodeDecodeError:
            continue
    with open(file_path, "r", encoding="utf-8", errors="replace") as file:
        return file.read()


def extract_domain(file_path: str, docs_dir: str) -> str:
    """根据文件相对 docs_dir 的第一层子目录确定领域。

    设计意图：把目录结构直接当领域标签，零成本、零配置。
    例如 docs/财务/2025预算.pdf -> 财务；docs/uploaded/a.pdf -> uploaded；
    直接放在 docs 根目录的文件 -> general。

    为什么只取第一层：
    领域过滤需要稳定、可枚举的标签；层级太深会让下拉框出现大量碎片。
    第一层已经足够表达“财务/法务/技术”这类业务分类。
    """
    relative_path = os.path.relpath(file_path, docs_dir)
    parts = relative_path.split(os.sep)
    if len(parts) > 1 and parts[0] not in (".", ".."):
        return parts[0]
    return "general"


def load_single_file(file_path: str, docs_dir: str, chunk_size: int = CHUNK_SIZE) -> list[dict[str, Any]]:
    """解析单个文件并切块，返回可直接写入 Chroma 的 chunk 列表。

    返回值字段：
        id / text / source / domain / page / section

    为什么 id 用“文件绝对路径 MD5 + 序号”：
    - 同一文件重复解析会得到相同 id，天然幂等；
    - 不同目录下的同名文件路径不同，不会撞 id；
    - 增量更新时按 source 删除旧 chunk，再重新 add，id 可复现让过程更安全。
    注意：这里哈希的是“路径”而不是“内容”，因为内容变化时我们本来就会
    先删除该 source 的所有旧 chunk；如果哈希内容，反而会让“删除旧 chunk”
    失去稳定的 id 依据。
    """
    extension = os.path.splitext(file_path)[1].lower()
    if extension not in SUPPORTED_EXTENSIONS:
        return []

    # 统一抽象：不管什么格式，先转成“页列表”。
    # PDF 保留真实页码/章节；其它格式统一记为第 1 页，section 标记文档类型。
    pages_data: list[dict[str, Any]] = []
    try:
        if extension == ".pdf":
            pages_data = read_pdf(file_path)
        elif extension == ".docx":
            pages_data = [{"text": read_docx(file_path), "page": 1, "section": "Word 文档"}]
        elif extension == ".pptx":
            pages_data = [{"text": read_pptx(file_path), "page": 1, "section": "PPT 文档"}]
        elif extension == ".xlsx":
            pages_data = [{"text": read_xlsx(file_path), "page": 1, "section": "Excel 表格"}]
        else:  # .txt / .md
            pages_data = [{"text": read_text_file(file_path), "page": 1, "section": "纯文本"}]
    except Exception as exc:
        print(f"读取文件失败 {file_path}: {exc}")
        return []

    domain = extract_domain(file_path, docs_dir)
    absolute_path = os.path.abspath(file_path)
    file_hash = hashlib.md5(absolute_path.encode("utf-8")).hexdigest()

    chunks: list[dict[str, Any]] = []
    chunk_index = 0
    for page_data in pages_data:
        text = page_data["text"]
        # 定长切块。虽然可能切断句子，但实现简单、chunk 大小稳定；
        # 检索时返回的是完整 chunk，LLM 仍能看到上下文，影响通常可接受。
        for start in range(0, len(text), chunk_size):
            chunk_text = text[start:start + chunk_size].strip()
            if not chunk_text:
                # 纯空白 chunk 没有检索价值，直接丢弃。
                continue
            chunks.append({
                "id": f"{file_hash}_{chunk_index}",
                "text": chunk_text,
                "source": absolute_path,
                "domain": domain,
                "page": page_data["page"],
                "section": page_data["section"],
            })
            chunk_index += 1
    return chunks


def load_documents_from_directory(directory: str) -> list[dict[str, Any]]:
    """递归加载目录下所有支持的文档，返回 chunk 列表。

    细节：
    - 目录不存在时返回空列表而不是抛异常，让调用方决定如何提示；
    - sorted + 跳过隐藏目录，保证遍历顺序稳定，日志/调试更容易复现；
    - 跳过 .git/.idea/__pycache__ 等隐藏目录，避免把工程文件当知识库。
    """
    chunks: list[dict[str, Any]] = []
    if not os.path.isdir(directory):
        return chunks

    for root, dirs, filenames in os.walk(directory):
        dirs[:] = sorted(directory_name for directory_name in dirs if not directory_name.startswith("."))
        for filename in sorted(filenames):
            file_path = os.path.join(root, filename)
            chunks.extend(load_single_file(file_path, directory))
    return chunks


# ==============================================================================
# 二、文件指纹与增量索引辅助
# ==============================================================================

def compute_file_hash(file_path: str) -> str:
    """计算文件内容 MD5，作为“文件是否变化”的判据。

    为什么分块读取：
    知识库里可能有几十 MB 的 PDF，一次性 read() 会占用大量内存；
    4KB 分块读取的哈希结果完全一致，但内存占用是常数级。
    读取失败时返回空字符串，调用方会把该文件当作“未知/需要重试”，
    而不是让整个扫描中断。
    """
    hasher = hashlib.md5()
    try:
        with open(file_path, "rb") as file:
            for block in iter(lambda: file.read(4096), b""):
                hasher.update(block)
        return hasher.hexdigest()
    except Exception as exc:
        print(f"计算哈希失败 {file_path}: {exc}")
        return ""


def scan_supported_files(directory: str) -> dict[str, str]:
    """扫描目录下所有支持的文档，返回 {绝对路径: 内容 MD5}。

    这是全量重建和增量更新共用的“磁盘现状快照”。
    两边使用同一个函数，可以保证“判断哪些文件变了”的规则完全一致，
    不会出现 rebuild 认为文件在、update 认为文件不在的诡异情况。
    """
    files: dict[str, str] = {}
    if not os.path.isdir(directory):
        return files

    for root, dirs, filenames in os.walk(directory):
        dirs[:] = sorted(directory_name for directory_name in dirs if not directory_name.startswith("."))
        for filename in sorted(filenames):
            extension = os.path.splitext(filename)[1].lower()
            if extension not in SUPPORTED_EXTENSIONS:
                continue
            file_path = os.path.abspath(os.path.join(root, filename))
            files[file_path] = compute_file_hash(file_path)
    return files


def load_hash_map(hash_map_path: str = DEFAULT_HASH_MAP) -> dict[str, str]:
    """读取增量索引哈希表；不存在或损坏时返回空字典。

    返回空字典意味着“把当前磁盘上的文件都当作新增”，
    后续 update_index 会先按 source 删除再写入，因此不会产生重复 chunk。
    """
    if not os.path.exists(hash_map_path):
        return {}
    try:
        with open(hash_map_path, "r", encoding="utf-8") as file:
            loaded = json.load(file)
        return loaded if isinstance(loaded, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def save_hash_map(hash_map_path: str, hash_map: dict[str, str]) -> None:
    """原子写入哈希表，避免进程中断留下半截 JSON。"""
    temp_path = f"{hash_map_path}.tmp"
    with open(temp_path, "w", encoding="utf-8") as file:
        json.dump(hash_map, file, ensure_ascii=False, indent=2)
    os.replace(temp_path, hash_map_path)


# ==============================================================================
# 三、Embedding 与 ChromaDB
# ==============================================================================

class LocalEmbedding(EmbeddingFunction[Documents]):
    """把本地 SentenceTransformer 适配成 Chroma 的 EmbeddingFunction。

    为什么必须实现这些方法（chromadb >= 0.5.x 的契约）：
    - PersistentClient 重新打开集合时，需要从 collection 保存的 config
      重建 embedding 函数；只实现 __call__ 不够；
    - name() 是集合与 embedding 函数的绑定标识；
    - get_config()/build_from_config() 负责把“模型名”持久化并恢复。

    模型加载失败时不让整个应用启动就崩：
    - __init__ 里捕获异常，标记 _load_ok=False；
    - 真正调用 __call__ 时才抛 RuntimeError；
    - 这样 Web 页面还能打开，用户可以看到错误并去设置页修模型路径，
      而不是整个进程直接退出。
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        # 用目录名作为展示名/绑定名，例如 bge-small-zh-v1.5。
        # 对 HuggingFace ID（BAAI/bge-small-zh-v1.5）同样取最后一段。
        self._name = os.path.basename(os.path.normpath(str(model_name))) or "local-embedding"
        try:
            # 传本地路径时完全离线；传 HF ID 时依赖本地缓存。
            # 由于 rag_core 顶部强制 HF_HUB_OFFLINE=1，这里不会尝试联网。
            self.model = SentenceTransformer(model_name)
            self._load_ok = True
        except Exception as exc:
            print(f"[LocalEmbedding] 模型加载失败：{model_name}, err: {exc}")
            self.model = None
            self._load_ok = False

    def name(self) -> str:
        return self._name

    def __call__(self, input: Documents) -> Embeddings:
        if not self._load_ok or self.model is None:
            raise RuntimeError(f"Embedding 模型未成功加载：{self.model_name}")
        # show_progress_bar=False：
        # 1) MCP 是 stdio 协议，任何进度条输出都可能污染协议流（虽然 tqdm
        #    默认走 stderr，但关掉最稳妥）；
        # 2) Web 每次查询都刷进度条会干扰日志。
        embeddings = self.model.encode(
            input,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()

    def get_config(self) -> dict[str, str]:
        # 持久化时必须把模型名写进 collection 配置，
        # 否则进程重启后 Chroma 不知道用哪个模型重建 embedding 函数。
        return {"model_name": self.model_name}

    @classmethod
    def build_from_config(cls, config: dict[str, Any]):
        return cls(config["model_name"])


# 进程级 collection 缓存。
# 为什么需要：
# - Streamlit 每次交互都会从头执行 app.py；
# - 如果每次 rerun 都 new 一个 SentenceTransformer，页面会卡几秒甚至几十秒；
# - Chroma PersistentClient 也持有 SQLite 连接，复用比反复打开更稳。
# 为什么用锁：
# Streamlit 可能为多个浏览器标签页开多个线程；MCP 也可能并发调用。
# 锁只保护“查找/创建缓存”这一小段，不保护实际查询，避免把检索串行化。
_COLLECTION_CACHE: dict[tuple[str, str], Any] = {}
_COLLECTION_CACHE_LOCK = threading.RLock()


def _collection_cache_key(chroma_dir: str, embedding_model: str) -> tuple[str, str]:
    """缓存键必须同时包含向量库路径和模型名。

    同一个路径换成不同 embedding 模型时，不能复用旧 collection，
    否则查询向量维度会对不上。
    """
    return os.path.abspath(chroma_dir), str(embedding_model)


def get_chroma_collection(chroma_dir: str, embedding_model: str):
    """获取（或创建）Chroma 集合，并复用进程级缓存。

    这个函数是 Web / CLI / MCP 共享的入口：
    - 第一次调用：创建 PersistentClient、加载 embedding 模型、get_or_create；
    - 后续调用：直接返回缓存对象，避免重复加载模型；
    - rebuild_index 会主动清理缓存，保证重建后拿到的是新集合。
    """
    # 统一把相对路径解析到项目根目录，避免调用方 CWD 不同导致写到别处。
    chroma_dir = resolve_project_path(chroma_dir)
    key = _collection_cache_key(chroma_dir, embedding_model)
    with _COLLECTION_CACHE_LOCK:
        cached = _COLLECTION_CACHE.get(key)
        if cached is not None:
            return cached

        # 模型加载可能较慢，这里放在锁内是刻意为之：
        # 宁可让并发的第二个请求等一会儿，也不要同时加载两份模型把内存翻倍。
        client = chromadb.PersistentClient(path=chroma_dir)
        embed_fn = LocalEmbedding(embedding_model)
        collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=embed_fn,
        )
        _COLLECTION_CACHE[key] = collection
        return collection


def clear_collection_cache(chroma_dir: str | None = None, embedding_model: str | None = None) -> None:
    """清理 collection 缓存。

    使用场景：
    - 重建索引后，旧 collection 句柄已经失效；
    - 用户在设置页切换模型/目录，需要强制重新打开；
    - 排查“明明重建了却还查到旧数据”的问题。
    不传参数时清空全部；传参数时只清指定 key。
    """
    with _COLLECTION_CACHE_LOCK:
        if chroma_dir is None or embedding_model is None:
            _COLLECTION_CACHE.clear()
            return
        _COLLECTION_CACHE.pop(_collection_cache_key(chroma_dir, embedding_model), None)


def _add_chunks_in_batches(collection, chunks: list[dict[str, Any]]) -> None:
    """把 chunk 分批写入 Chroma，降低 embedding 阶段的峰值内存。

    ids/documents/metadatas 三个列表必须严格一一对应，
    这是 Chroma add() 的硬性要求；少一个就会整批失败。
    """
    for start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[start:start + BATCH_SIZE]
        collection.add(
            ids=[chunk["id"] for chunk in batch],
            documents=[chunk["text"] for chunk in batch],
            metadatas=[
                {
                    "source": chunk["source"],
                    "domain": chunk["domain"],
                    "page": chunk.get("page", 1),
                    "section": chunk.get("section", "未知"),
                }
                for chunk in batch
            ],
        )


# ==============================================================================
# 四、索引维护
# ==============================================================================

def rebuild_index(
    chroma_dir: str,
    docs_dir: str,
    embedding_model: str,
    hash_map_path: str = DEFAULT_HASH_MAP,
) -> int:
    """全量重建索引：删除旧集合 -> 重新解析 -> 重新向量化 -> 写入。

    什么时候必须全量重建：
    - 更换 embedding 模型；
    - 调整切块策略；
    - 增量索引状态异常；
    - 首次部署/迁移数据。
    全量重建会重新计算所有 chunk，速度慢，但结果最干净。

    为什么重建后要写一份新的 file_hashes.json：
    旧版本这里是“删除哈希表”，想法是让下次增量从零开始；但实际效果是
    下次 update_index 会把所有文件当新增，而集合里已经有同 ID 的 chunk，
    add() 可能报重复 ID。正确做法是重建成功后写入当前磁盘的完整指纹，
    这样下一次增量更新能准确判断新增/修改/删除。

    hash_map_path 做成参数而不是写死全局常量：
    - 测试时可以把指纹表放到临时目录，避免污染项目根目录；
    - 未来要支持多知识库实例时，每个实例可以有独立的指纹表。
    """
    # 先解析成绝对路径再检查目录，避免“CWD 下没有 docs、但项目根目录有”的误报。
    docs_dir = resolve_project_path(docs_dir)
    if not os.path.isdir(docs_dir):
        # 目录填错时直接报错，避免“扫描到 0 个文件 -> 把旧索引清空”的误操作。
        raise FileNotFoundError(f"文档目录不存在：{docs_dir}")

    chroma_dir = resolve_project_path(chroma_dir)
    hash_map_path = resolve_project_path(hash_map_path)
    key = _collection_cache_key(chroma_dir, embedding_model)

    # 先让旧 collection 句柄失效，再删除集合。
    with _COLLECTION_CACHE_LOCK:
        _COLLECTION_CACHE.pop(key, None)

    client = chromadb.PersistentClient(path=chroma_dir)
    embed_fn = LocalEmbedding(embedding_model)

    try:
        client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        # 集合本来就不存在时会抛异常，属于正常情况，忽略。
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
    )

    # 重建开始时先删掉旧哈希表：
    # 如果重建中途失败，旧哈希表已经和新集合不一致，留着反而会误导增量逻辑；
    # 删掉后下次 update_index 会把所有文件当新增，并先按 source 删除再写入，
    # 结果仍然是正确的（只是会慢一点）。
    try:
        os.remove(hash_map_path)
    except FileNotFoundError:
        pass

    all_chunks = load_documents_from_directory(docs_dir)
    if all_chunks:
        _add_chunks_in_batches(collection, all_chunks)

    # 重建成功后写入新的指纹表，供后续增量更新使用。
    save_hash_map(hash_map_path, scan_supported_files(docs_dir))

    # 把新集合放回缓存，下一次查询可以直接复用。
    with _COLLECTION_CACHE_LOCK:
        _COLLECTION_CACHE[key] = collection

    return len(all_chunks)


def update_index(
    chroma_dir: str,
    docs_dir: str,
    embedding_model: str,
    hash_map_path: str = DEFAULT_HASH_MAP,
) -> dict[str, int]:
    """增量更新：只处理新增/修改/删除的文件。

    核心思想：
    - 维护 {绝对路径: 内容 MD5} 的指纹表；
    - 对比“磁盘现状”和“上次索引时的指纹”，得到三类差异；
    - 删除的文件：按 source 元数据删除其所有 chunk；
    - 新增/修改的文件：先按 source 删除旧 chunk，再重新解析写入。

    为什么新增文件也要先 delete：
    为了兼容“有集合但没有哈希表”的历史状态（例如旧版本重建后删了哈希表）。
    此时所有文件都会被当成新增；如果直接 add，可能和已有 chunk 撞 ID。
    先 delete 再 add 可以保证幂等，代价只是多一次空删除。

    返回的是“文件数量”，不是 chunk 数量：
    - added：新增文件数
    - modified：修改文件数
    - deleted：删除文件数
    - failed：本次处理失败、留待下次重试的文件数
    这样 UI 上更容易解释“本次更新了几篇文档”。
    """
    docs_dir = resolve_project_path(docs_dir)
    if not os.path.isdir(docs_dir):
        raise FileNotFoundError(f"文档目录不存在：{docs_dir}")

    chroma_dir = resolve_project_path(chroma_dir)
    hash_map_path = resolve_project_path(hash_map_path)

    collection = get_chroma_collection(chroma_dir, embedding_model)
    indexed_hashes = load_hash_map(hash_map_path)
    current_files = scan_supported_files(docs_dir)

    new_files = {path: digest for path, digest in current_files.items() if path not in indexed_hashes}
    modified_files = {
        path: digest
        for path, digest in current_files.items()
        if path in indexed_hashes and indexed_hashes[path] != digest
    }
    deleted_files = [path for path in indexed_hashes if path not in current_files]

    # 记录本次处理失败的文件：这些文件不能写进新指纹表，
    # 否则下一轮增量会认为它们“已经索引成功”，永远不会重试。
    failed_files: set[str] = set()

    # 1) 删除已不存在文件的所有 chunk。
    for file_path in deleted_files:
        try:
            collection.delete(where={"source": file_path})
        except Exception as exc:
            print(f"删除索引失败 {file_path}: {exc}")
            failed_files.add(file_path)

    # 2) 新增/修改的文件：先删旧 chunk，再重新解析写入。
    for file_path in list(new_files.keys()) + list(modified_files.keys()):
        try:
            collection.delete(where={"source": file_path})
        except Exception as exc:
            print(f"删除旧索引失败 {file_path}: {exc}")
            failed_files.add(file_path)
            continue

        chunks = load_single_file(file_path, docs_dir)
        if not chunks:
            # 扫描版 PDF、空文件等解析不出文本：保留“已处理”状态，
            # 避免每次增量都重复解析同一个无内容文件。
            continue
        try:
            _add_chunks_in_batches(collection, chunks)
        except Exception as exc:
            print(f"添加索引失败 {file_path}: {exc}")
            failed_files.add(file_path)

    # 3) 指纹表覆盖为当前磁盘快照，但排除本次失败的文件。
    # 这样失败文件在下一轮增量中仍会被识别为“新增/修改”，自动重试。
    saved_hashes = {
        path: digest for path, digest in current_files.items()
        if path not in failed_files
    }
    # 已删除但删除失败的文件也保留旧哈希，下一轮还会尝试删除。
    for path in deleted_files:
        if path in failed_files and path in indexed_hashes:
            saved_hashes[path] = indexed_hashes[path]
    save_hash_map(hash_map_path, saved_hashes)

    return {
        "added": len(new_files),
        "modified": len(modified_files),
        "deleted": len(deleted_files),
        "failed": len(failed_files),
    }


# ==============================================================================
# 五、检索与领域
# ==============================================================================

def query_collection(
    chroma_dir: str,
    query: str,
    embedding_model: str,
    top_k: int = 4,
    domain: str | None = None,
    collection=None,
):
    """在 Chroma 中做相似度检索，返回 (documents, metadatas)。

    参数 collection 是给 Web 层复用的优化点：
    app.py 在页面顶部已经 get_chroma_collection() 一次，
    查询时直接传进来即可，避免 query_collection 内部再次打开 client /
    再次加载模型。CLI、MCP 不传时保持原来的行为。

    领域过滤：
    - domain 为 None / 空 / "全部" 时不加 where 条件；
    - 否则 where={"domain": domain}，只在该领域内检索。
    这样用户选了“财务”就不会召回到“法务”的片段。
    """
    if collection is None:
        collection = get_chroma_collection(chroma_dir, embedding_model)

    # 空集合直接返回，避免某些 Chroma 版本在 n_results > 0 时抛异常。
    if collection.count() == 0:
        return [], []

    query_kwargs: dict[str, Any] = {
        "query_texts": [query],  # Chroma 接口要求 list，即使只有一个 query
        "n_results": max(1, int(top_k)),
    }
    if domain and domain not in ("全部", ""):
        query_kwargs["where"] = {"domain": domain}

    results = collection.query(**query_kwargs)

    # Chroma 返回结构是“按 query 分组”的：documents[0] 才是第一个 query 的结果。
    documents = results.get("documents") or []
    metadatas = results.get("metadatas") or []
    if not documents:
        return [], []

    document_list = documents[0] or []
    metadata_list = list((metadatas[0] if metadatas else []) or [])
    # 某些 Chroma 版本/配置下 metadata 可能缺失或长度不足；
    # 补齐空字典，保证调用方 zip(documents, metadatas) 不会丢片段。
    if len(metadata_list) < len(document_list):
        metadata_list.extend({} for _ in range(len(document_list) - len(metadata_list)))
    return document_list, metadata_list


def get_domains(docs_dir: str) -> list[str]:
    """列出知识库中存在的领域，供 UI 下拉框和 MCP list_domains 使用。

    领域 = docs_dir 下第一层子目录名；如果根目录直接放了受支持的文档，
    额外增加一个 "general"。

    这里只统计受支持的扩展名，而不是“任意文件”：
    旧实现会把 .DS_Store 当成文件，导致下拉框永远多出一个 "general"；
    这是一个真实存在的小 bug，修复后只有真正的知识文件才产生领域。
    """
    if not os.path.isdir(docs_dir):
        return []

    domains: set[str] = set()
    has_root_documents = False
    for entry in os.scandir(docs_dir):
        if entry.is_dir(follow_symlinks=False):
            if not entry.name.startswith("."):
                domains.add(entry.name)
        elif entry.is_file(follow_symlinks=False):
            extension = os.path.splitext(entry.name)[1].lower()
            if extension in SUPPORTED_EXTENSIONS:
                has_root_documents = True

    if has_root_documents:
        domains.add("general")
    return sorted(domains)

# ============================================================
# 【文件作用总览】
# 这是一个 MCP（Model Context Protocol）服务器，用 stdio 传输协议
# 把本地 RAG 的检索能力暴露给任意支持 MCP 的客户端（Claude Desktop、
# Cursor、各种 Agent 框架等）。
#
# 对外提供两个工具：
#   1) search_knowledge_base：按 query 检索知识库，可按 domain 过滤；
#   2) list_domains：列出知识库里有哪些领域，供客户端“先查领域再检索”。
#
# 【为什么用 stdio，而不是 HTTP】
# - stdio 是 MCP 的标准传输方式之一：客户端启动本进程，通过 stdin/stdout
#   收发 JSON-RPC，不需要开端口、不需要鉴权，天然隔离；
# - 对本地桌面客户端最友好，配置一行 command 就能用；
# - 代价是绝对不能往 stdout 打印调试信息（见下方“stdio 禁忌”）。
#
# 【stdio 禁忌（最容易踩的坑）】
# print() 默认写 stdout，会直接混进 JSON-RPC 消息流，客户端解析就会报错。
# 本文件禁止使用 print()；调试信息一律走 sys.stderr 或日志文件。
# 同理，任何第三方库如果往 stdout 打进度条，也要关闭/重定向
# （rag_core 里 SentenceTransformer 已经设置 show_progress_bar=False）。
#
# 【配置来源】
# 旧版本把 CHROMA_DIR / EMBEDDING_MODEL 写死为 ./chroma_db + HF ID，
# 而 Web 端用的是 ./chromadb/{模型名} + 本地路径，导致 MCP 经常检索到空库。
# 现在每次工具调用都重新读取 app_settings.json，并用
# chroma_dir_for_model() 推导路径，保证 MCP 与 Web 共享同一份索引。
# 每次调用都读文件的开销可以忽略，但能实时感知用户在设置页的修改。
# ============================================================

# ---------- 标准库导入 ----------
import asyncio  # MCP server 基于 asyncio；用 to_thread 包同步检索，避免阻塞事件循环
import os       # 取文件名 basename，减少返回给模型的路径噪声

# ---------- MCP 协议导入 ----------
from mcp.server import NotificationOptions, Server
from mcp.server.models import InitializationOptions
import mcp.server.stdio
import mcp.types as types

# ---------- 项目核心能力 ----------
from rag_core import (
    chroma_dir_for_model,
    get_domains,
    load_app_settings,
    query_collection,
    resolve_project_path,
)

# 服务器标识：握手时传给客户端，用于展示和日志。
SERVER_NAME = "local-rag-server"
SERVER_VERSION = "2.0.0"
DEFAULT_TOP_K = 4
MAX_TOP_K = 20  # 防止客户端一次要求过多片段，撑爆模型上下文
server = Server(SERVER_NAME)


def get_runtime_config() -> dict:
    """读取当前运行时配置。

    为什么不在模块加载时算一次常量：
    MCP 客户端通常会长期挂着一个 server 进程；用户在 Web 设置页
    改了 docs_dir / embedding_model / top_k 后，如果 MCP 只在启动时读一次，
    就会一直用旧配置，表现为“设置改了但 MCP 检索不到新内容”。
    每次工具调用重新读 app_settings.json，成本只有一次几 KB 的文件读取，
    却能让配置实时生效，是更符合本地工具使用习惯的选择。
    """
    settings = load_app_settings()
    embedding_model = str(settings["embedding_model"])
    try:
        configured_top_k = int(settings.get("top_k", DEFAULT_TOP_K))
    except (TypeError, ValueError):
        configured_top_k = DEFAULT_TOP_K
    return {
        "docs_dir": resolve_project_path(settings["docs_dir"]),
        "embedding_model": embedding_model,
        "chroma_dir": chroma_dir_for_model(embedding_model),
        "top_k": max(1, configured_top_k),
    }


@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """返回工具菜单。

    description 是给 LLM 看的，不是给人类看的：
    - 写清楚“什么时候用”比写“这个工具能做什么”更重要；
    - domain 参数要明确提示“先调 list_domains 获取合法值”，
      否则模型容易凭空编一个领域名，导致检索为空；
    - inputSchema 让客户端做参数校验和 UI 渲染，服务端仍会再做一次防御。
    """
    return [
        types.Tool(
            name="search_knowledge_base",
            description=(
                "在本地知识库中检索与查询相关的文档片段。"
                "可通过 domain 参数过滤特定领域；"
                "不传 domain 则检索全部领域。"
                "可用领域名请先调用 list_domains 工具获取。"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "用户查询的文本，尽量使用完整、具体的自然语言问题",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": f"返回的片段数量，默认 {DEFAULT_TOP_K}，最大 {MAX_TOP_K}",
                        "default": DEFAULT_TOP_K,
                    },
                    "domain": {
                        "type": "string",
                        "description": (
                            "知识领域名称，例如 uploaded、财务、法务。"
                            "不传则检索全部领域；可用领域需通过 list_domains 获取。"
                        ),
                    },
                },
                "required": ["query"],
            },
        ),
        types.Tool(
            name="list_domains",
            description="列出当前知识库中所有可用的领域名称。建议在检索前先调用，确认 domain 的合法取值。",
            inputSchema={"type": "object", "properties": {}},
        ),
    ]


def _format_results(documents: list, metadatas: list, domain: str | None) -> str:
    """把检索结果格式化成一段适合 LLM 阅读的文本。

    为什么把所有片段拼成一个 TextContent：
    - 客户端渲染简单；
    - 模型一次看到完整上下文，便于跨片段交叉引用；
    - 片段之间用空行分隔，避免不同来源的内容粘在一起。

    元数据里保留文件名而不是完整绝对路径：
    - 绝对路径又长又泄露本机目录结构，对模型回答没有帮助；
    - 页码/章节保留，方便用户核对原文。
    """
    if not documents:
        return f"在领域「{domain or '全部'}」中未找到相关内容。"

    formatted: list[str] = []
    for index, (document, metadata) in enumerate(zip(documents, metadatas), start=1):
        metadata = metadata or {}
        source = os.path.basename(metadata.get("source", "未知来源"))
        result_domain = metadata.get("domain", "未知领域")
        page = metadata.get("page", "未知")
        section = metadata.get("section", "未知")
        formatted.append(
            f"[片段 {index}] 领域: {result_domain} | 来源: {source} | "
            f"第 {page} 页 | 章节: {section}\n{document}\n"
        )
    return "\n".join(formatted)


@server.call_tool()
async def handle_call_tool(
    name: str, arguments: dict | None
) -> list[types.TextContent]:
    """执行客户端发起的工具调用。

    返回值必须是 list[TextContent]（MCP 规定工具结果以内容块列表返回）。
    业务性空结果（没检索到）用正常文本返回，不要抛异常：
    - 抛异常在协议层意味着“工具坏了”，模型可能重试或放弃；
    - 返回“未找到相关内容”让模型知道工具正常工作，只是库里没有，
      它可以据此告诉用户或换一个查询词。
    只有真正的参数错误/未知工具才抛异常。
    """
    config = get_runtime_config()

    # ---------- 工具 1：search_knowledge_base ----------
    if name == "search_knowledge_base":
        if not arguments or not str(arguments.get("query", "")).strip():
            raise ValueError("缺少参数 query")

        query = str(arguments["query"]).strip()

        # 客户端可能传字符串、None 或超出范围的值，这里统一兜底并夹紧。
        try:
            requested_top_k = int(arguments.get("top_k", config["top_k"]))
        except (TypeError, ValueError):
            requested_top_k = config["top_k"]
        top_k = max(1, min(requested_top_k, MAX_TOP_K))

        domain_value = arguments.get("domain")
        domain = str(domain_value).strip() if domain_value else None
        if domain in ("全部", ""):
            domain = None

        try:
            # query_collection 是同步函数（内部有 CPU 密集的向量计算和 IO）。
            # 直接放在 async 函数里会阻塞事件循环；用 to_thread 放到线程池，
            # 未来即使并发调用也不会互相卡住。
            documents, metadatas = await asyncio.to_thread(
                query_collection,
                config["chroma_dir"],
                query,
                config["embedding_model"],
                top_k=top_k,
                domain=domain,
            )
        except Exception as exc:
            # 把“知识库暂时不可用”作为正常文本返回，而不是协议错误。
            # 模型可以据此告诉用户去检查模型路径/索引，而不是反复重试。
            return [types.TextContent(
                type="text",
                text=f"知识库检索失败：{exc}",
            )]

        return [types.TextContent(
            type="text",
            text=_format_results(documents, metadatas, domain),
        )]

    # ---------- 工具 2：list_domains ----------
    if name == "list_domains":
        try:
            domains = await asyncio.to_thread(get_domains, config["docs_dir"])
        except Exception as exc:
            return [types.TextContent(
                type="text",
                text=f"读取知识库领域失败：{exc}",
            )]

        if not domains:
            return [types.TextContent(
                type="text",
                text="未找到任何领域。请先在 Web 设置页上传文档并重建索引。",
            )]
        return [types.TextContent(
            type="text",
            text=f"可用领域：{', '.join(domains)}",
        )]

    # ---------- 未知工具 ----------
    # 客户端传了未注册的工具名（版本不匹配/恶意调用），
    # 抛 ValueError 让协议层返回标准错误，而不是静默忽略。
    raise ValueError(f"未知工具: {name}")


async def main() -> None:
    """启动 stdio 服务器。

    注意：stdio 模式下绝对不能在 stdout 打印任何调试信息，
    否则会污染 JSON-RPC 协议流，导致客户端解析失败。
    需要调试时请使用 sys.stderr 或写日志文件。
    """
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name=SERVER_NAME,
                server_version=SERVER_VERSION,
                # get_capabilities 根据已注册的 handler 自动推导能力集合：
                # 我们注册了 list_tools + call_tool，所以 capabilities 里会带 tools。
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities=None,
                ),
            ),
        )


if __name__ == "__main__":
    # 以脚本方式运行时启动事件循环；被作为模块导入时不会自动启动。
    asyncio.run(main())

# ============================================================
# 【文件作用总览】
# 这是索引构建的命令行入口，负责把 rag_core 的建库能力暴露给终端：
#   python build_index.py                 # 按 app_settings.json 全量重建
#   python build_index.py --incremental   # 只处理新增/修改/删除的文件
#
# 【为什么还需要这个脚本】
# Web 设置页的“重建索引”按钮适合普通用户，但在以下场景 CLI 更方便：
#   1) 首次部署/迁移数据：不用启动 Web 服务，也不用登录；
#   2) CI/CD、Docker 启动脚本、定时任务：可以直接跑命令；
#   3) 排障：终端能看到完整 print 和异常栈；
#   4) 批量建库：可以传 --docs_dir / --embedding_model 覆盖配置。
#
# 【与 Web、MCP 的关系】
# - app.py：交互式建库（设置页按钮），底层也是 rag_core.rebuild_index；
# - build_index.py：离线/批量建库（本文件）；
# - mcp_server.py：在线检索，不负责建库。
# 三者共用 rag_core，保证建库/检索逻辑只有一份实现。
# ============================================================

# ---------- 标准库导入 ----------
import argparse  # 标准库参数解析：自带 --help、类型转换、错误提示
import os        # 路径检查
import sys       # 退出码 + stderr 输出

# 复用核心模块，避免在 CLI 里重复实现解析/切块/入库逻辑。
from rag_core import (
    chroma_dir_for_model,
    load_app_settings,
    rebuild_index,
    resolve_project_path,
    update_index,
)


def parse_args() -> argparse.Namespace:
    """定义并解析命令行参数。

    参数名使用下划线（--docs_dir）而不是连字符（--docs-dir）：
    argparse 默认会把 --foo-bar 转成 args.foo_bar，两种写法都能用；
    这里统一用下划线，是为了和 app_settings.json 的字段名保持一致，
    减少“配置里叫 docs_dir、命令行叫 docs-dir”的心智负担。
    """
    parser = argparse.ArgumentParser(
        description="本地 RAG 知识库索引构建工具（默认读取 app_settings.json）",
    )
    parser.add_argument(
        "--docs_dir",
        type=str,
        default=None,
        help="文档目录；不传时读取 app_settings.json 的 docs_dir",
    )
    parser.add_argument(
        "--chroma_dir",
        type=str,
        default=None,
        help="Chroma 持久化目录；不传时按 embedding 模型自动推导为 ./chromadb/{模型名}",
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        default=None,
        help="Embedding 模型本地路径或 HF ID；不传时读取 app_settings.json",
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="增量更新：只处理新增/修改/删除的文件（默认全量重建）",
    )
    return parser.parse_args()


def main() -> int:
    """CLI 主流程：解析参数 -> 解析路径 -> 调 rag_core -> 打印结果。

    返回退出码而不是直接 sys.exit()：
    - 0 表示成功；
    - 1 表示失败；
    - 方便被 shell / CI 判断，也方便未来写单元测试。
    """
    args = parse_args()

    # 读取与 Web 设置页同一份配置，保证 CLI 默认行为与 UI 一致。
    settings = load_app_settings()

    # 命令行显式传参 > app_settings.json > rag_core 默认值。
    docs_dir = resolve_project_path(args.docs_dir or settings["docs_dir"])
    embedding_model = args.embedding_model or settings["embedding_model"]
    chroma_dir = (
        resolve_project_path(args.chroma_dir)
        if args.chroma_dir
        else chroma_dir_for_model(embedding_model)
    )

    print("=" * 64)
    print("本地 RAG 索引构建")
    print(f"  文档目录   : {docs_dir}")
    print(f"  向量库目录 : {chroma_dir}")
    print(f"  Embedding  : {embedding_model}")
    print(f"  模式       : {'增量更新' if args.incremental else '全量重建'}")
    print("=" * 64)

    if not os.path.isdir(docs_dir):
        print(f"错误：文档目录不存在：{docs_dir}", file=sys.stderr)
        return 1

    try:
        if args.incremental:
            stats = update_index(chroma_dir, docs_dir, embedding_model)
            print(
                "增量更新完成！"
                f"新增 {stats['added']} 个文件，"
                f"修改 {stats['modified']} 个文件，"
                f"删除 {stats['deleted']} 个文件，"
                f"失败 {stats.get('failed', 0)} 个文件。"
            )
            if stats.get("failed"):
                print("提示：失败文件已保留旧指纹，下次增量更新会自动重试。")
        else:
            chunk_count = rebuild_index(chroma_dir, docs_dir, embedding_model)
            print(f"全量重建完成！共写入 {chunk_count} 个文档片段。")
            # 为 0 通常意味着目录路径错了、文件格式不支持，或全部是扫描版 PDF。
            if chunk_count == 0:
                print("提示：没有写入任何片段，请检查目录中是否有受支持的文档。")
    except Exception as exc:
        # 捕获后统一打印到 stderr，返回非零退出码；不要把异常栈直接吞掉。
        print(f"索引构建失败：{exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    # 只有“直接运行本文件”时才执行；被其它模块 import 时不会自动建索引，
    # 避免产生副作用（这是 Python 脚本的标准写法）。
    raise SystemExit(main())

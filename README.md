# 本地 RAG 助手

一个完全本地运行的文档问答系统：**Streamlit 界面 + ChromaDB 向量库 + 本地 SentenceTransformer + Ollama 生成**。文档解析、向量化、检索和生成都在本机完成，不依赖任何云服务。

> 技术栈：Streamlit · ChromaDB · SentenceTransformer · Ollama · MCP


## 一、它能做什么

- **多格式知识库**：支持 PDF、Word（docx）、PPT（pptx）、Excel（xlsx）、Markdown、纯文本。
- **可追溯回答**：每条回答都可以展开查看命中的原始片段，包含来源文件、页码、章节；片段正文会自动识别 Markdown 标题/列表/表格、代码块和 LaTeX 公式。
- **多会话管理**：新建、重命名、置顶、删除对话；历史记录持久化到 `chat_sessions.json`。
- **领域过滤**：把文档按子目录分类（如 `docs/财务/`），检索时可只查某个领域。
- **两种索引模式**：全量重建 + 增量更新；文档少改时不用每次重算全部向量。
- **本地 LLM**：通过 Ollama 调用 `qwen2.5` 等模型，流式输出，支持随时停止并保留半成品。
- **MCP 接入**：`mcp_server.py` 把检索能力暴露给 Claude Desktop、Cursor 等 MCP 客户端。
- **离线优先**：强制 `HF_HUB_OFFLINE=1`，模型从 `./models/` 本地加载；界面使用系统字体，不依赖外部 CDN。


## 二、项目结构

```text
rag-project/
├── app.py                  # Streamlit Web 入口：登录、会话、设置、聊天、Ollama 流式生成
├── rag_core.py             # 核心引擎：配置、解析、切块、Embedding、Chroma、检索
├── build_index.py          # 命令行建库：全量重建 / 增量更新
├── mcp_server.py           # MCP stdio 服务器：search_knowledge_base + list_domains
├── app_settings.json       # Web / CLI / MCP 共享配置
├── auth_config.yaml        # 登录账号、密码哈希、cookie 配置
├── chat_sessions.json      # 会话历史（运行时自动写入）
├── requirements.txt        # Python 依赖
├── .streamlit/
│   └── config.toml         # Streamlit 主题色（蓝色主色、浅色画布）
├── docs/                   # 知识库原始文档目录
│   └── uploaded/           # 设置页上传后自动同步到这里
├── uploaded_docs/          # 上传暂存目录
├── chromadb/               # 向量库持久化目录，按 embedding 模型分目录
│   └── bge-small-zh-v1.5/
├── models/                 # 本地 embedding 模型（体积较大，不要提交到版本库）
├── skills/                 # 第三方 UI/UX 设计技能资料（与运行时无关）
└── README.md
```

### 关键路径说明

| 路径 | 作用 |
|---|---|
| `docs/` | 建库时递归扫描的根目录；第一层子目录自动成为领域 |
| `docs/uploaded/` | 设置页上传文件后复制到这里，统一参与建库 |
| `uploaded_docs/` | file_uploader 的暂存区，保存后等待建库 |
| `chromadb/{模型名}/` | 每个 embedding 模型独立的 Chroma 持久化目录 |
| `file_hashes.json` | 增量索引的文件指纹表，重建/增量更新时自动维护 |
| `chat_sessions.json` | 所有会话的标题、消息、引用片段、置顶状态 |


## 三、环境准备

### 1. Python 环境

推荐 Python 3.10+（本项目在 Python 3.12 + conda 环境 `rag_project` 下验证）。

```bash
conda create -n rag_project python=3.12 -y
conda activate rag_project
python -m pip install -r requirements.txt
```

> `sentence-transformers` 会自动安装 `torch`。如果你需要 GPU 版 torch，请先按平台安装匹配的 torch，再安装其余依赖。

### 2. 本地 Embedding 模型

把模型放到 `models/` 下，例如：

```text
models/
├── bge-small-zh-v1.5/
├── all-MiniLM-L6-v2/
└── bge-m3/
```

`app_settings.json` 默认使用 `./models/bge-small-zh-v1.5`。设置页会自动扫描 `models/` 下的目录，不需要改代码。

> 本项目强制离线加载：`HF_HUB_OFFLINE=1`、`HF_DATASETS_OFFLINE=1`。如果模型目录缺失，启动不会崩，但检索时会提示模型未加载。

### 3. Ollama

```bash
# 安装并启动 Ollama 后，拉取一个模型
ollama pull qwen2.5:7b

# 确认服务在运行
curl http://localhost:11434/api/tags
```

在设置页填写：

- Ollama 模型名：`qwen2.5:7b`（或你实际拉取的模型）
- Ollama API 地址：`http://localhost:11434`

如果只想调试检索质量，可以关闭“启用本地 LLM 生成答案”，系统会只返回检索片段。

### 4. 登录配置

复制并修改 `auth_config.yaml`：

```yaml
credentials:
  usernames:
    yourname:
      email: you@example.com
      name: Your Name
      password: "$2b$12$..."   # bcrypt 哈希，不是明文
cookie:
  expiry_days: 30
  key: "placeholder_will_be_overwritten"
  name: rag_auth_cookie
preauthorized:
  emails:
    - you@example.com
```

生成密码哈希：

```bash
python -c "import streamlit_authenticator as stauth; print(stauth.Hasher.hash('你的密码'))"
```

把输出粘贴到 `password` 字段。

**Cookie 签名密钥**：项目根目录的 `.env` 里放：

```dotenv
COOKIE_KEY="一个足够长的随机字符串"
```

`app.py` 会在启动时读取 `.env`，并用 `COOKIE_KEY` 覆盖 `auth_config.yaml` 里的 `cookie.key`。修改这个值会让所有已登录用户重新登录。不要把它提交到版本库。

生成随机密钥：

```bash
python -c "import secrets; print(secrets.token_hex(32))"
```


## 四、启动与使用

### 启动 Web 界面

```bash
streamlit run app.py
```

浏览器打开终端提示的地址（默认 `http://localhost:8501`），使用 `auth_config.yaml` 中的账号登录。

### 首次建库

1. 把文档放进 `docs/`，或直接使用设置页的“文件上传”；
2. 进入 **设置 → 知识库与索引**，确认文档目录和 Embedding 模型；
3. 点击 **重建索引**；
4. 回到聊天页提问。

上传流程：

```text
file_uploader
  -> uploaded_docs/（暂存）
  -> 点击“重建索引/增量更新”
  -> docs/uploaded/（正式参与建库）
  -> Chroma 向量库
```

> 上传文件会被复制到 `docs/uploaded/`，之后即使从暂存区删除，知识库里的副本仍然存在。要彻底移除某个文件，请删除 `docs/uploaded/` 中的对应文件，再执行一次“增量更新”或“重建索引”。

### 命令行建库

```bash
# 全量重建（默认读取 app_settings.json）
python build_index.py

# 只处理新增/修改/删除的文件
python build_index.py --incremental

# 覆盖配置
python build_index.py \
  --docs_dir ./docs \
  --embedding_model ./models/bge-small-zh-v1.5

# 显式指定向量库目录
python build_index.py \
  --docs_dir ./docs \
  --chroma_dir ./chromadb/bge-small-zh-v1.5 \
  --embedding_model ./models/bge-small-zh-v1.5
```

> 不传 `--chroma_dir` 时，CLI 会按 `./chromadb/{embedding 模型目录名}` 自动推导，和 Web 端保持一致。
>
> 增量更新依赖 `file_hashes.json`；如果该文件不存在，会把当前文件都当作新增处理。新增/修改文件会先按 `source` 删除旧片段，再重新写入，因此不会产生重复 chunk。

### 领域过滤

领域 = `docs_dir` 下第一层子目录名：

```text
docs/
├── 财务/
│   └── 预算.pdf        -> domain = 财务
├── 法务/
│   └── 合同.docx       -> domain = 法务
└── 说明.md             -> domain = general
```

在设置页选择“知识领域过滤”，聊天时只检索该领域。`uploaded/` 也是一个领域，所以通过上传入口进来的文件默认属于 `uploaded`。

### 停止生成

生成过程中点击“停止生成”或底部“停止”按钮：

- 已经生成的部分会保存到会话；
- 消息末尾会标注“已手动停止生成”；
- 不会丢失已经输出的内容。


## 五、MCP Server

`mcp_server.py` 通过 stdio 提供两个工具：

| 工具 | 作用 |
|---|---|
| `search_knowledge_base` | 按 query 检索，可传 `domain`、`top_k` |
| `list_domains` | 列出可用领域，建议检索前先调用 |

MCP 配置示例（路径改成你的实际路径）：

```json
{
  "mcpServers": {
    "local-rag": {
      "command": "python",
      "args": ["/Users/you/rag-project/mcp_server.py"]
    }
  }
}
```

前提：

```bash
pip install mcp
```

> `command` 要指向安装了本项目依赖的 Python（例如 conda 环境的绝对路径 `/Users/you/miniconda3/envs/rag_project/bin/python`），否则客户端可能找不到 `mcp` 或 `rag_core`。

MCP server 每次工具调用都会重新读取 `app_settings.json`，所以在 Web 设置页修改目录/模型后不需要重启 MCP 进程。

> **stdio 注意**：`mcp_server.py` 绝对不能用 `print()` 往 stdout 输出调试信息，否则会污染 JSON-RPC 协议流。调试请用 `sys.stderr` 或日志文件。


## 六、配置说明

`app_settings.json`：

```json
{
  "docs_dir": "./docs",
  "embedding_model": "./models/bge-small-zh-v1.5",
  "use_llm": true,
  "llm_model": "qwen2.5:14b",
  "ollama_url": "http://localhost:11434",
  "selected_domain": "全部",
  "top_k": 10
}
```

| 字段 | 说明 |
|---|---|
| `docs_dir` | 原始文档目录，支持绝对路径或相对项目根目录 |
| `embedding_model` | 本地模型路径（推荐）或 HF 模型 ID |
| `use_llm` | `false` 时只检索、不生成 |
| `llm_model` | Ollama 模型名 |
| `ollama_url` | Ollama 服务地址 |
| `selected_domain` | 当前领域过滤，`全部` 表示不过滤 |
| `top_k` | 每次检索返回的片段数 |

设置页保存后会同时影响 Web、CLI 和 MCP。


## 七、实现要点

### 文档解析

所有格式先统一成 `[{text, page, section}]`，再走同一套切块逻辑：

- PDF：逐页提取，识别“第 N 章 / N.N”章节；
- Word：段落 + 表格；
- PPT：逐页文本框 + 表格；
- Excel：逐工作表导出，保留工作表名；
- txt/md：依次尝试 `utf-8-sig`、`utf-8`、`gb18030`，最后用替换字符兜底。

扫描版 PDF 没有文本层，当前不包含 OCR，无法直接检索。

### 切块与 ID

- 固定 1000 字切块，不做语义切块（保持与现有索引兼容）；
- chunk id = `文件绝对路径 MD5_序号`，同一文件重复解析得到相同 ID；
- 元数据包含 `source / domain / page / section`；
- 增量更新按 `source` 删除旧片段，再写入新片段。

### 向量库

- 集合名固定为 `local_kb`；
- 每个 embedding 模型一个独立目录，避免向量维度冲突；
- `LocalEmbedding` 实现 chromadb 的 `name / __call__ / get_config / build_from_config` 契约；
- 进程级 collection 缓存避免每次 Streamlit rerun 都重新加载模型。

### 生成

- Ollama `/api/generate`，`stream=True`，NDJSON 逐行解析；
- `temperature=0.1`、`top_p=0.8`、`repeat_penalty=1.1`，降低发散和重复；
- 系统提示词要求只依据参考资料、分层次回答、标注片段出处；
- 公式统一转换为 `$` / `$$`，适配 Streamlit Markdown。


## 八、界面设计

界面参考了 [UIUX-Demo-Design](https://github.com/Aiminnotes/UIUX-Demo-Design) 中的企业级 AI SaaS / UI/UX 设计规范：

- 浅灰蓝画布 + 白色卡片 + 低对比边框 + 柔和阴影；
- 主色 `#2563EB`，AI 强调色 `#7C3AED`，避免过度使用渐变；
- 4pt 间距体系，卡片圆角 16–20px，状态标签使用胶囊；
- 侧边栏固定会话列表，底部设置/退出始终可见；
- 超宽屏（侧边栏收起 ≥1900px / 展开 ≥2000px）时，聊天页的标题、状态和指标卡会固定在右侧空白区，滚动对话时始终可见；窄屏自动回到页头；
- 空状态、加载状态、错误状态都有明确提示；
- 所有按钮有 hover / active / focus-visible 反馈；
- 尊重 `prefers-reduced-motion`，关闭动画不影响功能；
- 使用系统字体（SF Pro / PingFang SC），离线环境不依赖 Google Fonts。

UI 通过 `app.py` 中的 `THEME_CSS` + `.streamlit/config.toml` 实现，没有引入新的前端框架或构建步骤。后者负责 Streamlit 原生组件（滑块、复选框、Tab 指示器）的主色，前者负责卡片、侧边栏、聊天消息、空状态等细节样式；两者使用同一套蓝色 token。


## 九、常见问题

### 1. 页面提示“初始化向量库失败”

- 检查 `models/` 下模型目录是否存在；
- 检查 `app_settings.json` 的 `embedding_model` 是否指向正确路径；
- 如果刚切换过模型，需要重建索引；
- 在聊天页点击“重新加载向量库”清理缓存后重试。

### 2. 索引为空 / 检索不到内容

- 确认 `docs_dir` 目录存在且包含受支持的文件；
- 确认点击过“重建索引”或“增量更新”；
- 扫描版 PDF 需要先做 OCR；
- 复选框 `use_llm` 只影响生成，不影响检索。

### 3. Ollama 调用失败

```bash
curl http://localhost:11434/api/tags
ollama list
```

确认模型名和设置页一致，服务已启动。默认读取超时 300 秒，本地大模型生成长回答时请耐心等待。

### 4. 登录后立刻掉线 / 修改密钥后需要重新登录

`COOKIE_KEY` 改变会使已有 cookie 失效，这是预期行为。重新登录即可。生产环境请使用足够长的随机密钥，不要使用 `auth_config.yaml` 里的占位值。

### 5. MCP 客户端提示协议解析错误

检查 `mcp_server.py` 是否有任何 `print()` 输出到 stdout。所有调试信息必须走 stderr。

### 6. 端口被占用

```bash
streamlit run app.py --server.port 8502
```

## 十、开发说明

项目没有自动化测试套件。修改代码后可以做基础验证：

```bash
# 语法检查
python -m py_compile app.py rag_core.py build_index.py mcp_server.py

# 检查核心模块能否导入
python -c "import rag_core; print(rag_core.load_app_settings())"

# 查看 CLI 参数
python build_index.py --help

# 启动 Web
streamlit run app.py
```

### 数据与版本控制

以下内容体积大或包含隐私/密钥，不建议提交到版本库：

- `models/`（模型权重，数 GB）
- `chromadb/`（向量库）
- `docs/`、`uploaded_docs/`（业务文档）
- `chat_sessions.json`（聊天记录）
- `.env`（cookie 密钥）
- `file_hashes.json`（索引指纹）

`app_settings.json` 和 `auth_config.yaml` 是否提交取决于团队规范；如果提交，请确认里面没有明文密码或真实密钥。


## 十一、已知限制

- 扫描版 PDF 不含 OCR，无法检索图片中的文字；
- 固定长度切块可能截断句子，重要内容建议单独成段；
- 默认只取 `docs_dir` 第一层子目录作为领域，层级更深不会细分；
- 向量库和模型都在本机，换机器需要重新拷贝 `models/` 并重建索引；
- 当前没有用户注册/权限分级，账号配置在 `auth_config.yaml` 中手工维护；
- `skills/` 是第三方 UI/UX 设计资料，不参与 RAG 运行时。

# ============================================================
# 【文件作用总览】
# 这是本地 RAG 的 Streamlit Web 入口，负责“人机交互”这一层：
#   1) 登录鉴权：streamlit-authenticator + cookie 免登录；
#   2) 会话管理：新建/重命名/置顶/删除多会话；
#   3) 设置页：文档目录、embedding 模型、Ollama、TopK、领域、上传；
#   4) 聊天页：检索知识库 -> 拼接上下文 -> 调用 Ollama 流式生成；
#   5) 引用展示：每条回答都能展开查看命中的原始片段；
#   6) 停止生成：保留已经生成的半成品，不丢内容。
#
# 【Streamlit 的核心机制（理解这个才能改对代码）】
# Streamlit 每次用户交互（点按钮、发消息、改输入框）都会把整个脚本
# 从头到尾重新执行一遍。普通 Python 变量在两次运行之间会丢失，
# 所以所有需要跨运行保留的状态必须放进 st.session_state 或磁盘 JSON。
# 本文件的两个持久化文件：
#   - chat_sessions.json：会话标题、消息、置顶状态；
#   - app_settings.json：目录、模型、TopK、领域等设置。
#
# 【UI 设计说明】
# 视觉规范参考了 ./skills 下的企业级 AI SaaS 设计技能：
#   - 浅灰蓝工作区 + 白色卡片 + 低对比边框 + 柔和阴影；
#   - 蓝色作为主操作色，紫色只做 AI 强调，避免“满屏蓝”；
#   - 4pt 间距体系、16-20px 卡片圆角、胶囊状态标签；
#   - 侧边栏固定、主工作区自适应，移动端自动收窄；
#   - 所有交互元素都有 hover / active / focus-visible 状态；
#   - 用系统字体（SF Pro / PingFang SC），不依赖 Google Fonts，
#     保证离线环境也能正常显示。
# 这些样式全部通过 st.markdown 注入 CSS 实现，不引入新的前端依赖，
# 也不改变任何业务逻辑。
# ============================================================

# ---------- 标准库导入 ----------
import html                # 转义用户/文档内容，防止 HTML 注入
import json                # 读写会话与设置 JSON
import os                  # 路径、环境变量
import re                  # 清洗 LaTeX 定界符、识别代码/标题/表格
import shutil              # 重建索引前把上传暂存目录复制到 docs/uploaded
import textwrap            # 代码块去缩进
import uuid                # 生成唯一会话 ID
from collections.abc import Iterator
from pathlib import Path
from typing import Any

# ---------- 第三方依赖 ----------
import requests
import streamlit as st
import streamlit_authenticator as stauth
import yaml

# ---------- 项目核心能力 ----------
# 这些函数全部来自 rag_core，Web / CLI / MCP 三个入口共用同一份实现。
from rag_core import (
    DEFAULT_SETTINGS,
    PROJECT_ROOT,
    SUPPORTED_EXTENSIONS,
    chroma_dir_for_model,
    clear_collection_cache,
    get_chroma_collection,
    get_domains,
    load_app_settings,
    query_collection,
    rebuild_index,
    resolve_project_path,
    save_app_settings,
    update_index,
)


# ==============================================================================
# 一、路径与常量
# ==============================================================================

PROJECT_ROOT_PATH = Path(PROJECT_ROOT)

# 会话记录：所有会话的标题、消息、置顶状态。
CHAT_FILE = PROJECT_ROOT_PATH / "chat_sessions.json"

# 用户设置：Web 设置页写、CLI/MCP 也读，是三个入口的共同配置源。
SETTINGS_FILE = PROJECT_ROOT_PATH / "app_settings.json"

# 登录配置：用户名、bcrypt 密码哈希、cookie 配置。
AUTH_CONFIG_FILE = PROJECT_ROOT_PATH / "dev_auth_config.yaml"

# .env：只放 COOKIE_KEY 这类不适合写进 YAML 的密钥。
ENV_FILE = PROJECT_ROOT_PATH / ".dev_env"

# 上传暂存目录：file_uploader 先写这里，点“重建索引/增量更新”时再复制到
# {docs_dir}/uploaded。这样建库逻辑只需要扫描 docs_dir，不用关心上传状态。
UPLOAD_STAGING_DIR = PROJECT_ROOT_PATH / "uploaded_docs"

# file_uploader 的 type 参数不需要点号。
UPLOAD_TYPES = sorted(extension.lstrip(".") for extension in SUPPORTED_EXTENSIONS)

# 固定的产品文案，避免散落在代码各处。
APP_TITLE = "本地 RAG 助手"
APP_EYEBROW = "LOCAL KNOWLEDGE ENGINE"


# ==============================================================================
# 二、主题样式
# ==============================================================================

# 说明：这是“设计系统层”，只负责视觉，不参与业务逻辑。
# 选择器优先使用 data-testid / st-key-* 这类稳定钩子，避免依赖 Streamlit
# 内部易变的 class 名。所有颜色、圆角、阴影都集中在这里，后续换肤只改这一块。
THEME_CSS = r"""
:root {
    /* ---- 颜色 token ---- */
    --rag-bg: #f3f6fb;
    --rag-surface: #ffffff;
    --rag-surface-soft: #f8fafd;
    --rag-border: #e4e9f2;
    --rag-border-strong: #d3dce8;
    --rag-text: #111827;
    --rag-text-secondary: #475467;
    --rag-muted: #667085;
    --rag-primary: #2563eb;
    --rag-primary-strong: #1d4ed8;
    --rag-primary-soft: #eaf1ff;
    --rag-accent: #7c3aed;
    --rag-success: #059669;
    --rag-warning: #b45309;
    --rag-danger: #dc2626;

    /* ---- 圆角 / 阴影 / 字体 ---- */
    --rag-radius-xl: 20px;
    --rag-radius-lg: 16px;
    --rag-radius-md: 12px;
    --rag-shadow-sm: 0 1px 2px rgba(16, 24, 40, .05);
    --rag-shadow: 0 1px 3px rgba(16, 24, 40, .06), 0 10px 30px rgba(16, 24, 40, .06);
    --rag-font: -apple-system, BlinkMacSystemFont, "SF Pro Display", "SF Pro Text",
                "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", "Segoe UI", sans-serif;
}

/* ---- 全局画布 ---- */
html, body, .stApp { color-scheme: light; }
.stApp {
    background:
        radial-gradient(1100px 520px at 18% -12%, rgba(37, 99, 235, .10), transparent 60%),
        radial-gradient(800px 420px at 92% 0%, rgba(124, 58, 237, .06), transparent 58%),
        var(--rag-bg);
    color: var(--rag-text);
    font-family: var(--rag-font);
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stToolbar"] { right: .75rem; }
/* 本地应用不需要 Streamlit Cloud 的 Deploy 按钮，隐藏后标题区更干净。 */
[data-testid="stAppDeployButton"] { display: none; }
#MainMenu, footer { visibility: hidden; }

/* 主工作区：限制阅读宽度，避免宽屏下聊天内容被拉得太散。 */
[data-testid="stMainBlockContainer"] {
    max-width: 1180px;
    padding: 2rem 2rem 7rem;
}

/* ---- 侧边栏 ---- */
[data-testid="stSidebar"] {
    background: rgba(255, 255, 255, .94);
    border-right: 1px solid var(--rag-border);
    backdrop-filter: blur(12px);
}
[data-testid="stSidebarUserContent"] { padding: 1.25rem 1rem 1.5rem; }
/* 让会话列表占满剩余高度、底部设置/退出贴底。
   选择器只命中“包含 footer 的那个垂直块”，而不是所有 SidebarUserContent
   子 div；这样即使 Streamlit 以后增加额外块，也不会把侧边栏撑成多屏高。 */
[data-testid="stSidebarUserContent"] [data-testid="stVerticalBlock"]:has(.st-key-sidebar_footer) {
    min-height: calc(100vh - 3.5rem);
    display: flex;
    flex-direction: column;
}
/* Streamlit 会把每个元素包一层 stLayoutWrapper，
   flex 布局必须作用在 wrapper 上，直接作用在 st-key-* 内部块上不会生效。 */
[data-testid="stSidebarUserContent"] [data-testid="stLayoutWrapper"]:has(.st-key-sidebar_sessions) {
    flex: 1 1 auto;
    min-height: 0;
}
.st-key-sidebar_sessions {
    height: 100%;
    overflow-y: auto;
    padding-right: 2px;
}
[data-testid="stSidebarUserContent"] [data-testid="stLayoutWrapper"]:has(.st-key-sidebar_footer) {
    margin-top: auto;
}
.st-key-sidebar_footer {
    flex: 0 0 auto;
    border-top: 1px solid var(--rag-border);
    padding-top: 1rem;
    padding-bottom: .25rem;
}

/* ---- 品牌区 ---- */
.rag-brand {
    display: flex;
    align-items: center;
    gap: .75rem;
    padding: .35rem .25rem .9rem;
}
.rag-brand-mark {
    width: 2.5rem;
    height: 2.5rem;
    border-radius: 14px;
    background: linear-gradient(145deg, #3b82f6, #1d4ed8 55%, #7c3aed);
    display: grid;
    place-items: center;
    box-shadow: 0 8px 20px rgba(37, 99, 235, .28);
    flex: 0 0 auto;
}
.rag-brand-mark svg { width: 1.35rem; height: 1.35rem; }
.rag-brand-title { font-size: 1rem; font-weight: 700; letter-spacing: -.01em; }
.rag-brand-sub {
    font-size: .65rem;
    color: var(--rag-muted);
    letter-spacing: .14em;
    font-weight: 600;
    margin-top: .15rem;
}

/* ---- 页面标题区 ---- */
.rag-page-header { padding: .25rem 0 1.35rem; }
.rag-eyebrow {
    font-size: .68rem;
    letter-spacing: .16em;
    color: var(--rag-primary);
    font-weight: 700;
    text-transform: uppercase;
    margin-bottom: .45rem;
}
.rag-title {
    font-size: clamp(1.75rem, 2.4vw, 2.35rem);
    line-height: 1.1;
    letter-spacing: -.03em;
    font-weight: 750;
    margin: 0;
}
.rag-subtitle {
    color: var(--rag-muted);
    font-size: .98rem;
    line-height: 1.6;
    margin: .55rem 0 0;
    max-width: 65ch;
}
.rag-header-row {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 1rem;
}
.rag-status-pill {
    display: inline-flex;
    align-items: center;
    gap: .45rem;
    border: 1px solid var(--rag-border);
    background: rgba(255, 255, 255, .82);
    border-radius: 999px;
    padding: .38rem .7rem;
    font-size: .75rem;
    font-weight: 600;
    color: var(--rag-text-secondary);
    white-space: nowrap;
    box-shadow: var(--rag-shadow-sm);
}
.rag-status-pill.is-warning {
    color: var(--rag-warning);
    background: #fffbeb;
    border-color: #fde68a;
}
.rag-dot {
    width: .5rem;
    height: .5rem;
    border-radius: 999px;
    background: var(--rag-success);
    box-shadow: 0 0 0 3px rgba(5, 150, 105, .12);
}
.rag-status-pill.is-warning .rag-dot {
    background: #f59e0b;
    box-shadow: 0 0 0 3px rgba(245, 158, 11, .14);
}

/* ---- 指标条：4 个等高小卡片 ---- */
.rag-stat-row {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: .65rem;
    margin-top: 1.1rem;
}
.rag-stat {
    background: rgba(255, 255, 255, .78);
    border: 1px solid var(--rag-border);
    border-radius: 14px;
    padding: .7rem .8rem;
    box-shadow: var(--rag-shadow-sm);
}
.rag-stat-label {
    display: block;
    font-size: .66rem;
    letter-spacing: .08em;
    text-transform: uppercase;
    color: var(--rag-muted);
    font-weight: 600;
}
.rag-stat-value {
    display: block;
    margin-top: .22rem;
    font-size: .95rem;
    font-weight: 700;
    color: var(--rag-text);
    font-variant-numeric: tabular-nums;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

/* ---- 通用按钮 ---- */
.stButton > button,
.stDownloadButton > button,
[data-testid="stFormSubmitButton"] button {
    border-radius: 10px;
    border: 1px solid var(--rag-border);
    background: var(--rag-surface);
    color: var(--rag-text);
    font-weight: 600;
    font-size: .88rem;
    min-height: 2.5rem;
    box-shadow: var(--rag-shadow-sm);
    transition: transform .12s ease, border-color .12s ease,
                background .18s ease, box-shadow .18s ease, color .18s ease;
}
.stButton > button:hover,
.stDownloadButton > button:hover,
[data-testid="stFormSubmitButton"] button:hover {
    border-color: #b9cdf5;
    background: #f8fbff;
    color: var(--rag-primary-strong);
    transform: translateY(-1px);
    box-shadow: 0 6px 16px rgba(16, 24, 40, .08);
}
.stButton > button:active,
.stDownloadButton > button:active,
[data-testid="stFormSubmitButton"] button:active {
    transform: scale(.98);
}
.stButton > button:focus-visible,
.stDownloadButton > button:focus-visible,
[data-testid="stFormSubmitButton"] button:focus-visible {
    outline: 3px solid rgba(37, 99, 235, .22);
    outline-offset: 2px;
}
.stButton > button[kind="primary"],
[data-testid="stFormSubmitButton"] button[kind="primary"] {
    background: linear-gradient(180deg, #3b82f6, #2563eb);
    color: #fff;
    border-color: transparent;
    box-shadow: 0 8px 18px rgba(37, 99, 235, .22);
}
.stButton > button[kind="primary"]:hover,
[data-testid="stFormSubmitButton"] button[kind="primary"]:hover {
    background: linear-gradient(180deg, #2563eb, #1d4ed8);
    color: #fff;
}
/* 登录表单的提交按钮 kind 是 secondaryFormSubmit，不是 primary；
   单独提亮成主操作，避免登录页只有一个灰按钮。 */
[data-testid="stFormSubmitButton"] button,
[data-testid="stBaseButton-secondaryFormSubmit"] {
    background: linear-gradient(180deg, #3b82f6, #2563eb);
    color: #fff;
    border-color: transparent;
    box-shadow: 0 8px 18px rgba(37, 99, 235, .22);
}
[data-testid="stFormSubmitButton"] button:hover,
[data-testid="stBaseButton-secondaryFormSubmit"]:hover {
    background: linear-gradient(180deg, #2563eb, #1d4ed8);
    color: #fff;
}

/* 侧边栏按钮：默认左对齐；当前会话用近黑色胶囊，形成强选中态。 */
[data-testid="stSidebar"] .stButton > button {
    justify-content: flex-start;
    text-align: left;
}
[data-testid="stSidebar"] .st-key-sidebar_sessions button[kind="primary"] {
    background: #0f172a;
    border-color: #0f172a;
    color: #fff;
    box-shadow: 0 6px 16px rgba(15, 23, 42, .18);
}
[data-testid="stSidebar"] .st-key-sidebar_sessions button[kind="primary"]:hover {
    background: #1e293b;
    border-color: #1e293b;
    color: #fff;
}
[data-testid="stSidebar"] .st-key-sidebar_sessions .stButton > button {
    min-height: 2.25rem;
    padding: .4rem .6rem;
    font-size: .86rem;
}
/* “开启新对话”保持蓝色主操作，不被上面的深色选中态覆盖。 */
[data-testid="stSidebar"] .st-key-new_chat button[kind="primary"] {
    background: linear-gradient(180deg, #3b82f6, #2563eb);
    border-color: transparent;
    color: #fff;
    box-shadow: 0 8px 18px rgba(37, 99, 235, .25);
}
[data-testid="stSidebar"] .st-key-new_chat button[kind="primary"]:hover {
    background: linear-gradient(180deg, #2563eb, #1d4ed8);
    color: #fff;
}

/* ---- 设置页卡片 ---- */
.st-key-settings_card_generate,
.st-key-settings_card_index,
.st-key-settings_card_upload {
    background: var(--rag-surface);
    border: 1px solid var(--rag-border);
    border-radius: var(--rag-radius-lg);
    padding: 1.15rem 1.25rem;
    box-shadow: var(--rag-shadow);
}
.rag-card-title {
    display: flex;
    align-items: center;
    gap: .5rem;
    font-size: 1rem;
    font-weight: 700;
    letter-spacing: -.01em;
    margin-bottom: .15rem;
}
.rag-card-desc {
    color: var(--rag-muted);
    font-size: .84rem;
    line-height: 1.55;
    margin-bottom: .9rem;
}

/* ---- 索引状态卡 ---- */
.rag-index-card {
    background: linear-gradient(180deg, #f8fbff, #ffffff);
    border: 1px solid var(--rag-border);
    border-radius: 14px;
    padding: .9rem 1rem;
    margin: .2rem 0 .9rem;
}
.rag-index-main {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: 1rem;
}
.rag-index-label {
    font-size: .72rem;
    letter-spacing: .08em;
    text-transform: uppercase;
    color: var(--rag-muted);
    font-weight: 700;
}
.rag-index-value {
    font-size: 1.55rem;
    font-weight: 750;
    letter-spacing: -.02em;
    font-variant-numeric: tabular-nums;
    color: var(--rag-primary-strong);
}
.rag-index-meta {
    margin-top: .35rem;
    color: var(--rag-muted);
    font-size: .76rem;
    line-height: 1.6;
    word-break: break-all;
}

/* ---- 空状态 ---- */
.rag-empty {
    margin: 1.25rem 0 1.5rem;
    padding: 2.5rem 1.5rem;
    text-align: center;
    border: 1px solid var(--rag-border);
    border-radius: 24px;
    background: linear-gradient(180deg, rgba(255, 255, 255, .94), rgba(248, 250, 253, .94));
    box-shadow: var(--rag-shadow);
}
.rag-empty-art {
    width: 92px;
    height: 92px;
    margin: 0 auto 1rem;
    display: grid;
    place-items: center;
    border-radius: 24px;
    background: var(--rag-primary-soft);
    border: 1px solid #d6e4ff;
}
.rag-empty h2 {
    font-size: 1.35rem;
    letter-spacing: -.02em;
    margin: 0;
}
.rag-empty p {
    color: var(--rag-muted);
    max-width: 52ch;
    margin: .45rem auto 0;
    line-height: 1.7;
    font-size: .92rem;
}
.rag-empty-points {
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    gap: .5rem;
    margin-top: 1.1rem;
}
.rag-empty-points span {
    font-size: .76rem;
    font-weight: 600;
    color: var(--rag-text-secondary);
    background: #f1f5fb;
    border: 1px solid var(--rag-border);
    border-radius: 999px;
    padding: .32rem .65rem;
}

/* ---- 聊天消息 ---- */
[data-testid="stChatMessage"] {
    background: transparent;
    padding: 0;
    gap: .75rem;
    margin-bottom: 1rem;
    animation: ragFadeUp .26s ease both;
}
[data-testid="stChatMessageContent"] {
    background: var(--rag-surface);
    border: 1px solid var(--rag-border);
    border-radius: var(--rag-radius-lg);
    padding: 1rem 1.15rem;
    box-shadow: var(--rag-shadow-sm);
    line-height: 1.75;
    font-size: .95rem;
}
/* 用户消息用淡蓝底，和助手的白底形成角色区分。
   :has() 在现代浏览器可用；不支持时退化为统一白底，不影响功能。 */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"])
    [data-testid="stChatMessageContent"] {
    background: #f4f8ff;
    border-color: #d8e6ff;
}

/* ---- 聊天输入框 ---- */
[data-testid="stChatInput"] {
    /* 底部输入区默认会撑满sidebar右侧的整个主区域；在超宽屏上会比
       对话卡片宽很多。限制到主内容宽度并居中，视觉上才和对话对齐。 */
    width: 100%;
    height: 150px;
    max-width: 1116px;
    margin: 0 auto;
    background: var(--rag-surface);
    border: 1px solid var(--rag-border);
    border-radius: var(--rag-radius-lg);
    box-shadow: var(--rag-shadow);
    transition: border-color .18s ease, box-shadow .18s ease;
}
[data-testid="stChatInput"]:focus-within {
    border-color: var(--rag-primary);
    box-shadow: 0 0 0 4px rgba(37, 99, 235, .12), var(--rag-shadow);
}
/* 内部还有一层 Streamlit 自带的浅灰背景，清掉后才是真正的白底。
   Streamlit 1.66 的 textarea 有独立的 data-testid，直接命中更稳。 */
[data-testid="stChatInput"] > div { background: transparent; }
[data-testid="stChatInputTextArea"] {
    background: transparent;
    color: var(--rag-text);
    font-family: var(--rag-font);
    font-size: .95rem;
    line-height: 1.6;
}
[data-testid="stChatInputSubmitButton"] {
    border-radius: 999px;
    background: var(--rag-primary-soft);
    color: var(--rag-primary);
}
[data-testid="stChatInputSubmitButton"]:not(:disabled) {
    background: var(--rag-primary);
    color: #fff;
}
[data-testid="stChatInputSubmitButton"]:disabled {
    background: var(--rag-surface-soft);
    color: var(--rag-border-strong);
}

/* ---- Streamlit 原生组件微调 ---- */
/* 显式设置文字色：即使系统/浏览器偏好是深色，白底卡片上的文字也不能变成浅色。 */
:where([data-testid="stMarkdownContainer"]),
:where([data-testid="stStatusWidget"]),
:where([data-testid="stExpander"] summary) {
    color: var(--rag-text);
}
:where([data-testid="stWidgetLabel"] p) { color: var(--rag-text-secondary); }
[data-baseweb="input"] input,
[data-baseweb="textarea"] textarea,
[data-baseweb="select"] > div,
[data-testid="stChatInput"] textarea {
    color: var(--rag-text);
}
[data-baseweb="popover"],
[data-baseweb="menu"] {
    background: var(--rag-surface);
}
[data-testid="stStatusWidget"] {
    border: 1px solid var(--rag-border);
    border-radius: 14px;
    background: var(--rag-surface);
    box-shadow: var(--rag-shadow-sm);
}
[data-testid="stExpander"] {
    border: 1px solid var(--rag-border);
    border-radius: 14px;
    background: var(--rag-surface);
    overflow: hidden;
}
[data-testid="stExpander"] summary {
    padding: .75rem 1rem;
    font-weight: 600;
}
[data-testid="stExpander"] summary:hover { background: #f8fafc; }
[data-testid="stAlert"] {
    border-radius: 12px;
    border: 1px solid var(--rag-border);
}
[data-testid="stFileUploaderDropzone"] {
    border: 1.5px dashed var(--rag-border-strong);
    border-radius: 16px;
    background: var(--rag-surface-soft);
    padding: 1.2rem;
    transition: border-color .18s ease, background .18s ease;
}
[data-testid="stFileUploaderDropzone"]:hover {
    border-color: var(--rag-primary);
    background: #f5f9ff;
}

/* 输入框 / 下拉框统一圆角，避免浏览器默认样式破坏整体感。
   Streamlit 1.66 的 text input 根节点是 stTextInputRootElement，
   不是 BaseWeb 的 [data-baseweb="input"]；用错选择器会出现“白底无边框”。 */
[data-testid="stTextInputRootElement"],
[data-testid="stNumberInputRootElement"],
[data-testid="stTextAreaRootElement"] {
    border: 1px solid var(--rag-border);
    border-radius: 10px;
    background: var(--rag-surface-soft);
    transition: border-color .16s ease, background .16s ease, box-shadow .16s ease;
}
[data-testid="stTextInputRootElement"]:focus-within,
[data-testid="stNumberInputRootElement"]:focus-within,
[data-testid="stTextAreaRootElement"]:focus-within {
    border-color: var(--rag-primary);
    background: var(--rag-surface);
    box-shadow: 0 0 0 4px rgba(37, 99, 235, .10);
}
[data-testid="stTextInputField"],
[data-testid="stNumberInputField"],
[data-testid="stTextAreaField"] {
    color: var(--rag-text);
    background: transparent;
}
[data-testid="stSelectbox"] [data-baseweb="select"] > div {
    border: 1px solid var(--rag-border);
    border-radius: 10px;
    background: var(--rag-surface-soft);
}
[data-testid="stSelectbox"] [data-baseweb="select"] > div:focus-within {
    border-color: var(--rag-primary);
    box-shadow: 0 0 0 4px rgba(37, 99, 235, .10);
}
[data-testid="stSlider"] [role="slider"] {
    background: var(--rag-primary);
    border-color: var(--rag-primary);
}

/* Tabs 做成胶囊导航：选中态近黑，未选中浅灰。
   注意：Streamlit 1.66 的 Tab 是 data-testid="stTab"（role=tab），
   不是 BaseWeb 的 [data-baseweb="tab"]；旧选择器在这里不生效。 */
[data-testid="stTabs"] [role="tablist"] {
    gap: .5rem;
    border-bottom: 0;
    flex-wrap: wrap;
}
[data-testid="stTab"] {
    border-radius: 999px;
    padding: .45rem .9rem;
    background: #eef2f8;
    color: var(--rag-text-secondary);
    font-weight: 600;
    cursor: pointer;
    transition: background .16s ease, color .16s ease;
}
[data-testid="stTab"]:hover { background: #e3ebf7; }
[data-testid="stTab"][aria-selected="true"] {
    background: #0f172a;
    color: #fff;
}
[data-testid="stTab"] [data-testid="stMarkdownContainer"] p {
    margin: 0;
    color: inherit;
}
/* 默认的红色下划线指示器与胶囊风格冲突，隐藏。 */
[data-testid="stTab"] .react-aria-SelectionIndicator { display: none; }

/* ---- 引用片段 ---- */
.rag-ref-meta {
    display: flex;
    flex-wrap: wrap;
    gap: .4rem;
    margin: .35rem 0 .55rem;
}
.rag-ref-chip {
    font-size: .7rem;
    font-weight: 600;
    color: var(--rag-text-secondary);
    background: #f3f6fb;
    border: 1px solid var(--rag-border);
    border-radius: 999px;
    padding: .2rem .5rem;
}
.rag-ref-index {
    color: var(--rag-primary);
    background: var(--rag-primary-soft);
    border-color: #d6e4ff;
}

/* ---- 用户信息 chip ---- */
.rag-user-chip {
    display: flex;
    align-items: center;
    gap: .6rem;
    padding: .55rem .6rem;
    border: 1px solid var(--rag-border);
    border-radius: 12px;
    background: var(--rag-surface-soft);
    margin-bottom: .5rem;
}
.rag-user-avatar {
    width: 1.9rem;
    height: 1.9rem;
    border-radius: 999px;
    background: var(--rag-primary-soft);
    color: var(--rag-primary);
    display: grid;
    place-items: center;
    font-weight: 700;
    font-size: .8rem;
    flex: 0 0 auto;
}
.rag-user-name { font-size: .82rem; font-weight: 600; }
.rag-user-role { font-size: .68rem; color: var(--rag-muted); }

/* ---- 登录页 ---- */
.rag-login-hero {
    max-width: 520px;
    margin: 1.5rem auto 1rem;
    text-align: center;
}
.rag-login-hero .rag-brand-mark {
    margin: 0 auto .8rem;
    width: 3rem;
    height: 3rem;
    border-radius: 16px;
}
.rag-login-hero h1 {
    font-size: 2rem;
    letter-spacing: -.03em;
    margin: 0;
}
.rag-login-hero p {
    color: var(--rag-muted);
    font-size: .92rem;
    margin: .5rem auto 0;
    max-width: 44ch;
    line-height: 1.65;
}
[data-testid="stForm"] {
    max-width: 420px;
    margin: 0 auto;
    background: var(--rag-surface);
    border: 1px solid var(--rag-border);
    border-radius: 20px;
    padding: 1.5rem 1.4rem;
    box-shadow: var(--rag-shadow);
}
/* 登录页的提示条和表单同宽，避免一条横贯全屏的黄色警告破坏构图。 */
.st-key-login_message {
    max-width: 420px;
    margin: 0 auto;
}

/* ---- 页脚 ---- */
.rag-footer {
    text-align: center;
    color: #98a2b3;
    font-size: .72rem;
    letter-spacing: .03em;
    padding: 2rem 0 .5rem;
}

/* ---- 宽屏右侧固定信息卡 ----
   主内容最大宽度 1180px。视口足够宽时，把聊天页的标题/状态/指标卡
   固定在右侧空白区，滚动对话时始终可见。
   为什么分两档：
   - 侧边栏收起时右侧空白更大，1900px 起就能放 280px 卡片；
   - 侧边栏展开时主内容整体右移，需要 2000px 起才放得下 230px 卡片。
   低于阈值时保持原布局（标题在内容顶部），不会遮挡对话。 */
@media (min-width: 1900px) {
    .stApp:has([data-testid="stSidebar"][data-collapsed="true"]) .st-key-chat_context_rail {
        position: fixed;
        top: 1.5rem;
        right: 12px;
        width: 280px;
        max-height: calc(100vh - 3rem);
        overflow-y: auto;
        z-index: 30;
        padding: 1rem 1rem .95rem;
        border: 1px solid var(--rag-border);
        border-radius: var(--rag-radius-xl);
        background: rgba(255, 255, 255, .88);
        box-shadow: var(--rag-shadow);
        backdrop-filter: blur(14px);
    }
    .stApp:has([data-testid="stSidebar"][data-collapsed="true"]) .st-key-chat_context_rail .rag-page-header { padding: 0; }
    .stApp:has([data-testid="stSidebar"][data-collapsed="true"]) .st-key-chat_context_rail .rag-header-row {
        flex-direction: column;
        gap: .55rem;
    }
    .stApp:has([data-testid="stSidebar"][data-collapsed="true"]) .st-key-chat_context_rail .rag-eyebrow {
        font-size: .58rem;
        margin-bottom: .3rem;
    }
    .stApp:has([data-testid="stSidebar"][data-collapsed="true"]) .st-key-chat_context_rail .rag-title {
        font-size: 1.35rem;
        letter-spacing: -.025em;
    }
    .stApp:has([data-testid="stSidebar"][data-collapsed="true"]) .st-key-chat_context_rail .rag-subtitle {
        font-size: .78rem;
        line-height: 1.55;
        margin-top: .35rem;
        max-width: none;
    }
    .stApp:has([data-testid="stSidebar"][data-collapsed="true"]) .st-key-chat_context_rail .rag-status-pill {
        align-self: flex-start;
        padding: .28rem .55rem;
        font-size: .68rem;
    }
    .stApp:has([data-testid="stSidebar"][data-collapsed="true"]) .st-key-chat_context_rail .rag-stat-row {
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: .45rem;
        margin-top: .8rem;
    }
    .stApp:has([data-testid="stSidebar"][data-collapsed="true"]) .st-key-chat_context_rail .rag-stat {
        padding: .5rem .55rem;
        border-radius: 10px;
    }
    .stApp:has([data-testid="stSidebar"][data-collapsed="true"]) .st-key-chat_context_rail .rag-stat-label { font-size: .58rem; }
    .stApp:has([data-testid="stSidebar"][data-collapsed="true"]) .st-key-chat_context_rail .rag-stat-value { font-size: .75rem; }
}

@media (min-width: 2000px) {
    .stApp:has([data-testid="stSidebar"][data-collapsed="false"]) .st-key-chat_context_rail {
        position: fixed;
        top: 1.5rem;
        right: 12px;
        width: 230px;
        max-height: calc(100vh - 3rem);
        overflow-y: auto;
        z-index: 30;
        padding: .9rem .9rem .85rem;
        border: 1px solid var(--rag-border);
        border-radius: var(--rag-radius-xl);
        background: rgba(255, 255, 255, .9);
        box-shadow: var(--rag-shadow);
        backdrop-filter: blur(14px);
    }
    .stApp:has([data-testid="stSidebar"][data-collapsed="false"]) .st-key-chat_context_rail .rag-page-header { padding: 0; }
    .stApp:has([data-testid="stSidebar"][data-collapsed="false"]) .st-key-chat_context_rail .rag-header-row {
        flex-direction: column;
        gap: .5rem;
    }
    .stApp:has([data-testid="stSidebar"][data-collapsed="false"]) .st-key-chat_context_rail .rag-eyebrow {
        font-size: .55rem;
        margin-bottom: .25rem;
    }
    .stApp:has([data-testid="stSidebar"][data-collapsed="false"]) .st-key-chat_context_rail .rag-title {
        font-size: 1.18rem;
        letter-spacing: -.02em;
    }
    .stApp:has([data-testid="stSidebar"][data-collapsed="false"]) .st-key-chat_context_rail .rag-subtitle {
        font-size: .74rem;
        line-height: 1.5;
        margin-top: .3rem;
        max-width: none;
    }
    .stApp:has([data-testid="stSidebar"][data-collapsed="false"]) .st-key-chat_context_rail .rag-status-pill {
        align-self: flex-start;
        padding: .24rem .5rem;
        font-size: .64rem;
    }
    .stApp:has([data-testid="stSidebar"][data-collapsed="false"]) .st-key-chat_context_rail .rag-stat-row {
        grid-template-columns: 1fr;
        gap: .35rem;
        margin-top: .7rem;
    }
    .stApp:has([data-testid="stSidebar"][data-collapsed="false"]) .st-key-chat_context_rail .rag-stat {
        padding: .42rem .5rem;
        border-radius: 9px;
    }
    .stApp:has([data-testid="stSidebar"][data-collapsed="false"]) .st-key-chat_context_rail .rag-stat-label { font-size: .55rem; }
    .stApp:has([data-testid="stSidebar"][data-collapsed="false"]) .st-key-chat_context_rail .rag-stat-value { font-size: .72rem; }
}

/* ---- 动效：只做一次轻微入场，尊重 reduced-motion ---- */
@keyframes ragFadeUp {
    from { opacity: 0; transform: translateY(8px); }
    to { opacity: 1; transform: none; }
}
.rag-page-header, .rag-empty, [data-testid="stChatMessage"] {
    animation: ragFadeUp .26s ease both;
}
@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
        animation-duration: .001ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: .001ms !important;
    }
}

/* ---- 响应式 ---- */
@media (max-width: 900px) {
    [data-testid="stMainBlockContainer"] { padding: 1.25rem 1rem 6rem; }
    .rag-title { font-size: 1.7rem; }
    .rag-stat-row { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .rag-header-row { flex-direction: column; }
}
"""

# 复用的品牌图形：用内联 SVG 而不是 emoji，离线可用且不会显得像临时占位。
LOGO_SVG = """
<svg viewBox="0 0 32 32" fill="none" aria-hidden="true">
  <path d="M8 9.5C8 8.1 9.1 7 10.5 7H21.5C22.9 7 24 8.1 24 9.5V22.5C24 23.9 22.9 25 21.5 25H10.5C9.1 25 8 23.9 8 22.5V9.5Z"
        stroke="white" stroke-width="1.8"/>
  <path d="M12 12H20M12 16H20M12 20H17" stroke="white" stroke-width="1.8" stroke-linecap="round"/>
</svg>
"""

EMPTY_STATE_SVG = """
<svg width="52" height="52" viewBox="0 0 96 96" fill="none" aria-hidden="true">
  <rect x="14" y="22" width="50" height="62" rx="12" fill="#EAF1FF" stroke="#BBD3FF"/>
  <rect x="26" y="12" width="50" height="62" rx="12" fill="#FFFFFF" stroke="#2563EB" stroke-width="2"/>
  <path d="M38 32H64M38 42H64M38 52H54" stroke="#98B7F0" stroke-width="4" stroke-linecap="round"/>
  <circle cx="68" cy="64" r="12" fill="#F5F3FF" stroke="#7C3AED" stroke-width="3"/>
  <path d="M77 73L85 81" stroke="#7C3AED" stroke-width="4" stroke-linecap="round"/>
</svg>
"""


# ==============================================================================
# 三、通用工具函数
# ==============================================================================

def load_dotenv_file(path: Path) -> None:
    """把 .env 里的 KEY=VALUE 读进 os.environ（不覆盖已有环境变量）。

    为什么不引入 python-dotenv：
    - 我们只需要最基础的 KEY=VALUE，不值得多一个依赖；
    - 离线环境安装依赖越少越稳；
    - 使用 setdefault 可以让“部署平台注入的环境变量”优先级高于 .env，
      这符合十二要素应用的习惯。

    当前只用于 COOKIE_KEY：auth_config.yaml 里的 cookie.key 是占位符，
    真正的签名密钥放在 .env，避免把密钥提交到版本库。
    """
    if not path.is_file():
        return
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return

    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


def load_json(path: str, default: Any) -> Any:
    """读取 JSON；文件不存在或损坏时返回 default。

    Streamlit 应用最怕“启动即崩”：用户手动改坏 chat_sessions.json，
    不应该导致整个页面白屏。回退到默认值后，用户至少还能进入应用。
    """
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as file:
                return json.load(file)
        except (OSError, json.JSONDecodeError):
            return default
    return default


def save_json(path: str, data: Any) -> None:
    """原子写入 JSON，避免进程中断留下半截文件。

    先写 .tmp 再 os.replace：这一步在 POSIX 上是原子的，
    可以保证 chat_sessions.json 要么是旧内容、要么是完整新内容。
    """
    temp_path = f"{path}.tmp"
    with open(temp_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
    os.replace(temp_path, path)


def clean_latex_format(text: str) -> str:
    r"""把 LLM 常见的 LaTeX 定界符统一成 Streamlit 能渲染的 $ / $$。

    背景：LLM 经常输出 \[...\] 和 \(...\)，但 Streamlit 的 markdown
    只认 $...$ / $$...$$。入口（历史消息渲染）和出口（流式输出、入库）
    都统一清洗，避免同一条公式在不同位置显示不一致。
    """
    if not text:
        return text
    # 用“成对匹配”而不是全局 replace：
    # 代码片段里可能出现未配对的 \[ 或 \(，全局替换会把正常文本改坏。
    text = re.sub(r"\\\[(.*?)\\\]", r"$$\1$$", text, flags=re.DOTALL)
    text = re.sub(r"\\\((.*?)\\\)", r"$\1$", text, flags=re.DOTALL)
    return text


# 代码行特征：只匹配“看起来像代码”的行，避免把普通英文句子误判成代码。
# 例如 if/for/while 必须带冒号，print/import/def 必须出现在行首，
# SQL 关键字要求大写，这样误判率最低。
_CODE_LINE_RE = re.compile(
    r"""^(?:
        async\s+def\s|def\s|class\s|import\s|from\s+.+\s+import\s|
        for\s+.+\s+in\s+.+:|while\s+.+:|if\s+.+:|elif\s+.+:|else\s*:|
        try\s*:|except\b|finally\s*:|with\s+.+:
        |return\b|yield\b|print\s*\(|lambda\b|
        \#include\b|SELECT\b|INSERT\b|UPDATE\b|DELETE\b|CREATE\s+TABLE\b|
        function\s|const\s|let\s|var\s|console\.log\s*\(|System\.out|
        public\s+static\b|private\s+|protected\s+|>>>\s|
        //|/\*|\*/
    )""",
    re.VERBOSE,
)

# 标题特征：Markdown 标题、PDF 常见的“第N章”、以及 “3.1 / 14.1.3” 编号。
_HEADING_RE = re.compile(
    r"^(?:#{1,6}\s+\S|第\s*\d+\s*章|\d+(?:\.\d+){1,3}[\s、.]\s*\S)"
)

# 行内 LaTeX 命令：只覆盖常见数学命令，避免把普通反斜杠文本误包成公式。
_INLINE_LATEX_RE = re.compile(
    r"(?<!\$)(\\(?:frac|sqrt|sum|int|prod|lim|alpha|beta|gamma|delta|theta|"
    r"lambda|mu|sigma|omega|pi|times|cdot|leq|geq|neq|approx|infty|partial|"
    r"nabla|text|mathbf|mathrm|left|right|pm|mp|in|notin|subset|subseteq|"
    r"cup|cap|forall|exists|rightarrow|leftarrow|Rightarrow)"
    r"(?:\{[^{}]*\})*)(?!\$)"
)

# 表格行特征：至少两个竖线。Excel 解析出来的行就是 "a | b | c" 这种形式。
_TABLE_ROW_MIN_PIPES = 2


def _normalize_math_environments(text: str) -> str:
    """把常见 LaTeX 环境转换成 Streamlit/KaTeX 能识别的 $$ 定界形式。

    背景：很多 PDF/论文片段里是 \\begin{equation}...\\end{equation}，
    而 Streamlit 只负责把 $$...$$ 交给 KaTeX。这里做一层“环境适配”：
    - equation/displaymath -> $$ ... $$
    - align/eqnarray -> $$ + aligned（KaTeX 对 aligned 支持更好）
    - gather/multline -> $$ + gathered
    - cases/matrix 等独立环境如果不在 $$ 内，自动补上 $$ 包裹

    注意：这是启发式转换，不追求 100% 的 LaTeX 解析器能力；
    目标是让知识库里常见的公式“能显示”，而不是替代专业排版引擎。
    """
    text = re.sub(r"\\begin\{(?:equation\*?|displaymath)\}", "$$", text)
    text = re.sub(r"\\end\{(?:equation\*?|displaymath)\}", "$$", text)

    text = re.sub(r"\\begin\{(?:align\*?|eqnarray\*?)\}", r"$$\n\\begin{aligned}", text)
    text = re.sub(r"\\end\{(?:align\*?|eqnarray\*?)\}", r"\\end{aligned}\n$$", text)

    text = re.sub(r"\\begin\{(?:gather\*?|multline\*?)\}", r"$$\n\\begin{gathered}", text)
    text = re.sub(r"\\end\{(?:gather\*?|multline\*?)\}", r"\\end{gathered}\n$$", text)

    # cases / matrix 等环境本身不带 $$，如果原文没有包起来，这里补上。
    # 负向环视避免把已经在 $$ 内的内容重复包裹。
    text = re.sub(
        r"(?<!\$)(\\begin\{(?:cases|matrix|pmatrix|bmatrix|vmatrix|array)\}.*?"
        r"\\end\{(?:cases|matrix|pmatrix|bmatrix|vmatrix|array)\})(?!\$)",
        r"$$\n\1\n$$",
        text,
        flags=re.DOTALL,
    )
    return text


def _looks_like_math_line(line: str) -> bool:
    """判断一行是否“几乎就是一条公式”，用于补 $$ 定界符。

    规则刻意保守：
    - 已经带 $ 的行不动，避免破坏用户/模型写好的公式；
    - 以 \\begin/\\end 开头的环境行交给 _normalize_math_environments；
    - 含中文的行只有出现 3 个以上 LaTeX 命令才当公式，避免把
      “使用 \\frac{a}{b} 计算”这种正常句子整行包进 $$；
    - 纯英文/符号行只要有 \\ 开头、等号、或 2 个以上命令就认为是公式。
    """
    stripped = line.strip()
    if not stripped or "$" in stripped:
        return False
    if stripped.startswith(("\\begin{", "\\end{")):
        return False

    commands = re.findall(r"\\[a-zA-Z]+", stripped)
    if not commands:
        return False

    has_cjk = bool(re.search(r"[\u4e00-\u9fff]", stripped))
    if has_cjk:
        return len(commands) >= 3 and len(stripped) <= 120
    return stripped.startswith("\\") or "=" in stripped or len(commands) >= 2


def _looks_like_code_line(line: str) -> bool:
    """判断一行是否像代码，用于自动包成 Markdown 代码块。

    为什么需要它：
    知识库里很多 Python/深度学习教材片段是“散文 + 代码”混排。
    如果直接交给 Markdown，代码会变成普通段落，缩进和等号都会乱。
    这里用保守规则识别常见代码形态，识别不出来的仍按普通文本渲染，
    不会影响可读性。
    """
    stripped = line.strip()
    if not stripped:
        return False
    # LaTeX 环境行以 } 结尾且含 {，很容易被下面的“大括号规则”误判成代码；
    # 必须在最前面排除，否则公式块会被拆成代码块。
    if stripped.startswith(("\\begin{", "\\end{")):
        return False
    if _CODE_LINE_RE.match(stripped):
        return True
    if stripped in ("{", "}"):
        return True
    if stripped.endswith(("{", "}")) and len(stripped) < 100 and any(
        token in stripped for token in ("(", "=", "{")
    ):
        return True
    # 赋值语句且右侧出现括号/列表/字典，通常是代码而不是数学等式。
    if re.match(r"^[A-Za-z_][\w\.]*\s*=\s*.+", stripped) and any(
        token in stripped for token in ("(", "[", "{")
    ):
        return True
    # 单独一行的函数调用，例如 np.array([1, 2]) / print(x)
    if re.match(r"^[A-Za-z_][\w\.]*\([^)]*\)\s*$", stripped):
        return True
    return False


def _guess_code_language(code: str) -> str:
    """根据代码内容猜一个 Markdown 代码块语言，用于语法高亮。

    猜错语言只会影响高亮颜色，不影响代码内容；所以这里用宽松关键词。
    """
    if re.search(r"\b(?:def|class|import|from|print|lambda|async)\b", code):
        return "python"
    if re.search(r"\b(?:SELECT|INSERT|UPDATE|DELETE|CREATE|FROM|WHERE)\b", code):
        return "sql"
    if any(token in code for token in ("function ", "const ", "let ", "var ", "console.log", "=>")):
        return "javascript"
    if "#include" in code or "int main" in code:
        return "c"
    return "text"


def _format_table(rows: list[str]) -> str:
    """把 "a | b | c" 形式的连续行转成 Markdown 表格。

    为什么需要：
    Excel 解析结果和部分 PDF 表格本身就是用 " | " 连接单元格的；
    直接当普通文本会变成一长串竖线，转成表格后列关系才看得清。
    """
    parsed: list[list[str]] = []
    for row in rows:
        cells = [cell.strip() for cell in row.strip().strip("|").split("|")]
        parsed.append(cells)

    width = max((len(row) for row in parsed), default=0)
    if len(parsed) < 2 or width < 2:
        return "\n".join(rows)

    def pad(row: list[str]) -> list[str]:
        return row + [""] * (width - len(row))

    has_separator = any(
        len(row) >= width
        and all(re.fullmatch(r":?-{3,}:?", cell) for cell in row if cell)
        for row in parsed[1:]
    )

    lines = ["| " + " | ".join(pad(parsed[0])) + " |"]
    if not has_separator:
        lines.append("| " + " | ".join(["---"] * width) + " |")
    for row in parsed[1:]:
        lines.append("| " + " | ".join(pad(row)) + " |")
    return "\n".join(lines)


def _normalize_bullet(line: str) -> str:
    """把 PDF 常见的圆点项目符号统一成 Markdown 的 "- "。"""
    return re.sub(r"^(\s*)[•·▪‣◦]\s+", r"\1- ", line)


def _wrap_inline_latex(line: str) -> str:
    """把中文句子里的常见 LaTeX 命令包成 $...$。

    例如：“其中 x = \\frac{a}{b}。” -> “其中 x = $\\frac{a}{b}$。”
    只处理常见数学命令，并且整行已经包含 $ 时跳过，避免二次包裹。
    """
    if "\\" not in line or "$" in line:
        return line
    return _INLINE_LATEX_RE.sub(lambda match: f"${match.group(1)}$", line)


def format_reference_text(raw: str) -> str:
    """把检索片段从“原始纯文本”整理成适合 Markdown/KaTeX 渲染的文本。

    处理顺序（顺序很重要）：
    1) 先统一 LaTeX 定界符和环境；
    2) 保留原文已有的 ``` 代码围栏，围栏内部完全不动；
    3) 连续表格行 -> Markdown 表格；
    4) 缩进块 / 代码特征行 -> 自动包成代码块；
    5) 公式行 -> $$...$$；
    6) 章节标题 -> ###；
    7) 圆点列表 -> Markdown 列表。

    为什么不用一次性正则替换：
    这些规则之间会互相干扰（例如代码里的 "|" 不应变成表格、代码里的
    "\\" 不应被当成 LaTeX），必须按行扫描并维护“当前处于代码块/表格块”
    的状态，才能避免把内容改坏。
    """
    if not raw:
        return ""

    text = _normalize_math_environments(clean_latex_format(str(raw)))
    lines = text.splitlines()

    blocks: list[str] = []
    code_buffer: list[str] = []
    table_buffer: list[str] = []
    in_fence = False
    in_math = False

    def flush_code() -> None:
        if not code_buffer:
            return
        code = textwrap.dedent("\n".join(code_buffer)).strip("\n")
        code_buffer.clear()
        if not code:
            return
        fence = "```" if "```" not in code else "````"
        language = _guess_code_language(code)
        if blocks and blocks[-1] != "":
            blocks.append("")
        blocks.append(f"{fence}{language}\n{code}\n{fence}")
        blocks.append("")

    def flush_table() -> None:
        if not table_buffer:
            return
        table = _format_table(table_buffer)
        table_buffer.clear()
        if blocks and blocks[-1] != "":
            blocks.append("")
        blocks.append(table)
        blocks.append("")

    for line in lines:
        stripped = line.strip()

        # 已经写好的代码围栏：整段原样保留，避免二次格式化。
        if stripped.startswith(("```", "~~~")):
            flush_code()
            flush_table()
            if not in_fence and blocks and blocks[-1] != "":
                blocks.append("")
            in_fence = not in_fence
            blocks.append(line)
            if not in_fence:
                blocks.append("")
            continue
        if in_fence:
            blocks.append(line)
            continue

        # 已经进入 $$ 数学块：内部可能包含 \begin{aligned}、&、\\ 等
        # 只有 KaTeX 认识的语法，不能再做代码/标题/表格判断。
        if in_math:
            blocks.append(line)
            if "$$" in stripped:
                in_math = False
            continue

        # $$ 开始/结束行：单行 $$...$$ 不改变状态；单独的 $$ 才切换。
        if stripped.startswith("$$") or stripped.endswith("$$"):
            flush_code()
            flush_table()
            blocks.append(line)
            if stripped.count("$$") % 2 == 1:
                in_math = True
            continue

        if not stripped:
            flush_code()
            flush_table()
            blocks.append("")
            continue

        # 缩进块优先当代码；很多 PDF 代码示例靠缩进表达层级。
        if line.startswith("    ") or line.startswith("\t"):
            flush_table()
            # 保留原始缩进：函数体缩进是语义的一部分，
            # 统一去缩进会得到 return a + b 这种错误代码。
            code_buffer.append(line)
            continue

        # 代码块里的顶层注释（# ...）继续留在代码块内，
        # 不要被 _HEADING_RE 当成 Markdown 标题。
        if code_buffer and stripped.startswith("#"):
            code_buffer.append(line)
            continue

        # 代码特征行优先于表格判断，避免代码里的 "|" 被误认成表格。
        if _looks_like_code_line(line):
            flush_table()
            code_buffer.append(line)
            continue

        flush_code()

        if stripped.count("|") >= _TABLE_ROW_MIN_PIPES:
            table_buffer.append(stripped)
            continue
        flush_table()

        if _looks_like_math_line(stripped):
            if blocks and blocks[-1] != "":
                blocks.append("")
            blocks.append("$$")
            blocks.append(stripped)
            blocks.append("$$")
            blocks.append("")
            continue

        if _HEADING_RE.match(stripped):
            # 已经是 Markdown 标题就保留级别；PDF 编号标题统一补 ###。
            heading = stripped if stripped.startswith("#") else f"### {stripped}"
            if blocks and blocks[-1] != "":
                blocks.append("")
            blocks.append(heading)
            blocks.append("")
            continue

        # 普通正文：如果上一行是列表项，先补一个空行，
        # 否则 Markdown 可能把后续段落吞进列表里。
        normalized = _normalize_bullet(_wrap_inline_latex(line))
        if (
            blocks
            and blocks[-1].startswith("- ")
            and not normalized.lstrip().startswith(("- ", "* ", "+ "))
        ):
            blocks.append("")
        blocks.append(normalized)

    flush_code()
    flush_table()

    # 压缩多余空行，避免 PDF 里的连续空行把页面拉得很长。
    result = re.sub(r"\n{3,}", "\n\n", "\n".join(blocks)).strip()
    return result


def as_int(value: Any, default: int, minimum: int | None = None, maximum: int | None = None) -> int:
    """把配置里的值安全转换为整数，并夹紧到合法区间。

    配置文件是用户可手改的，top_k 可能是 "10"、null 或乱码；
    如果不做兜底，页面会在 int() 处直接抛异常。统一走这个函数，
    所有入口的容错行为保持一致。
    """
    try:
        result = int(value)
    except (TypeError, ValueError):
        result = default
    if minimum is not None:
        result = max(minimum, result)
    if maximum is not None:
        result = min(maximum, result)
    return result


def display_path(path: str) -> str:
    """把绝对路径显示成相对项目根目录的形式，方便用户在设置页阅读。

    例如 /Users/x/rag-project/chromadb/bge -> ./chromadb/bge。
    如果路径在项目外，保留绝对路径，避免出现一堆 ../../。
    """
    relative = os.path.relpath(path, PROJECT_ROOT)
    if relative.startswith(".."):
        return path
    return f"./{relative}"


def discover_embedding_models() -> list[str]:
    """扫描 ./models 下的本地模型目录，生成设置页下拉选项。

    为什么不继续硬编码三个模型：
    - 用户可能自己下载了新的 embedding 模型放进 models/；
    - 硬编码会让新模型“明明在磁盘上却选不到”；
    - 动态扫描只依赖目录存在，不需要改代码。
    """
    models_dir = PROJECT_ROOT_PATH / "models"
    if not models_dir.is_dir():
        return [DEFAULT_SETTINGS["embedding_model"]]

    options = sorted(
        f"./models/{entry.name}"
        for entry in models_dir.iterdir()
        if entry.is_dir() and not entry.name.startswith(".")
    )
    return options or [DEFAULT_SETTINGS["embedding_model"]]


# ==============================================================================
# 四、会话状态
# ==============================================================================

def ensure_session_state() -> None:
    """初始化/修复 st.session_state。

    这是每次脚本运行都要执行的“状态自愈”逻辑：
    - 设置文件缺字段时补默认值；
    - 旧会话没有 pinned 字段时补 False；
    - 会话被删光时自动建一个“新对话”；
    - current_session_id 指向已删除会话时自动切到第一个。
    这样做的好处是：无论用户怎么点、文件怎么被手动改，应用都能继续用。
    """
    loaded_settings = load_app_settings()
    if "settings" not in st.session_state:
        st.session_state.settings = loaded_settings
    else:
        # 兼容旧版本设置文件：新字段用默认值补齐，已有字段保留用户选择。
        st.session_state.settings = {**DEFAULT_SETTINGS, **st.session_state.settings}

    if "chats" not in st.session_state:
        st.session_state.chats = load_json(str(CHAT_FILE), {})

    if not isinstance(st.session_state.chats, dict):
        st.session_state.chats = {}

    for session_id, chat in list(st.session_state.chats.items()):
        if not isinstance(chat, dict):
            del st.session_state.chats[session_id]
            continue
        chat.setdefault("title", "新对话")
        chat.setdefault("messages", [])
        chat.setdefault("pinned", False)
        if not isinstance(chat.get("title"), str) or not chat["title"].strip():
            chat["title"] = "新对话"
        if not isinstance(chat.get("messages"), list):
            chat["messages"] = []
        if not isinstance(chat.get("pinned"), bool):
            chat["pinned"] = bool(chat.get("pinned", False))

    if not st.session_state.chats:
        new_id = str(uuid.uuid4())
        st.session_state.chats[new_id] = {"title": "新对话", "messages": [], "pinned": False}
        save_json(str(CHAT_FILE), st.session_state.chats)

    if (
        "current_session_id" not in st.session_state
        or st.session_state.current_session_id not in st.session_state.chats
    ):
        st.session_state.current_session_id = next(iter(st.session_state.chats))

    if "page" not in st.session_state:
        st.session_state.page = "chat"

    st.session_state.setdefault("is_generating", False)
    st.session_state.setdefault("pending_prompt", None)
    st.session_state.setdefault("current_generating_text", "")
    st.session_state.setdefault("rename_target", None)
    st.session_state.setdefault("stop_clicked", False)
    st.session_state.setdefault("flash", None)


def get_current_chat() -> dict[str, Any]:
    """返回当前会话对象；如果 ID 失效则自动修复。"""
    chats = st.session_state.chats
    session_id = st.session_state.current_session_id
    if session_id not in chats:
        st.session_state.current_session_id = next(iter(chats))
        session_id = st.session_state.current_session_id
    return chats[session_id]


def append_message(chat: dict[str, Any], role: str, content: str, refs: list[dict[str, Any]] | None = None) -> None:
    """向会话追加一条消息并立即落盘。

    为什么每次都 save：
    Streamlit 没有“页面关闭前”的可靠钩子；用户可能直接关浏览器。
    只有每次消息变化都写盘，才能保证历史记录不丢。
    refs 只有在非空时才写入，避免旧消息格式多出一个空数组。
    """
    message: dict[str, Any] = {"role": role, "content": content}
    if refs:
        message["refs"] = refs
    chat["messages"].append(message)
    save_json(str(CHAT_FILE), st.session_state.chats)


def finish_generation() -> None:
    """清理生成状态。

    注意：停止生成时不能直接调用它，必须先把
    current_generating_text 里的半成品保存下来，否则会丢内容。
    """
    st.session_state.is_generating = False
    st.session_state.pending_prompt = None
    st.session_state.current_generating_text = ""
    st.session_state.stop_clicked = False


def handle_stop_signal() -> None:
    """处理“停止生成”信号。

    为什么停止要这么绕：
    流式生成是阻塞的，生成循环里没法可靠监听按钮。于是按钮只做两件事：
    置 stop_clicked=True + st.rerun()。下一次脚本从头运行时，这里检测到
    标志，把已经生成的半成品写入会话，再清理状态。这样既能“随点随停”，
    又不会丢掉已经生成的内容。
    """
    if not st.session_state.get("stop_clicked", False):
        return

    partial_text = st.session_state.get("current_generating_text", "")
    if partial_text:
        chat = get_current_chat()
        append_message(chat, "assistant", partial_text + "\n\n*(已手动停止生成)*")

    finish_generation()
    st.rerun()


def set_flash(message: str, kind: str = "success") -> None:
    """设置一次性提示，下一次 rerun 后显示一次就消失。

    为什么不直接 st.success：很多操作后面会跟 st.rerun()，
    直接显示的成功提示会被 rerun 立刻冲掉。存到 session_state 再显示，
    可以保证用户确实看到反馈。
    """
    st.session_state.flash = {"message": message, "kind": kind}


def render_flash() -> None:
    """显示并消费一次性提示。"""
    flash = st.session_state.pop("flash", None)
    if not flash:
        return
    renderer = {
        "success": st.success,
        "info": st.info,
        "warning": st.warning,
        "error": st.error,
    }.get(flash.get("kind", "success"), st.success)
    renderer(flash["message"])


# ==============================================================================
# 五、登录页
# ==============================================================================

def load_auth_config() -> dict[str, Any]:
    """读取 auth_config.yaml，缺失/损坏时给出明确提示并停止。

    为什么鉴权必须最先执行：
    Streamlit 自上而下执行；只有把登录和 st.stop() 放在所有业务代码之前，
    未登录用户才无法触发文件读取、向量库初始化等操作。
    """
    try:
        with open(AUTH_CONFIG_FILE, "r", encoding="utf-8") as file:
            config = yaml.load(file, Loader=yaml.SafeLoader)
    except FileNotFoundError:
        st.error("缺少 auth_config.yaml 配置文件，请参考 README 创建。")
        st.stop()
    except OSError as exc:
        st.error(f"无法读取 auth_config.yaml：{exc}")
        st.stop()
    except yaml.YAMLError as exc:
        st.error(f"auth_config.yaml 格式错误：{exc}")
        st.stop()

    if not isinstance(config, dict) or "credentials" not in config:
        st.error("auth_config.yaml 缺少 credentials 配置。")
        st.stop()
    return config


def render_login_hero() -> None:
    """登录页的品牌区，让登录界面不像一个裸表单。"""
    st.markdown(
        f"""
        <div class="rag-login-hero">
            <div class="rag-brand-mark">{LOGO_SVG}</div>
            <h1>{APP_TITLE}</h1>
            <p>把本地文档变成可追溯的知识库。检索、引用、生成全部在本机完成，数据不离开你的机器。</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==============================================================================
# 六、通用 UI 组件
# ==============================================================================

def render_brand() -> None:
    """侧边栏品牌区。"""
    st.markdown(
        f"""
        <div class="rag-brand">
            <div class="rag-brand-mark">{LOGO_SVG}</div>
            <div>
                <div class="rag-brand-title">{APP_TITLE}</div>
                <div class="rag-brand-sub">{APP_EYEBROW}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_page_header(
    title: str,
    subtitle: str,
    stats: list[tuple[str, str]],
    status_text: str,
    status_warning: bool = False,
) -> None:
    """渲染页面标题、状态胶囊和 4 个指标卡。

    参数 stats 是 [(label, value), ...]；所有值都经过 html.escape，
    因为模型路径、文档目录等可能包含用户输入的特殊字符。
    """
    stats_html = "".join(
        f"""
        <div class="rag-stat">
            <span class="rag-stat-label">{html.escape(str(label))}</span>
            <span class="rag-stat-value" title="{html.escape(str(value))}">{html.escape(str(value))}</span>
        </div>
        """
        for label, value in stats
    )
    status_class = " is-warning" if status_warning else ""
    st.markdown(
        f"""
        <div class="rag-page-header">
            <div class="rag-header-row">
                <div>
                    <div class="rag-eyebrow">{APP_EYEBROW}</div>
                    <h1 class="rag-title">{html.escape(title)}</h1>
                    <p class="rag-subtitle">{html.escape(subtitle)}</p>
                </div>
                <div class="rag-status-pill{status_class}">
                    <span class="rag-dot"></span>{html.escape(status_text)}
                </div>
            </div>
            <div class="rag-stat-row">{stats_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_card_heading(title: str, description: str) -> None:
    """设置页卡片的小标题 + 说明。"""
    st.markdown(
        f"""
        <div class="rag-card-title">{html.escape(title)}</div>
        <div class="rag-card-desc">{html.escape(description)}</div>
        """,
        unsafe_allow_html=True,
    )


def render_empty_state(title: str, description: str, hints: list[str]) -> None:
    """渲染空状态卡片（索引为空 / 还没有对话）。

    空状态不是“空白”，它需要告诉用户下一步做什么：
    - 标题说明当前状态；
    - 描述解释为什么；
    - hints 给出能力边界/操作提示。
    """
    hints_html = "".join(f"<span>{html.escape(hint)}</span>" for hint in hints)
    st.markdown(
        f"""
        <div class="rag-empty">
            <div class="rag-empty-art">{EMPTY_STATE_SVG}</div>
            <h2>{html.escape(title)}</h2>
            <p>{html.escape(description)}</p>
            <div class="rag-empty-points">{hints_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_refs(refs: list[dict[str, Any]], expanded: bool = False) -> None:
    """用可折叠面板展示检索到的原始片段。

    每条片段显示：序号、来源文件、页码、章节，再显示正文。
    正文不是直接丢纯文本，而是先经过 format_reference_text()：
    - Markdown 列表/标题/表格正常渲染；
    - ``` 代码块自动识别并高亮；
    - $$...$$ 公式交给 KaTeX 渲染。
    这样用户核对原文时看到的是“可读的片段”，而不是一坨 PDF 换行文本。
    """
    if not isinstance(refs, list) or not refs:
        return

    with st.expander(f"检索依据 · {len(refs)} 个片段", expanded=expanded):
        st.caption("片段正文已按 Markdown / LaTeX 自动格式化；代码块会自动识别并高亮。")
        for index, ref in enumerate(refs, start=1):
            if not isinstance(ref, dict):
                continue
            source = html.escape(str(ref.get("source", "未知来源")))
            page = html.escape(str(ref.get("page", "未知")))
            section = html.escape(str(ref.get("section", "未知")))
            st.markdown(
                f"""
                <div class="rag-ref-meta">
                    <span class="rag-ref-chip rag-ref-index">片段 {index}</span>
                    <span class="rag-ref-chip">来源 {source}</span>
                    <span class="rag-ref-chip">第 {page} 页</span>
                    <span class="rag-ref-chip">章节 {section}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            # 正文先做结构化格式化，再交给 Streamlit Markdown/KaTeX 渲染。
            formatted_text = format_reference_text(str(ref.get("text", "")))
            if formatted_text:
                st.markdown(formatted_text)
            else:
                st.caption("（该片段没有可显示的文本）")
            if index < len(refs):
                st.divider()


# ==============================================================================
# 七、侧边栏
# ==============================================================================

def render_sidebar(authenticator) -> None:
    """侧边栏：品牌、新建会话、会话列表、设置/退出。

    布局要点：
    - 会话列表放在可滚动容器里，占满剩余高度；
    - 设置/退出放在底部容器，始终可见；
    - 当前会话用 primary 按钮 + 深色胶囊，非当前会话用 secondary。
    """
    with st.sidebar:
        render_brand()

        # “开启新对话”单独放在一个 key 容器里，方便 CSS 保持蓝色主操作。
        with st.container(key="new_chat"):
            if st.button("＋ 开启新对话", use_container_width=True, type="primary"):
                new_id = str(uuid.uuid4())
                st.session_state.chats[new_id] = {"title": "新对话", "messages": [], "pinned": False}
                save_json(str(CHAT_FILE), st.session_state.chats)
                st.session_state.current_session_id = new_id
                st.session_state.page = "chat"
                finish_generation()
                st.rerun()

        st.markdown('<div class="rag-section-label">会话</div>', unsafe_allow_html=True)

        with st.container(key="sidebar_sessions"):
            # 排序规则：置顶优先；同组内按插入顺序倒序（最近创建的在上）。
            # 用 enumerate 先记录插入顺序，避免 sorted 里反复 list(keys).index()
            # 造成 O(n^2) 的额外开销。
            indexed_chats = list(enumerate(st.session_state.chats.items()))
            sorted_chats = sorted(
                indexed_chats,
                key=lambda item: (item[1][1].get("pinned", False), item[0]),
                reverse=True,
            )

            for _, (session_id, session_data) in sorted_chats:
                is_current = session_id == st.session_state.current_session_id
                pin_prefix = "置顶 · " if session_data.get("pinned", False) else ""

                # 重命名态：输入框 + 保存/取消。
                if st.session_state.rename_target == session_id:
                    new_title = st.text_input(
                        "重命名对话",
                        value=session_data.get("title", "新对话"),
                        key=f"rename_input_{session_id}",
                    )
                    col_save, col_cancel = st.columns(2)
                    if col_save.button("保存", key=f"rename_save_{session_id}", use_container_width=True):
                        st.session_state.chats[session_id]["title"] = new_title.strip() or "新对话"
                        st.session_state.rename_target = None
                        save_json(str(CHAT_FILE), st.session_state.chats)
                        st.rerun()
                    if col_cancel.button("取消", key=f"rename_cancel_{session_id}", use_container_width=True):
                        st.session_state.rename_target = None
                        st.rerun()
                    continue

                # 普通态：标题按钮 + “⋯”菜单。
                col_title, col_menu = st.columns([8, 2])
                title = session_data.get("title", "新对话")
                displayed_title = title[:16] + ("…" if len(title) > 16 else "")
                if col_title.button(
                    f"{pin_prefix}{displayed_title}",
                    key=f"open_{session_id}",
                    use_container_width=True,
                    type="primary" if is_current else "secondary",
                ):
                    st.session_state.current_session_id = session_id
                    st.session_state.page = "chat"
                    finish_generation()
                    st.rerun()

                with col_menu.popover("⋯", use_container_width=True):
                    if st.button("重命名", key=f"rename_{session_id}", use_container_width=True):
                        st.session_state.rename_target = session_id
                        st.rerun()

                    pin_label = "取消置顶" if session_data.get("pinned", False) else "置顶"
                    if st.button(pin_label, key=f"pin_{session_id}", use_container_width=True):
                        st.session_state.chats[session_id]["pinned"] = not session_data.get("pinned", False)
                        save_json(str(CHAT_FILE), st.session_state.chats)
                        st.rerun()

                    if st.button("删除", key=f"delete_{session_id}", use_container_width=True):
                        del st.session_state.chats[session_id]
                        if st.session_state.rename_target == session_id:
                            st.session_state.rename_target = None
                        # 如果删的是当前会话，需要把 current_session_id 指向一个仍存在的会话；
                        # 如果一个都不剩，就新建一个空会话，保证应用至少有一个会话。
                        if st.session_state.current_session_id == session_id:
                            if st.session_state.chats:
                                st.session_state.current_session_id = next(iter(st.session_state.chats))
                            else:
                                new_id = str(uuid.uuid4())
                                st.session_state.chats[new_id] = {
                                    "title": "新对话",
                                    "messages": [],
                                    "pinned": False,
                                }
                                st.session_state.current_session_id = new_id
                        save_json(str(CHAT_FILE), st.session_state.chats)
                        st.rerun()

        with st.container(key="sidebar_footer"):
            # 在聊天页显示“设置”，在设置页显示“返回对话”，减少一次寻找成本。
            toggle_label = "返回对话" if st.session_state.page == "settings" else "设置"
            if st.button(toggle_label, key="toggle_settings", use_container_width=True):
                if st.session_state.page == "settings":
                    st.session_state.page = "chat"
                else:
                    st.session_state.page = "settings"
                    finish_generation()
                st.rerun()

            user_name = str(st.session_state.get("name", "用户"))
            initial = html.escape(user_name[:1].upper() or "U")
            st.markdown(
                f"""
                <div class="rag-user-chip">
                    <div class="rag-user-avatar">{initial}</div>
                    <div>
                        <div class="rag-user-name">{html.escape(user_name)}</div>
                        <div class="rag-user-role">已登录 · 本地模式</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            authenticator.logout("退出登录", "sidebar", use_container_width=True)


# ==============================================================================
# 八、设置页
# ==============================================================================

def copy_uploaded_files(docs_dir: str) -> int:
    """把 uploaded_docs 暂存目录里的文件复制到 {docs_dir}/uploaded。

    为什么要有暂存目录：
    Streamlit 的 file_uploader 只在上传当次返回文件对象，页面刷新就没了；
    先把文件写到磁盘，用户再点“重建索引/增量更新”，才能保证文件还在。
    复制而不是移动，是为了让“上传但暂未建索引”的文件仍然保留在暂存区，
    用户下次还可以再次建索引。
    """
    staging_dir = UPLOAD_STAGING_DIR
    if not staging_dir.is_dir():
        return 0

    target_dir = Path(docs_dir) / "uploaded"
    target_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    for source in sorted(staging_dir.iterdir()):
        if not source.is_file():
            continue
        if source.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        shutil.copy2(source, target_dir / source.name)
        copied += 1
    return copied


def render_settings_page() -> None:
    """设置页：生成参数、知识库、上传、索引操作。

    页面结构：
    - 顶部：返回按钮 + 标题/指标；
    - 三个 Tab：生成设置 / 知识库与索引 / 文件上传；
    - 底部：保存、增量更新、重建索引（始终可见，避免用户切 Tab 找不到）。
    """
    settings = st.session_state.settings

    if st.button("← 返回对话", key="settings_back"):
        st.session_state.page = "chat"
        st.rerun()

    # 设置页的指标：当前保存的模型 / 索引目录 / 文档目录。
    embedding_options = discover_embedding_models()
    current_model = settings["embedding_model"]
    if current_model not in embedding_options:
        # 用户手动改过配置文件、或模型目录被移动时，保留当前值，
        # 否则 selectbox 会因为 index 找不到而报错。
        embedding_options.insert(0, current_model)

    saved_chroma_dir = chroma_dir_for_model(current_model)
    render_page_header(
        "系统设置",
        "调整检索与生成参数，维护本地知识库。所有配置保存后会同时作用于 Web、CLI 与 MCP。",
        stats=[
            ("当前模型", os.path.basename(os.path.normpath(current_model))),
            ("向量库", display_path(saved_chroma_dir)),
            ("文档目录", settings["docs_dir"]),
            ("生成模式", "LLM 开启" if settings.get("use_llm", True) else "仅检索"),
        ],
        status_text="配置可随时修改",
    )
    render_flash()

    tab_generate, tab_index, tab_upload = st.tabs(["生成设置", "知识库与索引", "文件上传"])

    # ---------- Tab 1：生成设置 ----------
    with tab_generate:
        with st.container(key="settings_card_generate"):
            render_card_heading(
                "生成参数",
                "控制是否调用本地 LLM，以及 Ollama 的连接方式和采样策略。",
            )
            use_llm = st.checkbox(
                "启用本地 LLM 生成答案",
                value=bool(settings.get("use_llm", True)),
                help="关闭后只展示检索片段，适合调试检索质量。",
            )
            llm_model = st.text_input(
                "Ollama 模型名",
                value=str(settings.get("llm_model", DEFAULT_SETTINGS["llm_model"])),
                help="例如 qwen2.5:7b。需要先在 Ollama 中 pull 该模型。",
            )
            ollama_url = st.text_input(
                "Ollama API 地址",
                value=str(settings.get("ollama_url", DEFAULT_SETTINGS["ollama_url"])),
                help="默认 http://localhost:11434，Ollama 服务需要保持运行。",
            )
            top_k = st.slider(
                "检索片段数量 (Top K)",
                min_value=1,
                max_value=10,
                value=as_int(settings.get("top_k"), DEFAULT_SETTINGS["top_k"], 1, 10),
                help="越大信息越全，但上下文更长、生成更慢。建议 4-8。",
            )
            st.caption("提示：温度、top_p 等采样参数在 app.py 的 Ollama 调用处统一配置，保持回答稳定。")

    # ---------- Tab 2：知识库与索引 ----------
    with tab_index:
        with st.container(key="settings_card_index"):
            render_card_heading(
                "知识库与索引",
                "文档目录决定检索范围；Embedding 模型决定向量维度，切换模型会使用独立的向量库目录。",
            )

            docs_dir_value = st.text_input(
                "原始资料目录",
                value=str(settings.get("docs_dir", DEFAULT_SETTINGS["docs_dir"])),
                help="支持绝对路径或相对项目根目录的路径。目录下的第一层子目录会自动成为领域。",
            )
            # 下拉选项来自本地 models/ 目录，避免硬编码。
            try:
                embed_index = embedding_options.index(current_model)
            except ValueError:
                embed_index = 0
            embedding_model = st.selectbox(
                "Embedding 模型（本地）",
                options=embedding_options,
                index=embed_index,
                help="切换模型后必须重建索引；不同模型的向量维度不同，不能混用。",
            )

            # 领域下拉：选项来自 docs_dir 的第一层子目录；
            # 目录不存在时只有“全部”，不会让页面崩。
            # 输入框为空时不能把空路径解析成项目根目录，否则会把
            # models/skills/docs 误当成“领域”，所以这里先判断 strip()。
            try:
                domains = get_domains(resolve_project_path(docs_dir_value)) if docs_dir_value.strip() else []
            except Exception:
                domains = []
            domain_options = ["全部"] + domains
            selected_domain_value = settings.get("selected_domain", "全部")
            if selected_domain_value not in domain_options:
                selected_domain_value = "全部"
            selected_domain = st.selectbox(
                "知识领域过滤",
                options=domain_options,
                index=domain_options.index(selected_domain_value),
                help="只在该领域的片段中检索；“全部”表示不过滤。",
            )

            # 当前已保存索引的状态卡。这里用保存值而不是表单当前值，
            # 避免用户刚切换模型下拉就触发一个不存在索引的加载。
            try:
                index_count = get_chroma_collection(saved_chroma_dir, current_model).count()
                index_count_text = f"{index_count:,}"
                index_meta = (
                    f"集合 local_kb · {display_path(saved_chroma_dir)} · "
                    f"模型 {os.path.basename(os.path.normpath(current_model))}"
                )
            except Exception as exc:
                index_count_text = "—"
                index_meta = f"索引暂不可用：{exc}"

            st.markdown(
                f"""
                <div class="rag-index-card">
                    <div class="rag-index-main">
                        <span class="rag-index-label">当前已保存索引片段</span>
                        <span class="rag-index-value">{html.escape(index_count_text)}</span>
                    </div>
                    <div class="rag-index-meta">{html.escape(index_meta)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.caption("修改目录或模型后，请点击下方“重建索引”；只新增/修改少量文件时可用“增量更新”。")

    # ---------- Tab 3：文件上传 ----------
    with tab_upload:
        with st.container(key="settings_card_upload"):
            render_card_heading(
                "文件上传",
                "文件先存入项目根目录的 uploaded_docs，点击“重建索引/增量更新”后才会进入知识库。",
            )
            uploaded_files = st.file_uploader(
                "选择要加入知识库的文件",
                accept_multiple_files=True,
                type=UPLOAD_TYPES,
                help="支持 txt / md / pdf / docx / pptx / xlsx。扫描版 PDF 没有文本层，无法直接检索。",
            )
            if uploaded_files:
                st.caption(f"已选择 {len(uploaded_files)} 个文件，保存后请继续点击下方索引按钮。")
                if st.button("保存上传文件", key="save_uploads", use_container_width=True):
                    UPLOAD_STAGING_DIR.mkdir(parents=True, exist_ok=True)
                    for uploaded_file in uploaded_files:
                        target_path = UPLOAD_STAGING_DIR / uploaded_file.name
                        with open(target_path, "wb") as file:
                            file.write(uploaded_file.getbuffer())
                    set_flash(
                        f"已保存 {len(uploaded_files)} 个文件。请点击“增量更新”或“重建索引”使其生效。",
                        "success",
                    )
                    st.rerun()
            st.caption("同名文件会被覆盖；上传目录只做暂存，真正建库时统一从 docs_dir 扫描。")

    # ---------- 底部操作区：始终可见 ----------
    st.markdown("---")
    action_save, action_incremental, action_rebuild = st.columns(3)

    # 先把当前表单值收集成一个完整配置，三个按钮共用。
    new_settings = {
        "docs_dir": docs_dir_value.strip() or DEFAULT_SETTINGS["docs_dir"],
        "embedding_model": embedding_model,
        "use_llm": bool(use_llm),
        "llm_model": llm_model.strip() or DEFAULT_SETTINGS["llm_model"],
        "ollama_url": ollama_url.strip() or DEFAULT_SETTINGS["ollama_url"],
        "selected_domain": selected_domain,
        "top_k": int(top_k),
    }

    def persist_settings() -> None:
        """把表单值写入 session_state 和 app_settings.json。"""
        st.session_state.settings = new_settings
        save_app_settings(new_settings)

    with action_save:
        if st.button("保存设置并返回", type="primary", use_container_width=True):
            persist_settings()
            set_flash("设置已保存。", "success")
            st.session_state.page = "chat"
            st.rerun()

    with action_incremental:
        if st.button("增量更新", use_container_width=True):
            persist_settings()
            docs_dir_resolved = resolve_project_path(new_settings["docs_dir"])
            chroma_dir = chroma_dir_for_model(new_settings["embedding_model"])
            try:
                copied = copy_uploaded_files(docs_dir_resolved)
                stats = update_index(chroma_dir, docs_dir_resolved, new_settings["embedding_model"])
                if stats["added"] or stats["modified"] or stats["deleted"]:
                    failed = stats.get("failed", 0)
                    message = (
                        "增量更新完成："
                        f"新增 {stats['added']} 个文件，修改 {stats['modified']} 个，"
                        f"删除 {stats['deleted']} 个；同步上传文件 {copied} 个。"
                    )
                    if failed:
                        message += f" 其中 {failed} 个文件处理失败，下次更新会自动重试。"
                    set_flash(message, "warning" if failed else "success")
                else:
                    set_flash("没有检测到文件变化，索引保持不变。", "info")
                st.rerun()
            except Exception as exc:
                st.error(f"增量更新失败：{exc}")

    with action_rebuild:
        if st.button("重建索引", use_container_width=True):
            persist_settings()
            docs_dir_resolved = resolve_project_path(new_settings["docs_dir"])
            chroma_dir = chroma_dir_for_model(new_settings["embedding_model"])
            try:
                with st.spinner("正在解析文档并生成向量，请稍候..."):
                    copied = copy_uploaded_files(docs_dir_resolved)
                    chunk_count = rebuild_index(chroma_dir, docs_dir_resolved, new_settings["embedding_model"])

                # 重建后旧回答可能已经过时，清空当前会话的消息和标题，
                # 避免用户继续在旧上下文里提问。
                current_chat = get_current_chat()
                current_chat["messages"] = []
                current_chat["title"] = "新对话"
                save_json(str(CHAT_FILE), st.session_state.chats)

                set_flash(
                    f"索引重建完成：共写入 {chunk_count:,} 个片段；同步上传文件 {copied} 个。",
                    "success",
                )
                st.rerun()
            except Exception as exc:
                st.error(f"重建索引失败：{exc}")


# ==============================================================================
# 九、聊天页
# ==============================================================================

def stream_ollama(ollama_url: str, model: str, prompt: str) -> Iterator[str]:
    """调用 Ollama /api/generate，逐段 yield 文本。

    为什么用 generate 而不是 chat：
    本项目已经自己拼好了 system prompt + 参考资料 + 用户问题，
    /api/generate 直接接收完整 prompt，逻辑更直观。
    stream=True 让 Ollama 返回 NDJSON，每行一个 JSON，逐行解析即可。

    超时用 (10, 300)：
    - 10 秒连接超时：Ollama 没启动时不要卡太久；
    - 300 秒读取超时：本地大模型生成长回答可能很慢，给足时间。
    """
    full_text = ""
    try:
        response = requests.post(
            f"{ollama_url}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": True,
                "options": {
                    "num_predict": 4096,   # 允许生成长篇答案
                    "temperature": 0.1,    # 极低温度：更确定、更少发散
                    "top_p": 0.8,          # 收窄采样范围
                    "repeat_penalty": 1.1, # 抑制重复啰嗦
                },
            },
            stream=True,
            timeout=(10, 300),
        )
        response.raise_for_status()

        for line in response.iter_lines():
            if not line:
                continue
            chunk = json.loads(line.decode("utf-8"))
            if "error" in chunk:
                raise RuntimeError(chunk["error"])
            piece = chunk.get("response", "")
            if not piece:
                continue
            full_text += piece
            # 每次更新中间文本，停止按钮才能保存“已经生成到哪”。
            st.session_state.current_generating_text = full_text
            yield clean_latex_format(piece)
    except Exception as exc:
        # 生成失败也作为回答的一部分展示并保存，用户能看到具体原因，
        # 而不是页面直接抛异常。
        error_message = f"\n\n**调用 Ollama 失败：{exc}**"
        full_text += error_message
        st.session_state.current_generating_text = full_text
        yield error_message


def render_generation(
    collection,
    settings: dict[str, Any],
    current_chat: dict[str, Any],
    chroma_dir: str,
) -> None:
    """处理一次“检索 + 生成”流程。

    两段式设计：
    1) chat_input 只写入 pending_prompt + is_generating，然后 rerun；
    2) 下一次运行在这里统一处理检索、生成、保存、rerun。
    这样用户消息能先正常显示，生成状态面板也能稳定地放在页面下方。
    """
    if not (st.session_state.is_generating and st.session_state.pending_prompt):
        return

    prompt = str(st.session_state.pending_prompt)
    st.session_state.pending_prompt = None  # 立刻消费，避免重复生成

    # 先记录用户消息并落盘；第一条消息顺便作为会话标题。
    append_message(current_chat, "user", prompt)
    if current_chat.get("title", "新对话") == "新对话":
        # 把换行/连续空格压成单个空格，避免侧边栏标题被拉成多行。
        current_chat["title"] = " ".join(prompt.split())[:20] or "新对话"
        save_json(str(CHAT_FILE), st.session_state.chats)

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.status("正在检索知识库…", expanded=True) as status:
            col_status, col_stop = st.columns([8, 2])
            with col_status:
                st.write("正在知识库中检索相关片段…")
            with col_stop:
                if st.button("停止生成", key="stop_in_status", use_container_width=True):
                    st.session_state.stop_clicked = True
                    st.rerun()

            domain_filter = None if settings.get("selected_domain") == "全部" else settings.get("selected_domain")
            try:
                documents, metadatas = query_collection(
                    chroma_dir,
                    prompt,
                    settings["embedding_model"],
                    top_k=as_int(settings.get("top_k"), 4, 1, 20),
                    domain=domain_filter,
                    collection=collection,
                )
            except Exception as exc:
                status.update(label="检索失败", state="error", expanded=False)
                append_message(current_chat, "assistant", f"检索失败：{exc}")
                finish_generation()
                st.rerun()

            if not documents:
                status.update(label="未找到相关内容", state="complete", expanded=False)
                append_message(
                    current_chat,
                    "assistant",
                    "未检索到相关内容。请尝试更换关键词，或在设置页检查知识库索引。",
                )
                finish_generation()
                st.rerun()

            # 把 doc 文本和元数据整理成统一 refs 结构，既用于展示也存入会话。
            refs = []
            for document, metadata in zip(documents, metadatas):
                metadata = metadata or {}
                refs.append({
                    "text": document,
                    "source": os.path.basename(metadata.get("source", "未知")),
                    "page": metadata.get("page", "未知"),
                    "section": metadata.get("section", "未知"),
                })

            st.write(f"命中 {len(refs)} 个相关片段。")
            render_refs(refs)
            status.update(label="正在生成回答…", state="running")

        if settings.get("use_llm", True):
            # 给每个片段编号、来源、章节、页码，方便模型在回答中引用。
            context = "\n\n".join(
                f"[片段{index}] 来源：{ref['source']} "
                f"({ref['section']} 第{ref['page']}页)\n{ref['text']}"
                for index, ref in enumerate(refs, start=1)
            )

            # 系统提示词是 RAG 效果的关键：
            # - 只依据资料，禁止脑补；
            # - 尽量详尽，分层次排版；
            # - 标注依据片段，方便用户核对；
            # - 公式统一用 $ / $$，避免 Streamlit 渲染失败。
            system_prompt = (
                "你是一个极其严谨、专业的资深知识库专家。你的任务是根据提供的【参考资料】，"
                "全面、详尽、准确地解答用户问题。\n\n"
                "【核心原则（必须严格遵守）】：\n"
                "1. **绝对忠实于资料**：你的所有结论、数据和步骤，必须百分之百来源于提供的"
                "【参考资料】。绝对禁止使用你自带的知识库进行脑补或编造！如果资料中确实没有答案，"
                "请直接回答“资料中未找到相关信息”。\n"
                "2. **穷尽所有细节**：在忠于资料的前提下，**回答得越详细越好**。不要只说一句话总结，"
                "请把资料中相关的背景、原理、步骤、优缺点、具体数据等所有细节都提取出来。\n"
                "3. **逻辑化重组**：资料可能是碎片化的，你需要把它们按照合理的逻辑串联起来，"
                "分点、分层次作答。可以使用标题、列表、加粗来增强可读性。\n"
                "4. **标注出处**：在回答的每个关键点后面，必须标注信息来源于哪个片段，"
                "例如 `（依据片段1、片段2）`，以便用户核对原文。\n\n"
                "【排版要求】：如果你需要输出数学公式或符号，请必须使用标准的 LaTeX 语法。"
                "行内公式请使用单个 `$` 包裹（例如 `$E=mc^2$`），独立公式请使用双 `$$` 包裹"
                "（例如 `$$ \\frac{a}{b} $$`）。绝对不要使用 `\\( \\)` 或 `\\[ \\]` 作为定界符。"
            )
            full_prompt = (
                f"{system_prompt}\n\n【参考资料】：\n{context}\n\n"
                f"【用户问题】：{prompt}\n\n【详尽且准确的解答】："
            )

            response_text = st.write_stream(
                stream_ollama(
                    settings["ollama_url"],
                    settings["llm_model"],
                    full_prompt,
                )
            )
            cleaned_text = clean_latex_format(response_text)
            status.update(label="生成完成", state="complete", expanded=False)
            append_message(current_chat, "assistant", cleaned_text, refs)
        else:
            # LLM 关闭时也要给用户一个明确结果，而不是让页面停在“生成中”。
            # 这是旧版本的一个实际 bug：use_llm=False 时 is_generating 永远为 True，
            # 输入框一直禁用。这里改成保存一条“仅检索”的回答并正常收尾。
            answer = (
                f"已检索到 {len(refs)} 个相关片段。当前已关闭 LLM 生成，"
                "请展开上方“检索依据”查看原文；如需生成式回答，请在设置页开启本地 LLM。"
            )
            st.markdown(answer)
            status.update(label="检索完成", state="complete", expanded=False)
            append_message(current_chat, "assistant", answer, refs)

    finish_generation()
    st.rerun()


def render_chat_input() -> None:
    """底部输入区：正常时是 chat_input，生成中替换为禁用框 + 停止按钮。

    为什么要替换而不是直接 disabled chat_input：
    Streamlit 目前不支持动态禁用 chat_input；用一个普通 text_input 模拟
    “正在生成中”的状态，交互上最接近原生聊天产品。
    """
    if st.session_state.is_generating:
        col_input, col_button = st.columns([9, 1])
        with col_input:
            st.text_input(
                "正在生成中，如需停止请点击右侧按钮",
                disabled=True,
                label_visibility="collapsed",
                placeholder="正在生成中…",
            )
        with col_button:
            if st.button("停止", key="stop_bottom", use_container_width=True, help="停止生成并保留已生成内容"):
                st.session_state.stop_clicked = True
                st.rerun()
        return

    if prompt := st.chat_input("给 RAG 助手发送消息…"):
        # 只做“记录 + 置位 + rerun”，真正的检索/生成放到 render_generation。
        st.session_state.pending_prompt = prompt
        st.session_state.is_generating = True
        st.session_state.current_generating_text = ""
        st.rerun()


def render_chat_page() -> None:
    """聊天页：初始化向量库 -> 渲染历史 -> 处理生成 -> 底部输入。"""
    settings = st.session_state.settings
    embedding_model = str(settings["embedding_model"])
    chroma_dir = chroma_dir_for_model(embedding_model)
    selected_domain = settings.get("selected_domain", "全部")
    top_k = as_int(settings.get("top_k"), 4, 1, 20)

    # 初始化向量库；失败时保留错误信息，页面继续渲染，避免白屏。
    collection = None
    doc_count = 0
    init_error = None
    try:
        collection = get_chroma_collection(chroma_dir, embedding_model)
        doc_count = collection.count()
    except Exception as exc:
        init_error = str(exc)

    if init_error:
        status_text = "向量库不可用"
        status_warning = True
    elif doc_count == 0:
        status_text = "索引为空"
        status_warning = True
    else:
        status_text = "知识库就绪"
        status_warning = False

    # 宽屏时把标题/状态/指标卡固定到右侧空白区（见 THEME_CSS 的
    # chat_context_rail 媒体查询）；窄屏时它就是普通的页头，不改变原布局。
    with st.container(key="chat_context_rail"):
        render_page_header(
            APP_TITLE,
            "只基于你的文档回答，并给出可核对的引用片段。检索与生成全部在本机完成。",
            stats=[
                ("片段总数", f"{doc_count:,}" if isinstance(doc_count, int) else str(doc_count)),
                ("知识领域", selected_domain),
                ("Embedding", os.path.basename(os.path.normpath(embedding_model))),
                ("Top K", str(top_k)),
            ],
            status_text=status_text,
            status_warning=status_warning,
        )
    render_flash()

    if init_error:
        st.error(f"初始化向量库失败：{init_error}")
        if st.button("重新加载向量库", key="retry_index"):
            clear_collection_cache()
            st.rerun()
        st.markdown(
            '<div class="rag-footer">本地运行 · 数据不离开本机</div>',
            unsafe_allow_html=True,
        )
        return

    current_chat = get_current_chat()

    # 先渲染历史消息，保证刷新/切换会话后上下文还在。
    for message in current_chat["messages"]:
        # 兼容旧数据：content 理论上一定是字符串，但手动改过 JSON 时可能缺失。
        if not isinstance(message, dict):
            continue
        role = message.get("role", "assistant")
        content = str(message.get("content", "") or "")
        with st.chat_message(role):
            st.markdown(clean_latex_format(content))
            if message.get("refs"):
                render_refs(message["refs"])

    if doc_count == 0:
        render_empty_state(
            "知识库还是空的",
            "先去设置页上传文档并重建索引，之后就可以在这里提问了。首次建库需要解析文档并生成向量，文档较多时会花一些时间。",
            ["支持 PDF / Word / PPT / Excel / 文本", "领域由目录自动识别", "索引完成后即可检索"],
        )
        _, center_mid, _ = st.columns([1, 1.2, 1])
        with center_mid:
            if st.button("前往设置重建索引", type="primary", use_container_width=True):
                st.session_state.page = "settings"
                st.rerun()
        st.markdown(
            '<div class="rag-footer">本地运行 · 数据不离开本机</div>',
            unsafe_allow_html=True,
        )
        return

    # 没有任何历史消息时，给一个“开始提问”的空状态，而不是大片空白。
    if not current_chat["messages"]:
        render_empty_state(
            "从一个有依据的问题开始",
            "回答只会引用知识库中的原文，并在下方列出检索依据。你可以问概念、流程、代码示例，或让助手对比多个文档中的说法。",
            ["离线向量检索", "引用可追溯", "支持多轮追问"],
        )

    render_generation(collection, settings, current_chat, chroma_dir)
    render_chat_input()
    st.markdown(
        '<div class="rag-footer">本地运行 · 数据不离开本机 · 回答仅供核对，不替代原文</div>',
        unsafe_allow_html=True,
    )


# ==============================================================================
# 十、应用入口
# ==============================================================================

def main() -> None:
    """应用主流程。

    顺序非常重要：
    1) 先加载 .env（必须在读取 auth_config 之前）；
    2) set_page_config 必须是第一个 Streamlit 命令；
    3) 注入主题（登录页也需要样式）；
    4) 鉴权，未登录直接 st.stop()；
    5) 初始化会话状态；
    6) 渲染侧边栏和当前页面。
    """
    # .env 里目前只放 COOKIE_KEY；先加载，后面读 auth_config 时才能拿到。
    load_dotenv_file(ENV_FILE)

    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(f"<style>{THEME_CSS}</style>", unsafe_allow_html=True)

    # ---------- 鉴权 ----------
    auth_config = load_auth_config()
    cookie_config = auth_config.get("cookie", {})
    cookie_key = os.environ.get("COOKIE_KEY") or cookie_config.get("key") or "local-rag-cookie-key"
    cookie_name = cookie_config.get("name", "rag_auth_cookie")
    cookie_expiry = cookie_config.get("expiry_days", 30)

    authenticator = stauth.Authenticate(
        auth_config["credentials"],
        cookie_name,
        cookie_key,
        cookie_expiry,
    )

    # 先用一个占位符预留登录页品牌区的位置，再执行 login()。
    # 这样已登录（cookie 有效）的用户不会先看到一瞬登录页标题；
    # 未登录时再把品牌内容填进占位符，位置仍然在登录表单上方。
    hero_placeholder = st.empty()
    authenticator.login(fields={
        "Form name": "登录",
        "Username": "用户名",
        "Password": "密码",
        "Login": "登录",
    })

    authentication_status = st.session_state.get("authentication_status")
    if authentication_status is False:
        with hero_placeholder.container():
            render_login_hero()
        with st.container(key="login_message"):
            st.error("用户名或密码不正确，请重试。")
        st.stop()
    if authentication_status is None:
        with hero_placeholder.container():
            render_login_hero()
        with st.container(key="login_message"):
            st.warning("请输入用户名和密码以访问系统。")
        st.stop()

    # ---------- 登录后的业务区 ----------
    ensure_session_state()
    handle_stop_signal()
    render_sidebar(authenticator)

    if st.session_state.page == "settings":
        render_settings_page()
    else:
        render_chat_page()


if __name__ == "__main__":
    # Streamlit 以脚本方式执行本文件，__name__ 为 "__main__"；
    # 被其它模块 import 时不会自动渲染页面，避免测试/复用产生副作用。
    main()

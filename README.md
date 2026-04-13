# AI Testing Agent

> 让任何文本大模型都能理解图片和 PDF 文档的 Agent 框架

---

## 核心能力

| 能力 | 说明 |
|------|------|
| **图片理解** | 通过 Vision 模型（豆包/OpenAI）分析图片，转为文字描述 |
| **PDF 解析** | PyMuPDF4LLM 提取文本 + Vision 模型识别内嵌图片 |
| **本地 PDF 路径识别** | 支持 Windows/Linux 本地路径自动检测与解析 |
| **在线 PDF URL 识别** | 自动识别并下载解析 `arxiv.org/pdf/xxx` 等在线 PDF |
| **智能缓存** | 基于 MD5 内容哈希的 SQLite 持久化 LRU 缓存，同文件不重复处理 |
| **双模式切换** | 文本降级模式（Vision+DeepSeek）/ 豆包原生多模态模式 |

---

## 系统架构

```
前端上传图片/PDF
       │
       ▼
┌─────────────────────────────────────────────────────────┐
│          check_message_flow (before_model 中间件)         │
│                                                         │
│  ┌────────────────────┐    ┌──────────────────────┐      │
│  │ use_multimodal=ON  │    │ use_multimodal=OFF  │      │
│  │  豆包原生多模态     │    │  Vision + 文本降级   │      │
│  └────────┬───────────┘    └──────────┬───────────┘      │
│           │                          │                   │
└───────────┼──────────────────────────┼───────────────────┘
            │                          │
            ▼                          ▼
    ┌──────────────┐         ┌──────────────────────┐
    │  豆包多模态   │         │  message_transformer  │
    │  直接处理     │         │                       │
    │              │         │  image_url ──→ Vision  │
    │  返回最终回答 │         │  file(pdf) ──→ PDF解析 │
    └──────────────┘         │  PDF URL ──→ 下载+解析  │
                             │  本地路径 ──→ 路径+解析  │
                             └──────────────┬───────────┘
                                            │
                                            ▼
                               ┌──────────────────────┐
                               │  DeepSeek / 其他文本模型 │
                               │  （只收到纯文本）       │
                               └──────────────────────┘
```

---

## 目录结构

```
ai-testing-agent/
├── start_server.py              # 服务启动入口
├── graph.json                   # LangGraph 图配置
├── pyproject.toml               # 项目依赖
│
├── src/
│   ├── core/                    # 核心模块
│   │   ├── cache.py             # SQLite LRU 缓存
│   │   ├── file_utils.py        # Base64 解码 & 文件保存
│   │   ├── image_analyzer.py    # Vision 模型图片分析
│   │   ├── pdf_analyzer.py      # PDF 解析（含在线 URL / 本地路径）
│   │   ├── message_transformer.py # 多模态→纯文本转换器
│   │   ├── middleware.py        # before_model / after_model 钩子
│   │   ├── hatch_agent.py       # Agent 创建入口
│   │   └── ARCHITECTURE.md      # 详细架构文档
│   │
│   ├── agents/
│   │   └── testcases/
│   │       └── agent.py         # 测试用例设计 Agent
│   │
│   └── processors/
│       └── base64_processor.py  # 独立 PDF 处理器（可复用组件）
│
├── utils/
│   └── model_factory.py         # 多厂商模型工厂（DeepSeek/豆包/OpenAI/...）
│
└── examples/
    ├── hatch_agent.py           # 主 Agent 示例
    ├── hatch_agent_normal.py    # 普通 Agent 示例
    ├── hatch_deepagent.py       # DeepSeek Agent 示例
    ├── hatch_subagent.py        # 子 Agent 示例
    ├── hatch_local_mcp.py       # 本地 MCP Agent 示例
    ├── mcp_agent.py             # MCP Agent 示例
    ├── parse_online_file.py     # 在线文件解析示例
    ├── tool_factory.py          # 工具工厂
    └── base64_processor.py      # Base64 处理器示例
```

---

## 快速开始

### 1. 安装依赖

```bash
# 建议使用 uv 管理虚拟环境
uv venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

uv pip install \
    langchain \
    langchain-core \
    langchain-openai \
    langchain-community \
    langchain-pymupdf4llm \
    pymupdf \
    python-dotenv \
    langgraph \
    langgraph-checkpoint \
    langgraph-sdk \
    sse-starlette \
    httpx
```

### 2. 配置环境变量

创建 `.env` 文件：

```env
# DeepSeek（主文本模型）
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxx
DEEPSEEK_MODEL_NAME=deepseek-chat

# 豆包（Vision 图片分析 / 多模态模式）
DOUBAO_API_KEY=xxxxxxxxxxxxxxxx
DOUBAO_MODEL_NAME=doubao-seed-2-0-lite-260215
DOUBAO_BASE_URL=https://ark.cn-beijing.volces.com/api/v3

# OpenAI（Vision 备选）
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxx
```

### 3. 启动服务

```bash
python start_server.py
```

服务地址：`http://localhost:2026`

| 地址 | 说明 |
|------|------|
| `http://localhost:2026` | API 服务 |
| `http://localhost:2026/docs` | API 文档（Swagger） |
| `http://localhost:2026/ui` | LangGraph Studio |
| `http://localhost:2026/ok` | 健康检查 |

---

## 核心模块说明

### 缓存机制（`cache.py`）

基于 **MD5 内容哈希**的 SQLite 持久化 LRU 缓存：

```python
from src.core.cache import get_pdf_cached, put_pdf_cache, get_cache_stats

# 查询缓存（内存 → SQLite 自动回填）
result = get_pdf_cached("md5_hash_key")

# 写入缓存（同时写内存 + SQLite）
put_pdf_cache("md5_hash_key", "解析结果文本")

# 查看统计
print(get_cache_stats())
# {'image_memory_size': 3, 'pdf_memory_size': 1, 'image_total_entries': 5, ...}
```

**缓存策略：**
- 图片和 PDF 独立缓存，互不干扰
- 内存 LRU（128 条）+ SQLite WAL 持久化
- 系统重启后自动从 SQLite 恢复缓存

### 消息转换器（`message_transformer.py`）

将多模态 HumanMessage 转换为纯文本：

```python
from langchain_core.messages import HumanMessage
from src.core.message_transformer import transform_multimodal_message

msg = HumanMessage(content=[
    {"type": "text", "text": "分析这个 PDF 讲了什么"},
    {"type": "file", "source": "...", "filename": "doc.pdf"}
])

new_msg = transform_multimodal_message(msg)
# new_msg.content = 纯文本（前端可见）
# new_msg.additional_kwargs["attachments"] = 元信息（供前端渲染）
```

### 中间件（`middleware.py`）

支持两种运行模式（前端通过 `configurable` 切换）：

| 模式 | 配置键 | 说明 |
|------|--------|------|
| 文本降级（默认） | `use_multimodal_model: False` | Vision 模型分析图片/PDF，转纯文本发给 DeepSeek |
| 多模态直通 | `use_multimodal_model: True` | 直接发给豆包多模态模型处理 |

---

## 配置说明

### 环境变量

| 变量 | 必需 | 默认值 | 说明 |
|------|------|--------|------|
| `DEEPSEEK_API_KEY` | 是 | - | DeepSeek API 密钥 |
| `DEEPSEEK_MODEL_NAME` | 否 | `deepseek-chat` | DeepSeek 模型名 |
| `DOUBAO_API_KEY` | 推荐 | - | 豆包 API 密钥（Vision 分析用） |
| `DOUBAO_MODEL_NAME` | 否 | `doubao-seed-2-0-lite-260215` | 豆包模型名 |
| `DOUBAO_BASE_URL` | 否 | 火山引擎地址 | 豆包 API 地址 |
| `OPENAI_API_KEY` | 推荐 | - | OpenAI API 密钥（Vision 备选） |

### 启动参数（`start_server.py`）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `host` | `0.0.0.0` | 监听地址 |
| `port` | `2026` | 监听端口 |
| `workers` | `1` | 工作进程数（Windows 开发建议用 1） |

---

## 调试工具

### 查看缓存状态

```python
from src.core.cache import get_cache_stats, clear_all_caches

stats = get_cache_stats()
print(stats)
```

### 清空缓存

```python
from src.core.cache import clear_all_caches

clear_all_caches()                    # 清空内存 + SQLite
clear_all_caches(clear_persistent=False)  # 只清空内存
```

### 测试单个模块

```bash
# 测试 PDF 解析
python -c "
from src.core.pdf_analyzer import analyze_pdf
result = analyze_pdf('D:/test.pdf', '这个文档讲了什么')
print(result)
"

# 测试消息转换
python -c "
from langchain_core.messages import HumanMessage
from src.core.message_transformer import transform_multimodal_message
msg = HumanMessage(content=[{'type': 'text', 'text': '分析这张图'}, {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,...'}}])
new_msg = transform_multimodal_message(msg)
print(new_msg.content)
"
```

---

## 常见问题

### Q: 两个浏览器同时上传，解析是并行的吗？

单 `workers=1` 模式下，uvicorn 使用 **asyncio 异步并发**，I/O 等待期间可自动切换处理其他请求。纯 CPU 密集计算（如大模型推理）会排队，但 Vision/PDF 解析部分主要是网络 I/O，可以有效并发。

### Q: 服务重启后缓存会丢失吗？

不会。缓存使用 **SQLite WAL 模式持久化**，重启后自动从 `.attachment_cache.db` 恢复。

### Q: 本地 PDF 路径支持哪些格式？

支持 Windows 反斜杠路径（如 `D:\docs\file.pdf`）和 Linux 正斜杠路径（如 `/home/user/docs/file.pdf`）。系统会根据文件修改时间（mtime）自动判断是否需要重新解析。

### Q: 在线 PDF 支持哪些格式？

支持带 `.pdf` 后缀的标准 URL（如 `https://example.com/doc.pdf`）以及 arxiv 格式（如 `https://arxiv.org/pdf/2604.08000`）。

---

## 开发指南

### 添加新的 Vision 模型提供商

在 `src/core/image_analyzer.py` 中添加新函数，并在 providers 列表中注册：

```python
def _analyze_with_new_provider(image_path: str, prompt: str) -> Optional[str]:
    # ... 构建客户端和调用逻辑 ...
    return response.content

# 在 analyze_image() 的 providers 列表中追加
providers = [
    ("doubao", _analyze_with_doubao),
    ("openai", _analyze_with_openai),
    ("new_provider", _analyze_with_new_provider),  # 新增
]
```

### 添加新的文件类型支持

在 `src/core/message_transformer.py` 的 `_handle_file_part()` 中添加分支处理逻辑。

---

## 许可证

内部项目，保留所有权利。

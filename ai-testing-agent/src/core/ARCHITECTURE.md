# Hatch Agent Core 架构文档

> 版本：v2.0 | 最后更新：2026-04-10

---

## 1. 概述

Hatch Agent Core 是一个**多模态附件处理中间件**，让非多模态大模型（如 DeepSeek）也能"理解"图片和 PDF 文档。

### 核心能力

| 能力 | 说明 |
|------|------|
| **图片理解** | 通过 Vision 模型（豆包/OpenAI）分析图片，转为文字描述 |
| **PDF 解析** | PyMuPDF4LLM 提取文本 + LLMImageBlobParser 识别内嵌图 |
| **智能缓存** | 基于 MD5 内容哈希的 LRU 缓存，同文件不重复处理 |
| **消息转换** | 多模态 HumanMessage → 纯文本，对上层透明 |
| **双模式** | 文本降级模式 / 豆包原生多模态模式，前端可切换 |

---

## 2. 目录结构

```
ai-testing-agent/
├── core/                          # ← 核心模块（本次拆分）
│   ├── __init__.py                # 包入口，导出 agent 实例
│   ├── hatch_agent.py             # Agent 创建入口（精简）
│   ├── cache.py                   # LRU 缓存模块
│   ├── file_utils.py              # Base64 解码 & 文件保存
│   ├── image_analyzer.py          # Vision 模型图片分析
│   ├── pdf_analyzer.py            # PDF 文档解析
│   ├── message_transformer.py     # 多模态→纯文本转换器
│   └── middleware.py              # Agent 中间件 (before/after model)
│
├── processors/
│   └── base64_processor.py         # 独立 PDF 处理器（可复用组件）
│
├── agents/
│   └── testcases/agent.py          # 测试用例 Agent
│
└── utils/
    └── model_factory.py            # 模型工厂（外部依赖）
```

---

## 3. 模块详解

### 3.1 `cache.py` — LRU 缓存

**职责：** 避免同一文件重复调用 Vision 模型 / PDF 解析器

```
输入: base64 数据 → MD5(content) → 哈希值作为缓存 Key
                              ↓
                    命中? → 直接返回结果
                    未命中 → 执行分析 → 写入缓存 → 返回结果
```

**核心类/函数：**

| 名称 | 类型 | 说明 |
|------|------|------|
| `compute_content_hash()` | function | 从 data URL/base64 计算 MD5 |
| `get_image_cached()` / `put_image_cache()` | function | 图片分析缓存读写 |
| `get_pdf_cached()` / `put_pdf_cache()` | function | PDF 解析缓存读写 |
| `AttachmentCache` | class | 统一缓存管理（面向对象风格）|
| `get_cache_stats()` / `clear_all_caches()` | function | 监控与管理接口 |

**配置参数：**
```python
MAX_CACHE_SIZE = 128   # 每种类型最大缓存条目数，超出自动淘汰最久未用的
```

**依赖：** 标准库 (`hashlib`, `re`, `collections.OrderedDict`)

---

### 3.2 `file_utils.py` — 文件工具

**职责：** Base64 编码数据解码并保存为本地临时文件

| 函数 | 说明 |
|------|------|
| `extract_base64_from_data_url(data_url)` | 解析 Data URL → `(mime_type, bytes)` |
| `save_base64_to_local(data_url, save_dir, filename)` | 解码 + 推断扩展名 + 保存到本地 |
| `save_base64_image_to_local(data_url)` | 向后兼容包装，仅返回路径 |

**支持 MIME 类型映射：**
- 图片：png, jpg, gif, webp, svg, bmp, tiff
- 文档：pdf, doc, docx, txt, json, csv
- 音频：mp3, wav
- 视频：mp4

**依赖：** 标准库 (`os`, `re`, `base64`, `tempfile`, `uuid`)

---

### 3.3 `image_analyzer.py` — 图片分析

**职责：** 使用 Vision 模型将图片转为文字描述

```
图片文件 → 构建 base64 data URL + 分析提示词
         → 尝试 doubao Vision → 成功？→ 返回描述
                                ↓ 失败
         → 尝试 openai GPT-4o-mini → 成功？→ 返回描述
                                       ↓ 失败
         → 返回兜底占位提示
```

| 函数 | 说明 |
|------|------|
| `analyze_image(image_path, user_text)` | 主入口，自动按优先级选择模型 |
| `_analyze_with_doubao(image_path, prompt)` | 豆包多模态实现 |
| `_analyze_with_openai(image_path, prompt)` | OpenAI GPT-4o-mini 实现 |

**依赖：** `langchain-openai`, `langchain-core`, `python-dotenv`

---

### 3.4 `pdf_analyzer.py` — PDF 解析

**职责：** 使用 PyMuPDF4LLM + LLMImageBlobParser 提取完整文档内容

```
PDF 文件 → PyMuPDF4LLMLoader(mode="page", extract_images=True)
         → LLMImageBlobParser(Vision 模型) 处理内嵌图
         → 按页组装结构化输出
```

| 函数 | 说明 |
|------|------|
| `analyze_pdf(pdf_path, user_text)` | 主入口，完整解析链路 |
| `_get_vision_model_client()` | 获取 PDF 内嵌图的 Vision 客户端 |
| `_assemble_output()` | 组装最终输出文本 |
| `_try_basic_extraction()` | 降级策略：纯 PyMuPDF 文本提取 |

**依赖：** `langchain-pymupdf4llm`, `langchain-community`, `PyMuPDF(fitz)`, `utils.model_factory`

---

### 3.5 `message_transformer.py` — 消息转换器

**职责：** 将多模态 HumanMessage 转换为纯文本格式（核心编排层）

```
HumanMessage(content=[...])  →  遍历 parts
  ├─ text       → text_parts
  ├─ image_url  → image_analyzer.analyze() (+ cache) → image_descriptions
  ├─ file(pdf)  → pdf_analyzer.analyze() (+ cache)    → document_contents
  └─ file(other) → 元信息占位符                        → document_contents
                               ↓
  ┌─ new_msg.content = 用户可见文字（纯净）
  │
  └─ new_msg.additional_kwargs["model_context"] = 完整分析报告（模型专用）
```

| 函数 | 说明 |
|------|------|
| `transform_multimodal_message(message)` | 公共主入口 |
| `_process_part()` | 单个 part 分发处理 |
| `_handle_image_part()` | 图片块：缓存查询 → Vision 分析 → 缓存写入 |
| `_handle_file_part()` | 文件块：类型判断 → PDF解析 或 兜底 |
| `_build_visible_text()` | 组装前端可见的 content |
| `_build_model_context()` | 组装模型专用的上下文 |
| `_assemble_message()` | 组装最终 HumanMessage 对象 |

**依赖：** `cache`, `file_utils`, `image_analyzer`, `pdf_analyzer`

---

### 3.6 `middleware.py` — Agent 中间件

**职责：** LangChain Agent 的 before_model / after_model 钩子

```
用户消息进入 → [before_model] check_message_flow()
  │
  ├─ 纯文本消息 → 直接传递 ✅
  │
  └─ 多模态消息?
      │
      ├─ use_multimodal_mode=ON  → _call_doubao_multimodal() → goto end
      │                              （豆包原生处理图片/PDF）
      │
      └─ use_multimodal_mode=OFF → transform_multimodal_message() → 继续传递
                                      （Vision+文本降级）

模型响应返回 → [after_model] log_response() → 打印摘要日志
```

| 装饰器/函数 | 类型 | 说明 |
|-------------|------|------|
| `@before_model check_message_flow()` | middleware | 多模态检测与路由核心 |
| `@after_model log_response()` | middleware | 响应日志记录 |
| `_call_doubao_multimodal()` | internal | 豆包原生多模态调用 |
| `extract_attachment_metadata()` | utility | 轻量级元信息提取 |

**依赖：** `message_transformer`, `langchain.agents.middleware`

---

### 3.7 `hatch_agent.py` — 入口

**职责：** 创建并导出 Agent 实例（仅 ~30 行代码）

```python
agent = create_agent(
    model=GetModelByVendor().generate_model_client(),
    middleware=[check_message_flow, log_response],
    tools=[get_weather_tool],
    system_prompt="",
)
```

---

## 4. 数据流架构

```
┌─────────────────────────────────────────────────────────────┐
│                       前端 (Browser)                         │
│  上传图片/PDF → base64 data URL → HTTP POST                  │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              check_message_flow (before_model)               │
│                                                              │
│  ┌─────────────────┐    ┌────────────────────┐               │
│  │ use_multi=ON    │    │ use_multi=OFF      │               │
│  │                 │    │                     │               │
│  │ doubao 原生处理  │    │ transform_          │               │
│  │ → goto __end__  │    │ multimodal_message  │               │
│  └────────┬────────┘    └────────┬────────────┘               │
│           │                      │                            │
└───────────┼──────────────────────┼────────────────────────────┘
            │                      │
            ▼                      ▼
┌─────────────────────┐  ┌──────────────────────────────────┐
│  doubao 多模态模型    │  │      transform_multimodal        │
│  (原生理解图片/PDF)  │  │                                   │
│                     │  │  image_url ──→ image_analyzer     │
│  返回最终回答文字    │  │    ↓ (带 LRU 缓存)               │
│                     │  │  Vision 模型分析                  │
└─────────────────────┘  │                                   │
                          │  file(pdf) ──→ pdf_analyzer       │
                          │    ↓ (带 LRU 缓存)               │
                          │  PyMuPDF4LLM + Vision            │
                          │                                   │
                          │  输出:                             │
                          │  ├─ content = 纯文本(前端可见)     │
                          │  └─ kwargs.model_context = 分析报告│
                          └──────────────┬────────────────────┘
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────┐
│                  deepseek / 其他文本大模型                     │
│           （只看到纯文本 content，无需多模态能力）              │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              log_response (after_model)                      │
│                    打印响应摘要                               │
└─────────────────────────────────────────────────────────────┘
```

---

## 5. 缓存机制详解

### 5.1 工作原理

```
第1次上传 photo.jpg:
  compute_content_hash(base64) → "a3f2b8c1..." → 未命中
  → analyze_image() → Vision 模型调用 → 得到描述
  → put_image_cache("a3f2b8c1...", 描述) → 写入缓存

第2次上传同一张 photo.jpg (或改名 my_photo.png):
  compute_content_hash(base64) → "a3f2b8c1..." → 命中!
  → get_image_cached("a3f2b8c1...") → 直接复用描述
  → 跳过 Vision 模型调用 ✅
```

### 5.2 缓存隔离

| 缓存实例 | 存储内容 | 最大容量 | 淘汰策略 |
|---------|---------|----------|---------|
| `_image_cache` | Vision 模型图片分析结果 | 128 条 | LRU |
| `_pdf_cache` | PDF 文档解析结果 | 128 条 | LRU |

两种缓存完全独立，互不影响。

### 5.3 生命周期

- **作用域**: 进程内全局（Python 模块级变量）
- **持久化**: 无（服务重启自动清空）
- **手动管理**: `clear_all_caches()` / `get_cache_stats()`

---

## 6. 配置项

### 6.1 环境变量

| 变量名 | 必需 | 用途 | 默认值 |
|--------|------|------|--------|
| `DOUBAO_API_KEY` | 推荐 | 火山引擎豆包 API 密钥 | - |
| `DOUBAO_MODEL_NAME` | 否 | 豆包模型名称 | `doubao-seed-2-0-lite-260215` |
| `DOUBAO_BASE_URL` | 否 | 豆包 API 地址 | `https://ark.cn-beijing.volces.com/api/v3` |
| `OPENAI_API_KEY` | 推荐 | OpenAI API 密钥 | - |
| `OPENAI_BASE_URL` | 否 | OpenAI API 地址 | OpenAI 官方 |

### 6.2 运行时配置

通过 LangChain `config.configurable` 传入：

| 键名 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `use_multimodal_model` | bool | False | 是否启用豆包原生多模态模式 |

---

## 7. 依赖清单

```
# 核心运行时依赖
langchain>=0.3
langchain-core>=0.3
langchain-openai>=0.2
langchain-community>=0.3
langchain-pymupdf4llm>=0.1
PyMuPDF>=1.24
python-dotenv>=1.0

# 外部项目依赖（ai-testing-agent 内部）
utils.model_factory  # 模型工厂模块
```

安装命令：
```bash
pip install langchain langchain-examples langchain-openai langchain-community \
            langchain-pymupdf4llm pymupdf python-dotenv
```

---

## 8. 扩展指南

### 8.1 添加新的 Vision 模型提供商

在 `image_analyzer.py` 中添加新函数并在 providers 列表中注册：

```python
def _analyze_with_new_provider(image_path: str, prompt: str) -> Optional[str]:
    """新的 Vision 模型实现。"""
    # ... 构建客户端和消息 ...
    response = model.invoke([HumanMessage(content=...)])
    return response.content

# 在 analyze_image() 的 providers 列表中追加：
providers = [
    ("doubao", _analyze_with_doubao),
    ("openai", _analyze_with_openai),
    ("new_provider", _analyze_with_new_provider),  # ← 新增
]
```

### 8.2 支持新的文件类型

在 `message_transformer.py` 的 `_handle_file_part()` 中添加分支：

```python
elif is_docx and source_data:
    # Word 文档处理逻辑
    doc_content = analyze_docx(docx_local_path)
    document_contents.append(doc_content)
```

### 8.3 自定义缓存策略

继承或替换 `cache.py` 中的 `AttachmentCache` 类：

```python
from examples.cache import AttachmentCache, lru_put, lru_get
from collections import OrderedDict
import time


class TTLAttachmentCache(AttachmentCache):
    """带过期时间的缓存扩展。"""

    def __init__(self, ttl_seconds: int = 3600):
        super().__init__()
        self._ttl = ttl_seconds
        self._timestamps: dict[str, float] = {}

    def get(self, cache_type: str, key: str):
        # 检查是否过期 ...
        return super().get(cache_type, key)
```

---

## 9. 调试技巧

### 9.1 查看缓存状态

```python
from examples.cache import get_cache_stats

print(get_cache_stats())
# {'image_cache_size': 3, 'pdf_cache_size': 1, 'max_cache_size': 128}
```

### 9.2 清空缓存

```python
from examples.cache import clear_all_caches

clear_all_caches()
```

### 9.3 手动测试单个模块

```python
# 测试图片分析
from examples.image_analyzer import analyze_image

result = analyze_image("/path/to/photo.png")
print(result)

# 测试 PDF 解析
from examples.pdf_analyzer import analyze_pdf

result = analyze_pdf("/path/to/doc.pdf")
print(result)

# 测试消息转换
from langchain_core.messages import HumanMessage
from examples.message_transformer import transform_multimodal_message

msg = HumanMessage(content=[{"type": "text", "text": "看看这张图"},
                            {"type": "image_url", ...}])
new_msg = transform_multimodal_message(msg)
print(new_msg.content)  # 可见内容
print(new_msg.additional_kwargs["model_context"])  # 模型专用
```

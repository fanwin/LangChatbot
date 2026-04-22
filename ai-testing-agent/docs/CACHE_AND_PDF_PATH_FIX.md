# 附件缓存与 PDF 路径识别问题修复报告

> **日期**: 2026-04-22  
> **影响范围**: `ai-testing-agent/src/core/` 缓存模块 + 消息转换模块 + 中间件  
> **修复文件**: `cache.py`, `message_transformer.py`, `middleware.py`, `pdf_analyzer.py`  

---

## 一、问题描述

### 问题 1：本地 PDF 文件路径被当作普通文本处理

**现象**: 用户输入包含本地 PDF 路径的指令时（如 `解析一下这个文档：D:\owin\AI\test_datas\trip.pdf`），系统没有触发 PDF 解析逻辑，而是将整条消息当作纯文本直接传给大模型。

**复现条件**:
```
1. 用户在对话框中输入: "解析一下这个文档 D:\owin\AI\test_datas\trip.pdf"
2. 系统输出: 普通文本回复（未读取PDF内容）
3. 同样的，在线PDF URL (如 https://arxiv.org/pdf/2604.08000) 也可能存在类似问题
```

**根因定位**: `middleware.py:136-150` 的纯文本检测逻辑**只检查了 PDF URL**（`extract_pdf_urls`），**遗漏了本地 PDF 路径**（`extract_pdf_paths`）。

```python
# 修复前：middleware.py
has_pdf_url = False
if isinstance(content, str):
    from src.core.message_transformer import extract_pdf_urls
    pdf_urls = extract_pdf_urls(content)
    has_pdf_url = len(pdf_urls) > 0

if is_multimodal or has_pdf_url:        # ← 缺少 has_pdf_path
    # ... 触发转换
```

### 问题 2：相同附件重复解析浪费 Token

**现象**: 用户在同一会话中多次发送相同附件（或同一 PDF 路径），系统每次都重新调用 Vision 模型 / PyMuPDF 解析器进行完整分析，导致：

- 响应延迟增加（每次 5~30 秒）
- 多模态模型 Token 消耗成倍增长
- 用户体验差（等待时间长）

**根因分析**: 本地 PDF 路径的缓存链路存在两个漏洞：

| 入口函数 | 是否有缓存 | Cache Key |
|----------|-----------|-----------|
| `_handle_image_part` (base64 图片) | ✅ 有 | `MD5(base64_payload)` |
| `_handle_pdf_file` (base64 PDF 上传) | ✅ 有 | `MD5(source_data)` |
| `_process_pdf_urls` (在线 PDF URL) | ⚠️ pdf_analyzer 内部有 | `MD5(url)` + `MD5(content)` |
| **`_process_pdf_paths` (本地 PDF 路径)** | ❌ **无** | — |
| **`pdf_analyzer.analyze_pdf()` (本地文件)** | ❌ **无内部缓存** | — |

---

## 二、解决方案总览

| # | 修复项 | 文件 | 改动量 |
|---|--------|------|--------|
| 1 | middleware 补充本地 PDF 路径检测 | `middleware.py` | +6 行 |
| 2 | message_transformer 为本地 PDF 路径添加缓存 | `message_transformer.py` | +20 行 |
| 3 | pdf_analyzer 内部添加安全网缓存 | `pdf_analyzer.py` | +25 行 |
| 4 | cache.py 升级为 diskcache 持久化 + TTL | `cache.py` | 完全重写 (~340行) |
| 5 | 新增 `compute_file_hash()` 函数 | `cache.py` | +25 行 |
| 6 | 创建 `requirements.txt` | `requirements.txt` | 新文件 |
| 7 | 创建 `.gitignore` | `.gitignore` | 新文件 |

---

## 三、详细修改说明

### 3.1 修复 1：middleware.py — 补充本地 PDF 路径检测

**文件**: `ai-testing-agent/src/core/middleware.py:136-150`

**变更内容**:

```python
# ====== 修复前 ======
has_pdf_url = False
if isinstance(content, str):
    from src.core.message_transformer import extract_pdf_urls
    pdf_urls = extract_pdf_urls(content)
    has_pdf_url = len(pdf_urls) > 0
    if has_pdf_url:
        print(f"\n🔗 [检测到 PDF URL] {len(pdf_urls)} 个")

if is_multimodal or has_pdf_url:
    # ...

# ====== 修复后 ======
has_pdf_url = False
has_pdf_path = False
if isinstance(content, str):
    from src.core.message_transformer import extract_pdf_urls, extract_pdf_paths
    pdf_urls = extract_pdf_urls(content)
    pdf_paths = extract_pdf_paths(content)          # ← 新增
    has_pdf_url = len(pdf_urls) > 0
    has_pdf_path = len(pdf_paths) > 0              # ← 新增
    if has_pdf_url:
        print(f"\n🔗 [检测到 PDF URL] {len(pdf_urls)} 个")
    if has_pdf_path:
        print(f"\n📂 [检测到本地 PDF 路径] {len(pdf_paths)} 个")  # ← 新增

if is_multimodal or has_pdf_url or has_pdf_path:   # ← 条件补充
    # ...
```

**影响范围**: 仅影响中间件路由判断，不改变任何下游处理逻辑。所有后续处理（transform → analyze）已具备完整能力。

---

### 3.2 修复 2：message_transformer.py — 本地 PDF 路径缓存

**文件**: `ai-testing-agent/src/core/message_transformer.py:311-381`

**核心变更**: 在 `_process_pdf_paths()` 中添加完整的「查询缓存 → 未命中则解析 → 写入缓存」流程。

```python
def _process_pdf_paths(pdf_paths, user_text, document_contents, ...) -> None:
    for path in pdf_paths:
        # ---- 基于文件元数据查询缓存（新增）----
        file_cache_key = compute_file_hash(path)           # MD5(路径|mtime_ns|size)
        cached_result = get_pdf_cached(file_cache_key)

        if cached_result is not None:
            print(f"[transformer] 📂 本地 PDF 命中缓存")
            document_contents.append(cached_result)
            continue                                        # 直接返回，跳过解析

        # ---- 未命中，执行完整解析（原有逻辑）----
        doc_content = analyze_pdf(path, user_text)

        if doc_content:
            put_pdf_cache(file_cache_key, doc_content)      # 写入缓存（新增）
            document_contents.append(doc_content)
```

**Cache Key 设计**: `compute_file_hash()` 使用文件元数据而非内容哈希：

```python
meta_string = f"{file_path}|{stat.st_mtime_ns}|{stat.st_size}"
hash = hashlib.md5(meta_string.encode()).hexdigest()
```

| 属性 | 说明 |
|------|------|
| 零性能开销 | 不读取文件内容，仅调用 `os.stat()` |
| 自动失效 | 文件被编辑/替换后 mtime 变化 → 哈希变化 → 缓存未命中 |
| 纳秒精度 | 使用 `st_mtime_ns` 避免快速连续修改时的碰撞 |

**同时更新了 import**:
```python
from src.core.cache import compute_content_hash, compute_file_hash, \
                            get_image_cached, put_image_cache, get_pdf_cached, put_pdf_cache
                                                            ^^^^^^^^^^^^^^^^
                                                         新增导入
```

---

### 3.3 修复 3：pdf_analyzer.py — 安全网缓存

**文件**: `ai-testing-agent/src/core/pdf_analyzer.py:49-89`

**目的**: 在 `analyze_pdf()` 公共接口入口处添加内部缓存，作为**安全网** —— 即使有代码绕过 transformer 层直接调用此函数，也能避免重复解析。

```python
def analyze_pdf(pdf_path, user_text="") -> Optional[str]:
    print(f"[pdf_analyzer] 📕 开始解析本地文件: {pdf_path}")

    # ---- 安全网：基于文件元数据的缓存查询（新增）----
    from src.core.cache import get_pdf_cached, put_pdf_cache, compute_file_hash

    file_cache_key = compute_file_hash(pdf_path)
    if file_cache_key:
        cached_result = get_pdf_cached(file_cache_key)
        if cached_result is not None:
            print(f"[pdf_analyzer] ✅ 缓存命中（文件元数据哈希）")
            return cached_result

    # ---- 未命中缓存，执行完整解析（原有逻辑）----
    result = _analyze_pdf_internal(pdf_path, user_text, source_type="local")

    # ---- 写入缓存（新增）----
    if result and file_cache_key and not result.startswith("📎 **文件附件"):
        put_pdf_cache(file_cache_key, result)

    return result
```

**注意**: 此处的缓存与 `message_transformer` 中的缓存是**同一套后端**，key 相同时不会重复写入（LRU 的 `put` 会更新已有条目）。

---

### 3.4 修复 4：cache.py — 架构升级为 diskcache 持久化

**文件**: `ai-testing-agent/src/core/cache.py` （完全重写）

#### 架构设计

```
┌─────────────────────────────────────────────────────┐
│                 公共接口层 (不变)                     │
│  get_image_cached() / put_image_cache()             │
│  get_pdf_cached()   / put_pdf_cache()               │
│  get_cache_stats()  / clear_all_caches()            │
│  compute_content_hash() / compute_file_hash()       │
└──────────────────┬──────────────────────────────────┘
                   │
          ┌────────▼────────┐
          │  _get_backend()  │  路由层
          └────┬───────┬────┘
     ┌─────────┐    ┌──────────────┐
     │diskcache │    │ Memory Fallback│
     │ (主力)   │    │ (降级)        │
     │ SQLite   │    │ OrderedDict   │
     │ 持久化   │    │ 内存-only      │
     │ LRU+TTL  │    │ 无TTL         │
     └─────────┘    └──────────────┘
```

#### 双模式自动降级机制

| 场景 | 行为 |
|------|------|
| `pip install diskcache` 已执行 | ✅ 使用 SQLite 磁盘持久化 |
| `diskcache` 未安装 | ⚠️ 自动回退到内存模式（功能正常但无持久化） |
| 磁盘目录不可写 | ⚠️ 自动回退到内存模式 |
| 进程重启 | ✅ diskcache 模式下缓存保留；内存模式清空 |

#### 新增配置常量

```python
MAX_CACHE_SIZE = 512          # 原 128 → 512（磁盘空间便宜）
DEFAULT_TTL_SECONDS = 86400  # 默认 24 小时过期（原无过期）
_DEFAULT_CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", ".cache", "attachment_cache")
                             # 可通过环境变量 ATTACHMENT_CACHE_DIR 覆盖
```

#### 新增 Hash 函数

**`compute_file_hash(file_path)`**: 基于文件元数据计算哈希（见 3.2 节详细说明）。

#### 后端抽象类设计

```python
class _CacheBackend:           # 统一接口
    def get(self, key) -> Optional[str]: ...
    def put(self, key, value) -> None: ...
    def clear(self) -> None: ...
    def size(self) -> int: ...

class _DiskCacheBackend(_CacheBackend):   # 主力：diskcache 封装
class _MemoryFallbackBackend(_CacheBackend):  # 回退：OrderedDict
```

Key 格式统一使用前缀区分类型：`"image:{hash}"` 和 `"pdf:{hash}"`。

#### 进程退出清理

使用 `atexit` 注册清理钩子，确保 diskcache 连接正确关闭：
```python
import atexit
atexit.register(_cleanup)  # _backend.close()
```

---

## 四、Cache Key 策略全景图

| 数据来源 | Cache Key 生成方式 | 失效条件 | 所在位置 |
|----------|-------------------|----------|----------|
| Base64 图片上传 | `MD5(base64 payload)` | base64 数据不同 | `message_transformer._handle_image_part` |
| Base64 PDF 上传 | `MD5(source_data)` | base64 数据不同 | `message_transformer._handle_pdf_file` |
| 在线 PDF URL | `MD5(url)` + `MD5(pdf_bytes)` | URL 变或内容变 | `pdf_analyzer.analyze_pdf_from_url` |
| **本地 PDF 路径** | `MD5(路径\|mtime_ns\|size)` | **文件被修改/替换** | `message_transformer._process_pdf_paths` <br> `pdf_analyzer.analyze_pdf` (安全网) |

---

## 五、生产环境适配指南

### 5.1 部署清单

```bash
# 1. 安装依赖
cd ai-testing-agent
pip install -r requirements.txt

# 2. （可选）自定义缓存存储目录
export ATTACHMENT_CACHE_DIR=/data/cache/attachment_cache

# 3. 启动服务
python main.py
```

启动日志应显示:
```
[cache] ✅ diskcache 初始化成功 | 目录=/path/to/.cache/attachment_cache | 上限=512条 | TTL=86400s
```

如果看到:
```
[cache] ⚠️ diskcache 未安装 (pip install diskcache)，将使用内存回退模式
```
→ 请确认 `pip install diskcache` 执行成功。

### 5.2 运维操作

```python
# 查看缓存状态
from src.core.cache import get_cache_stats
stats = get_cache_stats()
# {'backend': 'diskcache', 'image_cache_size': 15, 'pdf_cache_size': 8, 'total_size': 23, ...}

# 手动清空缓存
from src.core.cache import clear_all_caches
clear_all_caches()

# 或通过类接口
from src.core.cache import AttachmentCache
cache = AttachmentCache()
cache.clear()           # 清空全部
cache.clear("image")    # 只清图片缓存
print(cache.stats())    # 查看统计
```

### 5.3 监控建议

| 监控项 | 方法 | 告警阈值 |
|--------|------|---------|
| 缓存命中率 | 日志统计 `[cache] 📁 文件元数据哈希...` 后续是否有 `命中缓存` | 命中率 < 60% |
| 缓存大小 | `get_cache_stats()['total_size']` | 接近 MAX_CACHE_SIZE |
| 磁盘空间 | `.cache/attachment_cache/` 目录大小 | > 500MB |
| 解析耗时 | 日志时间戳差值（首次 vs 缓存命中） | 首次 > 30s |

### 5.4 未来升级路径

当遇到以下瓶颈时可考虑升级缓存后端：

| 触发条件 | 推荐方案 | 改动量 |
|----------|---------|--------|
| 多台机器需共享缓存 | Redis | 替换 `_DiskCacheBackend` 实现 |
| 缓存 > 10GB | PostgreSQL | 同上 |
| 需要跨语言访问 | PostgreSQL / Redis | 同上 |
| 当前规模 | **diskcache（已实现）** | 无需改动 |

---

## 六、测试验证用例

### 6.1 本地 PDF 路径识别测试

```
输入: "帮我分析一下 D:\owin\AI\test_datas\trip.pdf 的内容"
期望:
  - 日志出现: 📂 [检测到本地 PDF 路径] 1 个
  - 日志出现: [transformer] 📂 解析本地 PDF: D:\...\trip.pdf
  - 返回内容包含 PDF 分析结果（非"无法理解"类回复）
```

### 6.2 在线 PDF URL 识别测试

```
输入: "解析一下这篇论文 https://arxiv.org/pdf/2604.08000"
期望:
  - 日志出现: 🔗 [检测到 PDF URL] 1 个
  - 日志出现: [transformer] 🌐 解析在线 PDF
  - 返回内容包含论文摘要/分析
```

### 6.3 缓存命中测试

```
第 1 次: 发送 "解析 D:\owin\AI\test_datas\trip.pdf"
  → 日志: [pdf_analyzer] 📕 开始解析本地文件...
  → 日志: [cache] 📁 文件元数据哈希: a3f8b2c1d4e5...
  → 日志: [transformer] 📂 本地 PDF 已缓存 (hash=a3f8b2c1...)

第 2 次: 再次发送相同消息
  → 日志: [cache] 📁 文件元数据哈希: a3f8b2c1d4e5...  (相同!)
  → 日志: [transformer] 📂 本地 PDF 命中缓存 (hash=a3f8b2c1...)  ← 关键！
  → 不应再出现 [pdf_analyzer] 📕 开始解析本地文件

重启服务后再发送:
  → diskcache 模式: 仍命中缓存 ✅
  → 内存回退模式: 重新解析（预期行为）
```

### 6.4 文件修改后缓存失效测试

```
步骤 1: 发送 trip.pdf → 缓存命中 hash=A
步骤 2: 用编辑器修改 trip.pdf（改几个字）→ 保存
步骤 3: 再次发送 trip.pdf
  → 日志: [cache] 文件元数据哈希: b7x9y2w3...  (不同! 因为 mtime 变了)
  → 重新解析 ✅
```

---

## 七、文件变更清单

| 文件 | 操作 | 行数变化 | 说明 |
|------|------|---------|------|
| `core/cache.py` | 重写 | ~216行 → ~340行 | diskcache 双模式架构 |
| `core/message_transformer.py` | 修改 | ~591行 → ~616行 | import 更新 + 缓存逻辑 |
| `core/middleware.py` | 修改 | +6行 | 补充 `extract_pdf_paths` 检测 |
| `core/pdf_analyzer.py` | 修改 | +25行 | `analyze_pdf` 内部安全网缓存 |
| `requirements.txt` | **新建** | ~22行 | 项目依赖声明 |
| `.gitignore` | **新建** | ~24行 | 排除缓存/临时/敏感文件 |

---

## 八、风险与注意事项

| 风险点 | 影响 | 应对措施 |
|--------|------|---------|
| `diskcache` 未安装 | 回退到内存模式，无持久化 | `pip install diskcache`；启动日志明确提示模式 |
| 磁盘权限不足 | 回退到内存模式 | 确保 `.cache/` 目录可写；可通过 `ATTACHMENT_CACHE_DIR` 自定义 |
| 缓存目录被删除 | 所有缓存丢失，重新解析 | 属于可接受行为；首次解析后会重建 |
| 文件 mtime 未变化但内容变了 | 极端场景：同秒内替换文件且大小不变 | 概率极低；如遇问题可手动 `clear_all_caches()` |
| 并发写入同一 key | 可能短暂重复解析 | diskcache 天然线程安全；最终一致性保证 |

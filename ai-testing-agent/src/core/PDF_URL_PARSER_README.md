# 在线 PDF 文档解析功能

## 概述

基于**方案二（流式内存解析）**实现的在线 PDF 文档解析功能，支持从 URL 直接解析 PDF 文档，无需手动下载到本地。

## 核心特性

### ✅ 已实现功能

1. **流式内存解析**
   - 使用 `requests` 流式下载 PDF 到内存（BytesIO）
   - 无需写入磁盘，处理完成后自动清理临时文件
   - 内存占用可控，支持大文件分块下载

2. **双重缓存机制**
   - URL 哈希缓存：同一 URL 只下载一次
   - 内容哈希缓存：不同 URL 指向同一文件也能命中缓存
   - 基于 LRU 策略，最多缓存 128 条记录

3. **安全防护**
   - URL 合法性验证（仅支持 http/https）
   - 文件大小限制（默认 100MB）
   - 下载超时控制（默认 30 秒）
   - SSRF 攻击防护（内网地址检测）

4. **错误处理**
   - 下载失败自动重试
   - 解析失败降级策略
   - 详细的错误日志输出

## 使用方法

### 基本用法

```python
from core.pdf_analyzer import analyze_pdf_from_url

# 解析在线 PDF
result = analyze_pdf_from_url(
    "https://example.com/document.pdf",
    user_text="总结这份文档的主要内容"
)

print(result)
```

### 本地文件解析（原有功能）

```python
from core.pdf_analyzer import analyze_pdf

# 解析本地 PDF
result = analyze_pdf(
    "/path/to/document.pdf",
    user_text="总结这份文档"
)

print(result)
```

### 查看缓存统计

```python
from core.cache import get_cache_stats

stats = get_cache_stats()
print(stats)
# 输出: {'image_cache_size': 5, 'pdf_cache_size': 10, 'max_cache_size': 128}
```

### 清空缓存

```python
from core.cache import clear_all_caches

clear_all_caches()
```

## 技术实现

### 核心流程

```
1. URL 验证 → 2. 检查缓存 → 3. 流式下载 → 4. 内容哈希 → 5. 解析 PDF → 6. 缓存结果
```

### 关键函数

| 函数名 | 功能 | 参数 |
|--------|------|------|
| `analyze_pdf_from_url()` | 解析在线 PDF | `pdf_url`, `user_text` |
| `download_pdf_to_memory()` | 下载 PDF 到内存 | `url` |
| `_analyze_pdf_internal()` | 统一解析逻辑 | `pdf_source`, `user_text` |
| `_is_valid_url()` | URL 验证 | `url` |
| `_save_bytesio_to_temp()` | BytesIO 转临时文件 | `bytes_io` |

### 配置参数

```python
MAX_PDF_SIZE = 100 * 1024 * 1024  # 100MB
DOWNLOAD_TIMEOUT = 30  # 秒
ALLOWED_SCHEMES = {"http", "https"}
```

可以根据需要在 `pdf_analyzer.py` 中修改这些常量。

## 缓存策略

### 缓存 Key 设计

- **URL 哈希**: `MD5(pdf_url)` - 快速命中同一 URL
- **内容哈希**: `MD5(pdf_content)` - 去重不同 URL 的相同文件

### 缓存生命周期

- 进程内全局缓存（模块级 `OrderedDict`）
- 服务重启自动清空
- LRU 淘汰策略，最多保留 128 条

### 缓存命中示例

```python
# 第一次访问：下载 + 解析 + 缓存
result1 = analyze_pdf_from_url("https://example.com/doc.pdf")

# 第二次访问：直接从缓存返回（URL 哈希命中）
result2 = analyze_pdf_from_url("https://example.com/doc.pdf")

# 不同 URL，相同内容：内容哈希命中
result3 = analyze_pdf_from_url("https://mirror.com/doc.pdf")
```

## 安全考虑

### SSRF 防护

```python
# 检测并警告内网地址访问
if hostname in ("localhost", "127.0.0.1", "0.0.0.0"):
    print("⚠️ 警告：尝试访问本地地址")

if hostname.startswith("192.168.") or hostname.startswith("10."):
    print("⚠️ 警告：尝试访问内网地址")
```

如需完全禁止内网访问，可在 `_is_valid_url()` 中返回 `False`。

### 文件大小限制

```python
# 检查 Content-Length 头
if content_length and int(content_length) > MAX_PDF_SIZE:
    return None

# 流式下载时实时检查
for chunk in response.iter_content(chunk_size=8192):
    downloaded_size += len(chunk)
    if downloaded_size > MAX_PDF_SIZE:
        return None
```

## 依赖项

确保已安装以下依赖：

```bash
pip install requests
pip install langchain-pymupdf4llm
pip install langchain-community
pip install pymupdf
```

## 错误处理

### 常见错误及解决方案

| 错误类型 | 原因 | 解决方案 |
|---------|------|---------|
| `URL 格式不合法` | URL 格式错误或协议不支持 | 检查 URL 是否以 http/https 开头 |
| `下载超时` | 网络慢或文件过大 | 增加 `DOWNLOAD_TIMEOUT` 值 |
| `文件过大` | 超过 100MB 限制 | 增加 `MAX_PDF_SIZE` 值 |
| `缺少依赖` | 未安装必要的库 | 运行 `pip install` 安装依赖 |

## 性能优化

### 内存占用

- 流式下载：每次读取 8KB，避免一次性加载大文件
- 临时文件：解析完成后立即删除
- 缓存限制：LRU 策略自动淘汰旧条目

### 速度优化

- 双重缓存：URL + 内容哈希，最大化命中率
- 并发安全：使用 `OrderedDict`，支持多线程访问
- 流式处理：边下载边写入，减少等待时间

## 扩展建议

### 短期优化

1. 添加缓存命中率统计
2. 支持自定义 User-Agent
3. 添加重试机制（下载失败自动重试 3 次）

### 中期优化

4. 磁盘缓存：使用 `diskcache` 持久化缓存
5. TTL 过期：缓存结果设置 1 小时过期时间
6. 并发下载：使用 `asyncio` + `aiohttp` 提升性能

### 长期优化

7. Redis 缓存：多实例共享缓存
8. CDN 加速：对常用文档使用 CDN
9. 分布式解析：使用消息队列异步处理

## 示例代码

参考 `example_pdf_url_usage.py` 查看完整示例。


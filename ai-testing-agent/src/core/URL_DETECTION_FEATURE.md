# 纯文本中 PDF URL 自动识别功能

## 问题描述

之前的实现中，当用户在纯文本中输入包含 PDF URL 的消息时（例如："解析一下这个在线文档 https://example.com/doc.pdf"），系统会将其判断为普通文本，不会触发 PDF 解析逻辑。

## 解决方案

在 `message_transformer.py` 和 `middleware.py` 中添加了 PDF URL 自动识别和解析功能。

### 核心改动

#### 1. message_transformer.py

**新增功能：**

- `extract_pdf_urls(text)` - 使用正则表达式提取文本中的 PDF URL
- `_handle_plain_text_with_urls()` - 处理包含 PDF URL 的纯文本消息
- `_process_pdf_urls()` - 批量处理 PDF URL 列表

**正则表达式：**

```python
PDF_URL_PATTERN = re.compile(
    r'https?://[^\s<>"{}|\\^`\[\]]+\.pdf(?:\?[^\s<>"{}|\\^`\[\]]*)?',
    re.IGNORECASE
)
```

匹配规则：
- 支持 http/https 协议
- URL 必须以 `.pdf` 结尾（忽略大小写）
- 支持查询参数（如 `?page=1`）
- 自动去重

**处理流程：**

```
纯文本输入
    ↓
提取 PDF URL
    ↓
调用 analyze_pdf_from_url()
    ↓
将解析结果注入到消息中
    ↓
返回转换后的消息
```

#### 2. middleware.py

**新增检测逻辑：**

```python
# 检查纯文本中是否包含 PDF URL
has_pdf_url = False
if isinstance(content, str):
    from src.core.message_transformer import extract_pdf_urls
    pdf_urls = extract_pdf_urls(content)
    has_pdf_url = len(pdf_urls) > 0
    if has_pdf_url:
        print(f"\n🔗 [检测到 PDF URL] {len(pdf_urls)} 个")

if is_multimodal or has_pdf_url:
    # 触发转换逻辑
    _handle_multimodal_mode_off(state, last_msg)
```

## 使用示例

### 示例 1：单个 PDF URL

**用户输入：**
```
解析一下这个在线文档 https://www.who.int/docs/default-source/coronaviruse/situation-reports/20200121-sitrep-1-2019-ncov.pdf
```

**系统处理：**
1. `middleware.py` 检测到 PDF URL
2. 调用 `transform_multimodal_message()`
3. `extract_pdf_urls()` 提取 URL
4. `analyze_pdf_from_url()` 下载并解析 PDF
5. 将解析结果注入到消息的 `<!-- __HATCH_AGENT_INTERNAL_START__ -->` 标记中
6. 大模型收到完整的文档内容

**输出结构：**
```
解析一下这个在线文档 https://www.who.int/docs/...pdf
<!-- __HATCH_AGENT_INTERNAL_START__ -->
📕 **PDF 文档分析报告**
文件名：20200121-sitrep-1-2019-ncov.pdf
总页数：5 页

--- **第 1 页** ---
[PDF 内容...]
<!-- __HATCH_AGENT_INTERNAL_END__ -->
```

### 示例 2：多个 PDF URL

**用户输入：**
```
对比这两份报告：
https://example.com/report1.pdf
https://example.com/report2.pdf
```

**系统处理：**
- 自动识别并解析两个 PDF
- 将两份文档的内容都注入到消息中
- 大模型可以同时访问两份文档进行对比分析

### 示例 3：混合内容

**用户输入：**
```
根据这份报告 https://example.com/data.pdf 中的数据，
结合我上传的图片，给出分析结论。
```

**系统处理：**
- 同时处理 PDF URL 和上传的图片
- 两者的分析结果都会注入到消息中

## 技术细节

### 缓存机制

PDF URL 解析使用双重缓存：

1. **URL 哈希缓存** - `MD5(url)`
   - 同一 URL 只下载一次
   
2. **内容哈希缓存** - `MD5(pdf_content)`
   - 不同 URL 指向同一文件也能命中缓存

### 安全措施

- URL 验证（仅支持 http/https）
- 文件大小限制（100MB）
- 下载超时控制（30秒）
- SSRF 防护（内网地址检测）

### 错误处理

如果 PDF 下载或解析失败：
- 记录错误日志
- 不影响其他 URL 的处理
- 返回原始消息（不中断用户体验）

## 日志输出

成功解析时的日志：

```
[transformer] 🔗 检测到纯文本中的 PDF URL: 1 个
[pdf_analyzer] 🌐 开始解析在线 PDF: https://example.com/doc.pdf
[pdf_analyzer] 📥 开始下载: https://example.com/doc.pdf
[pdf_analyzer] ✅ 下载完成: 245.67KB
[pdf_analyzer] ✅ 缓存命中（内容哈希）
[transformer] ✅ 在线 PDF 解析成功: doc.pdf
[transformer] ✅ 纯文本 URL 转换完成 | 模型上下文: 15234 字符
```

## 兼容性

- ✅ 向后兼容：不影响现有的多模态附件处理
- ✅ 支持多个 URL：自动去重并批量处理
- ✅ 支持查询参数：如 `doc.pdf?version=2`
- ✅ 大小写不敏感：`.PDF` 和 `.pdf` 都能识别

## 测试建议

1. 测试单个 PDF URL
2. 测试多个 PDF URL
3. 测试带查询参数的 URL
4. 测试无效 URL（应该优雅降级）
5. 测试超大文件（应该被拒绝）
6. 测试缓存命中（第二次访问应该很快）


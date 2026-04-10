"""
在线 PDF 解析使用示例

演示如何使用 analyze_pdf_from_url() 函数解析在线 PDF 文档。
"""

from pdf_analyzer import analyze_pdf_from_url, analyze_pdf
from cache import get_cache_stats


def example_online_pdf():
    """示例：解析在线 PDF 文档"""
    
    # 示例 1：解析公开的 PDF 文档
    pdf_url = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"
    
    print("=" * 60)
    print("示例 1：首次解析在线 PDF")
    print("=" * 60)
    
    result = analyze_pdf_from_url(pdf_url, user_text="这份文档的主要内容是什么？")
    
    if result:
        print("\n解析结果：")
        print(result)
    else:
        print("解析失败")
    
    # 查看缓存统计
    print("\n" + "=" * 60)
    print("缓存统计：")
    print(get_cache_stats())
    
    # 示例 2：再次解析同一个 URL（应该命中缓存）
    print("\n" + "=" * 60)
    print("示例 2：再次解析同一 URL（测试缓存）")
    print("=" * 60)
    
    result2 = analyze_pdf_from_url(pdf_url, user_text="总结一下")
    
    if result2:
        print("\n✅ 缓存命中，立即返回结果")
    
    # 查看缓存统计
    print("\n缓存统计：")
    print(get_cache_stats())


def example_local_pdf():
    """示例：解析本地 PDF 文档（对比）"""
    
    print("\n" + "=" * 60)
    print("示例 3：解析本地 PDF 文档")
    print("=" * 60)
    
    # 替换为你的本地 PDF 文件路径
    local_pdf_path = "/path/to/your/document.pdf"
    
    result = analyze_pdf(local_pdf_path, user_text="总结这份文档")
    
    if result:
        print("\n解析结果：")
        print(result)
    else:
        print("解析失败")


def example_error_handling():
    """示例：错误处理"""
    
    print("\n" + "=" * 60)
    print("示例 4：错误处理测试")
    print("=" * 60)
    
    # 测试无效 URL
    invalid_urls = [
        "ftp://example.com/file.pdf",  # 不支持的协议
        "not-a-url",  # 无效格式
        "https://example.com/nonexistent.pdf",  # 不存在的文件
    ]
    
    for url in invalid_urls:
        print(f"\n测试 URL: {url}")
        result = analyze_pdf_from_url(url)
        if result:
            print(f"结果: {result[:100]}...")


if __name__ == "__main__":
    # 运行示例
    example_online_pdf()
    
    # 如果有本地 PDF 文件，可以取消注释下面的行
    # example_local_pdf()
    
    # 测试错误处理
    # example_error_handling()


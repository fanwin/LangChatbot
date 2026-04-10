"""
LRU 缓存模块 - 避免同一附件重复调用 Vision 模型 / PDF 解析器

基于内容哈希（MD5）的去重机制：
  - 同一文件无论文件名如何变化，只要 base64 内容一致就命中缓存
  - 使用 OrderedDict 实现 LRU 淘汰策略，防止内存无限增长
  - 图片缓存和 PDF 缓存独立管理，互不干扰

使用方式：
    from examples.cache import get_cache_stats, clear_attachment_cache, AttachmentCache

    # 通过 AttachmentCache 类操作（推荐）
    cache = AttachmentCache()
    cache.put("image", hash_key, description)
    result = cache.get("image", hash_key)

    # 或通过便捷函数
    stats = get_cache_stats()
    clear_attachment_cache()
"""

import hashlib
import re
from collections import OrderedDict
from typing import Optional


# ============================================================
# 配置常量
# ============================================================

MAX_CACHE_SIZE = 128  # 每种类型最大缓存条目数


# ============================================================
# 全局缓存实例（模块级单例）
# ============================================================

_image_cache: "OrderedDict[str, str]" = OrderedDict()
_pdf_cache: "OrderedDict[str, str]" = OrderedDict()


# ============================================================
# 内容哈希计算
# ============================================================

def compute_content_hash(data_url: str) -> Optional[str]:
    """
    从 data URL 或纯 base64 字符串中提取 payload 并计算 MD5 哈希。

    支持两种输入格式：
      - Data URL:   data:image/png;base64,iVBORw0KGgo...
      - 纯 Base64:  iVBORw0KGgo...

    Args:
        data_url: 完整的 data URL 字符串或纯 base64 编码数据

    Returns:
        32 位 MD5 哈希字符串；解析失败返回 None
    """
    if not data_url or not data_url.strip():
        return None

    try:
        # 提取 base64 payload 部分
        if data_url.startswith("data:"):
            match = re.match(r"data:[^;]+;base64,(.+)", data_url, re.DOTALL)
            if not match:
                return None
            b64_payload = match.group(1)
        else:
            b64_payload = data_url.strip()

        # 标准化（处理 URL 安全变体：- _ → + /）
        b64_payload = b64_payload.replace("-", "+").replace("_", "/")
        padding = len(b64_payload) % 4
        if padding:
            b64_payload += "=" * (4 - padding)

        return hashlib.md5(b64_payload.encode()).hexdigest()
    except Exception as e:
        print(f"[cache] 计算内容哈希失败: {e}")
        return None


# ============================================================
# LRU 缓存操作
# ============================================================

def lru_get(cache: OrderedDict, key: str) -> Optional[str]:
    """从 LRU 缓存获取值。命中时将条目标记为最近使用。"""
    if key in cache:
        cache.move_to_end(key)
        return cache[key]
    return None


def lru_put(cache: OrderedDict, key: str, value: str, max_size: int = MAX_CACHE_SIZE) -> None:
    """
    写入 LRU 缓存。超出容量时自动淘汰最久未使用的条目。

    Args:
        cache: OrderedDict 缓存实例
        key: 缓存键（MD5 哈希值）
        value: 缓存值（分析/解析结果文本）
        max_size: 最大容量限制
    """
    if key in cache:
        cache.move_to_end(key)
        cache[key] = value
    else:
        cache[key] = value
        while len(cache) > max_size:
            evicted_key, _ = cache.popitem(last=False)
            print(f"[cache] 🗑️ 缓存已满，淘汰旧条目: {evicted_key[:12]}...")


# ============================================================
# 便捷接口函数
# ============================================================

def get_image_cached(key: str) -> Optional[str]:
    """查询图片分析缓存。"""
    return lru_get(_image_cache, key)


def put_image_cache(key: str, value: str) -> None:
    """写入图片分析缓存。"""
    lru_put(_image_cache, key, value)


def get_pdf_cached(key: str) -> Optional[str]:
    """查询 PDF 解析缓存。"""
    return lru_get(_pdf_cache, key)


def put_pdf_cache(key: str, value: str) -> None:
    """写入 PDF 解析缓存。"""
    lru_put(_pdf_cache, key, value)


def get_cache_stats() -> dict:
    """
    返回当前缓存统计信息。

    Returns:
        包含 image_cache_size、pdf_cache_size、max_cache_size 的字典
    """
    return {
        "image_cache_size": len(_image_cache),
        "pdf_cache_size": len(_pdf_cache),
        "max_cache_size": MAX_CACHE_SIZE,
    }


def clear_all_caches() -> None:
    """清空所有附件处理缓存。"""
    global _image_cache, _pdf_cache
    _image_cache.clear()
    _pdf_cache.clear()
    print("[cache] 🧹 附件处理缓存已全部清空")


# ============================================================
# 统一缓存访问类（面向对象风格，可选使用）
# ============================================================

class AttachmentCache:
    """
    统一的附件处理结果缓存管理类。

    将图片和 PDF 的缓存操作封装为统一接口，
    内部维护两个独立的 LRU 缓存。

    用法：
        cache = AttachmentCache()
        cache.put("image", hash_key, vision_result)
        result = cache.get("image", hash_key)
    """

    CACHE_TYPES = ("image", "pdf")

    def __init__(self):
        self._caches = {
            "image": _image_cache,
            "pdf": _pdf_cache,
        }

    def get(self, cache_type: str, key: str) -> Optional[str]:
        """从指定类型的缓存中获取值。"""
        if cache_type not in self._caches:
            raise ValueError(f"不支持的缓存类型: {cache_type}，可选: {self.CACHE_TYPES}")
        return lru_get(self._caches[cache_type], key)

    def put(self, cache_type: str, key: str, value: str) -> None:
        """向指定类型的缓存写入值。"""
        if cache_type not in self._caches:
            raise ValueError(f"不支持的缓存类型: {cache_type}，可选: {self.CACHE_TYPES}")
        lru_put(self._caches[cache_type], key, value)

    def stats(self) -> dict:
        """返回所有缓存的统计信息。"""
        return {
            ctype: len(c) for ctype, c in self._caches.items()
        }

    def clear(self, cache_type: Optional[str] = None) -> None:
        """清空指定类型或全部缓存。"""
        if cache_type is None:
            for c in self._caches.values():
                c.clear()
        elif cache_type in self._caches:
            self._caches[cache_type].clear()
        else:
            raise ValueError(f"不支持的缓存类型: {cache_type}，可选: {self.CACHE_TYPES}")

"""
附件处理结果缓存模块 - 避免重复调用 Vision 模型 / PDF 解析器

架构（双模式自动降级）:
  ┌─ 主力模式: diskcache（磁盘持久化）
  │   - 进程重启后缓存仍然有效
  │   - 内置 LRU 淘汰 + TTL 过期
  │   - 线程安全，支持并发读写
  │
  └─ 回退模式: 内存 OrderedDict（diskcache 不可用时自动降级）

缓存策略:
  - Base64 上传内容 → compute_content_hash() → MD5(base64 payload)
  - 本地 PDF 文件    → compute_file_hash()   → MD5(路径|mtime|size)
  - 在线 PDF URL     → pdf_analyzer 内部处理  → MD5(url) / MD5(content)

使用方式（接口完全不变）:
    from src.core.cache import get_image_cached, put_image_cache, get_pdf_cached, put_pdf_cache

    # 查询/写入
    result = get_image_cached(hash_key)
    put_pdf_cache(hash_key, doc_content)

    # 统计/管理
    stats = get_cache_stats()
    clear_all_caches()
"""

import hashlib
import os
import re
import time
from collections import OrderedDict
from typing import Optional


# ============================================================
# 配置常量
# ============================================================

MAX_CACHE_SIZE = 512          # 每种类型最大缓存条目数（原 128 → 512，磁盘空间便宜）
DEFAULT_TTL_SECONDS = 86400  # 默认过期时间：24 小时（0 = 不过期）

# 缓存存储目录（可通过环境变量 ATTACHMENT_CACHE_DIR 覆盖）
_DEFAULT_CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", ".cache", "attachment_cache")


# ============================================================
# 哈希计算函数（纯工具函数，不依赖任何后端）
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


def compute_file_hash(file_path: str) -> Optional[str]:
    """
    基于本地文件元数据计算缓存哈希（无需读取文件内容）。

    使用路径 + 修改时间 + 文件大小生成 MD5，当文件被修改或替换时自动失效。
    这比读取整个文件内容计算哈希高效得多。

    Args:
        file_path: 本地文件的绝对路径

    Returns:
        32 位 MD5 哈希字符串；文件不存在或无法访问时返回 None
    """
    if not file_path or not file_path.strip():
        return None

    try:
        if not os.path.isfile(file_path):
            print(f"[cache] compute_file_hash: 文件不存在: {file_path}")
            return None

        stat = os.stat(file_path)
        meta_string = f"{file_path}|{stat.st_mtime_ns}|{stat.st_size}"
        file_hash = hashlib.md5(meta_string.encode('utf-8')).hexdigest()

        print(f"[cache] 文件元数据哈希: {file_hash[:12]}... "
              f"(path={os.path.basename(file_path)}, size={stat.st_size}B)")
        return file_hash
    except Exception as e:
        print(f"[cache] 计算文件元数据哈希失败: {e}")
        return None


# ============================================================
# 后端抽象层 —— 统一接口屏蔽底层差异
# ============================================================

class _CacheBackend:
    """缓存后端基类，定义统一的 get/put/clear/stats 接口。"""

    def get(self, key: str) -> Optional[str]: ...       # type: ignore[empty-body]
    def put(self, key: str, value: str) -> None: ...     # type: ignore[empty-body]
    def clear(self) -> None: ...                         # type: ignore[empty-body]
    def size(self) -> int: ...                           # type: ignore[empty-body]


class _DiskCacheBackend(_CacheBackend):
    """
    基于 diskcache 的磁盘持久化缓存后端。

    特性：
      - 进程重启不丢失
      - 内置 LRU + TTL 双重淘汰
      - SQLite 存储引擎（无需额外服务）
      - 线程安全、进程安全
    """

    def __init__(self, cache_dir: str, max_size: int = MAX_CACHE_SIZE, ttl: int = DEFAULT_TTL_SECONDS):
        self._cache_dir = cache_dir
        self._max_size = max_size
        self._ttl = ttl
        self._cache = None
        self._init_cache()

    def _init_cache(self) -> None:
        """初始化 diskcache.Cache 实例，失败时降级为 None。"""
        try:
            import diskcache
            os.makedirs(self._cache_dir, exist_ok=True)
            self._cache = diskcache.Cache(
                directory=self._cache_dir,
                size_limit=self._max_size * 1024 * 1024,  # 转为字节限制
                default_ttl=self._ttl if self._ttl > 0 else None,
            )
            print(f"[cache] ✅ diskcache 初始化成功 | 目录={os.path.abspath(self._cache_dir)} | "
                  f"上限={self._max_size}条 | TTL={'∞' if self._ttl <= 0 else f'{self._ttl}s'}")
        except ImportError:
            print("[cache] ⚠️ diskcache 未安装 (pip install diskcache)，将使用内存回退模式")
            self._cache = None
        except Exception as e:
            print(f"[cache] ⚠️ diskcache 初始化失败: {e}，将使用内存回退模式")
            self._cache = None

    @property
    def available(self) -> bool:
        """检查 diskcache 是否可用。"""
        return self._cache is not None

    def get(self, key: str) -> Optional[str]:
        try:
            result = self._cache.get(key)  # type: ignore[union-attr]
            return str(result) if result is not None else None
        except Exception as e:
            print(f"[cache] diskcache.get 异常: {e}")
            return None

    def put(self, key: str, value: str) -> None:
        try:
            self._cache.set(key, value)  # type: ignore[union-attr]
        except Exception as e:
            print(f"[cache] diskcache.put 异常: {e}")

    def clear(self) -> None:
        try:
            self._cache.clear()  # type: ignore[union-attr]
            print("[cache] 🧹 diskcache 已清空")
        except Exception as e:
            print(f"[cache] diskcache.clear 异常: {e}")

    def size(self) -> int:
        try:
            return len(self._cache)  # type: ignore[union-attr]
        except Exception:
            return 0

    def close(self) -> None:
        """关闭缓存连接（进程退出前应调用以释放资源）。"""
        if self._cache is not None:
            try:
                self._cache.close()  # type: ignore[union-attr]
            except Exception:
                pass


class _MemoryFallbackBackend(_CacheBackend):
    """
    基于 OrderedDict 的内存缓存后端（原实现）。

    当 diskcache 不可用时作为降级方案。
    """

    def __init__(self, max_size: int = MAX_CACHE_SIZE):
        self._store: "OrderedDict[str, str]" = OrderedDict()
        self._max_size = max_size
        print(f"[cache] 📦 使用内存缓存模式 | 上限={self._max_size}条")

    def get(self, key: str) -> Optional[str]:
        if key in self._store:
            self._store.move_to_end(key)
            return self._store[key]
        return None

    def put(self, key: str, value: str) -> None:
        if key in self._store:
            self._store.move_to_end(key)
            self._store[key] = value
        else:
            self._store[key] = value
            while len(self._store) > self._max_size:
                evicted_key, _ = self._store.popitem(last=False)
                print(f"[cache] 内存缓存已满，淘汰: {evicted_key[:12]}...")

    def clear(self) -> None:
        self._store.clear()
        print("[cache] 🧹 内存缓存已清空")

    def size(self) -> int:
        return len(self._store)


# ============================================================
# 全局缓存实例（模块级单例，延迟初始化）
# ============================================================

_backend: Optional[_DiskCacheBackend] = None
_fallback_backends: dict[str, _MemoryFallbackBackend] = {
    "image": _MemoryFallbackBackend(),
    "pdf": _MemoryFallbackBackend(),
}


def _get_backend(cache_type: str) -> _CacheBackend:
    """
    获取指定类型的缓存后端实例。

    优先使用 diskcache（如果已初始化且可用），
    否则回退到该类型的独立内存缓存。
    """
    global _backend

    # 延迟初始化 diskcache（首次调用时才尝试加载）
    if _backend is None:
        cache_dir = os.environ.get(
            "ATTACHMENT_CACHE_DIR",
            os.path.abspath(_DEFAULT_CACHE_DIR),
        )
        _backend = _DiskCacheBackend(
            cache_dir=cache_dir,
            max_size=MAX_CACHE_SIZE,
            ttl=DEFAULT_TTL_SECONDS,
        )

    # diskcache 可用 → 所有类型共享同一个 Cache 实例（用 key 前缀区分类型）
    if _backend.available:
        return _backend  # type: ignore[return-value]

    # diskcache 不可用 → 回退到各类型的独立内存缓存
    if cache_type not in _fallback_backends:
        _fallback_backends[cache_type] = _MemoryFallbackBackend()
    return _fallback_backends[cache_type]


# ============================================================
# 公共便捷接口函数（保持与旧版 API 完全兼容）
# ============================================================

def get_image_cached(key: str) -> Optional[str]:
    """查询图片分析缓存。命中返回分析结果文本；未命中返回 None。"""
    if not key:
        return None
    return _get_backend("image").get(f"image:{key}")


def put_image_cache(key: str, value: str) -> None:
    """写入图片分析缓存。"""
    if not key or not value:
        return
    _get_backend("image").put(f"image:{key}", value)


def get_pdf_cached(key: str) -> Optional[str]:
    """查询 PDF 解析缓存。命中返回解析结果文本；未命中返回 None。"""
    if not key:
        return None
    return _get_backend("pdf").get(f"pdf:{key}")


def put_pdf_cache(key: str, value: str) -> None:
    """写入 PDF 解析缓存。"""
    if not key or not value:
        return
    _get_backend("pdf").put(f"pdf:{key}", value)


def get_cache_stats() -> dict:
    """
    返回当前缓存统计信息。

    Returns:
        包含以下字段的字典：
        - backend: 当前使用的后端 ("diskcache" 或 "memory")
        - image_cache_size: 图片缓存条目数
        - pdf_cache_size: PDF 缓存条目数
        - total_size: 总条目数
        - max_cache_size: 最大容量
        - ttl_seconds: 过期时间配置（秒）
        - cache_dir: 磁盘缓存目录（仅 diskcache 模式）
    """
    img_backend = _get_backend("image")
    pdf_backend = _get_backend("pdf")

    # 判断是否使用共享的 diskcache
    using_disk = (_backend is not None and _backend.available)
    same_backend = using_disk and (img_backend is pdf_backend)

    stats = {
        "backend": "diskcache" if using_disk else "memory",
        "image_cache_size": img_backend.size(),
        "pdf_cache_size": pdf_backend.size() if not same_backend else 0,
        "total_size": img_backend.size() if same_backend else (img_backend.size() + pdf_backend.size()),
        "max_cache_size": MAX_CACHE_SIZE,
        "ttl_seconds": DEFAULT_TTL_SECONDS,
    }

    if using_disk:
        stats["cache_dir"] = os.path.abspath(_backend._cache_dir)  # type: ignore[union-attr]

    return stats


def clear_all_caches() -> None:
    """清空所有附件处理缓存。"""
    # 清空 diskcache（如果在使用）
    if _backend is not None and _backend.available:
        _backend.clear()

    # 同时清空内存回退缓存（防止残留）
    for fallback in _fallback_backends.values():
        fallback.clear()

    print("[cache] 🧹 附件处理缓存已全部清空")


# ============================================================
# 统一缓存访问类（面向对象风格，可选使用）
# ============================================================

class AttachmentCache:
    """
    统一的附件处理结果缓存管理类。

    将图片和 PDF 的缓存操作封装为统一接口，
    自动适配 diskcache 或内存回退两种后端。

    用法：
        cache = AttachmentCache()
        cache.put("image", hash_key, vision_result)
        result = cache.get("image", hash_key)
    """

    CACHE_TYPES = ("image", "pdf")

    def __init__(self):
        pass  # 不再需要内部状态，直接委托给全局实例

    def get(self, cache_type: str, key: str) -> Optional[str]:
        """从指定类型的缓存中获取值。"""
        if cache_type not in self.CACHE_TYPES:
            raise ValueError(f"不支持的缓存类型: {cache_type}，可选: {self.CACHE_TYPES}")
        return _get_backend(cache_type).get(f"{cache_type}:{key}")

    def put(self, cache_type: str, key: str, value: str) -> None:
        """向指定类型的缓存写入值。"""
        if cache_type not in self.CACHE_TYPES:
            raise ValueError(f"不支持的缓存类型: {cache_type}，可选: {self.CACHE_TYPES}")
        _get_backend(cache_type).put(f"{cache_type}:{key}", value)

    def stats(self) -> dict:
        """返回缓存的统计信息。"""
        return get_cache_stats()

    def clear(self, cache_type: Optional[str] = None) -> None:
        """清空指定类型或全部缓存。"""
        if cache_type is None:
            clear_all_caches()
        elif cache_type in self.CACHE_TYPES:
            _get_backend(cache_type).clear()
        else:
            raise ValueError(f"不支持的缓存类型: {cache_type}，可选: {self.CACHE_TYPES}")


# ============================================================
# 清理钩子（模块卸载时关闭 diskcache 连接）
# ============================================================

import atexit

def _cleanup():
    """进程退出时释放资源。"""
    if _backend is not None:
        _backend.close()

atexit.register(_cleanup)

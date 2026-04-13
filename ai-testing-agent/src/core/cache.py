"""
LRU 缓存模块（SQLite 持久化版）- 避免同一附件重复调用 Vision 模型 / PDF 解析器

基于内容哈希（MD5）的去重机制：
  - 同一文件无论文件名如何变化，只要 base64 内容一致就命中缓存
  - 内存 LRU（热数据）+ SQLite（持久层）双层架构
  - 图片缓存和 PDF 缓存独立管理，互不干扰
  - ✅ SQLite WAL 模式持久化，支持多进程/多线程安全
  - ✅ 系统重启后自动恢复，避免重复解析浪费 token

架构：
  ┌──────────────┐     get      ┌──────────────┐
  │ 调用方        │ ──────────► │ 内存 LRU 缓存 │ ← O(1) 命中返回
  │ (transformer) │             │ (OrderedDict)│
  └──────────────┘              └──────┬───────┘
                                       │ miss
                                       ▼
                               ┌────────────────┐
                               │  SQLite (WAL)   │ ← 持久存储，支持并发
                               │ attachment_cache│   崩溃安全，原子写入
                               │    .db          │
                               └────────────────┘

使用方式（与之前完全一致）：
    from src.core.cache import get_cache_stats, clear_attachment_cache, AttachmentCache

    # 通过便捷函数
    cached = get_pdf_cached(hash_key)
    put_pdf_cache(hash_key, result_text)

    # 或通过类接口
    cache = AttachmentCache()
    cache.put("pdf", hash_key, description)
    result = cache.get("pdf", hash_key)

    stats = get_cache_stats()
    clear_attachment_cache()
"""

import hashlib
import os
import re
import sqlite3
import threading
from collections import OrderedDict
from typing import Optional


# ============================================================
# 配置常量
# ============================================================

MAX_CACHE_SIZE = 128  # 每种类型最大内存缓存条目数

# 数据库文件路径（与 cache.py 同目录下）
_DB_DIR = os.path.dirname(os.path.abspath(__file__))
_DB_FILE = os.path.join(_DB_DIR, ".attachment_cache.db")

# SQLite 连接管理
_db_lock = threading.Lock()  # 保护连接创建和初始化
_connections: dict[str, sqlite3.Connection] = {}  # 按线程缓存连接


# ============================================================
# 全局内存缓存实例（模块级单例）
# ============================================================

_image_cache: "OrderedDict[str, str]" = OrderedDict()
_pdf_cache: "OrderedDict[str, str]" = OrderedDict()


# ============================================================
# SQLite 数据库管理
# ============================================================

def _get_connection(cache_type: str) -> sqlite3.Connection:
    """
    获取当前线程的 SQLite 连接（线程本地缓存）。

    SQLite 连接不应跨线程共享，每个线程使用独立连接。
    WAL 模式下多连接可安全并发读，写操作由 SQLite 内部串行化。

    Args:
        cache_type: 'image' 或 'pdf'

    Returns:
        线程安全的 sqlite3.Connection 实例
    """
    thread_id = threading.current_thread().ident

    if thread_id in _connections:
        return _connections[thread_id]

    with _db_lock:
        # 双重检查：等待锁期间可能已被其他线程创建
        if thread_id in _connections:
            return _connections[thread_id]

        conn = sqlite3.connect(
            _DB_FILE,
            timeout=10.0,       # 写锁等待超时（秒），防止死锁
            check_same_thread=False,
        )
        # WAL 模式：读写不阻塞，崩溃自动恢复
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")  # 平衡安全和性能
        # 外键约束支持
        conn.execute("PRAGMA foreign_keys=ON")

        # 确保表存在
        _ensure_table(conn)

        _connections[thread_id] = conn
        print(f"[cache] 🗄️ SQLite 连接已创建 (thread={thread_id}, type={cache_type})")

    return conn


def _ensure_table(conn: sqlite3.Connection) -> None:
    """
    创建缓存表（如果不存在）。

    表结构：
      cache_type  - 缓存类型分区 ('image' | 'pdf')
      cache_key   - 缓存键（MD5哈希或路径键），主键
      value       - 缓存的解析结果文本
      created_at  - 创建时间戳
      accessed_at - 最后访问时间戳（用于 LRU 淘汰排序）
      size_bytes  - 值的字节大小
    """
    conn.execute("""
        CREATE TABLE IF NOT EXISTS attachment_cache (
            cache_type  TEXT NOT NULL,
            cache_key   TEXT NOT NULL,
            value       TEXT NOT NULL,
            created_at  REAL NOT NULL DEFAULT (strftime('%s', 'now')),
            accessed_at REAL NOT NULL DEFAULT (strftime('%s', 'now')),
            size_bytes INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (cache_type, cache_key)
        )
    """)
    # 加速 LRU 淘汰查询：按访问时间排序查找最旧条目
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_cache_lru
        ON attachment_cache (cache_type, accessed_at ASC)
    """)
    conn.commit()


def _close_all_connections() -> None:
    """关闭所有线程的数据库连接。"""
    with _db_lock:
        for tid, conn in list(_connections.items()):
            try:
                conn.close()
            except Exception:
                pass
        _connections.clear()


# ============================================================
# SQLite CRUD 操作
# ============================================================

def _db_get(cache_type: str, key: str) -> Optional[str]:
    """
    从 SQLite 查询缓存值。

    Args:
        cache_type: 'image' 或 'pdf'
        key: 缓存键

    Returns:
        命中的值；未命中返回 None
    """
    try:
        conn = _get_connection(cache_type)
        cursor = conn.execute(
            "SELECT value FROM attachment_cache WHERE cache_type = ? AND cache_key = ?",
            (cache_type, key),
        )
        row = cursor.fetchone()
        if row is not None:
            # 更新访问时间（用于 LRU 排序）
            conn.execute(
                "UPDATE attachment_cache SET accessed_at = strftime('%s', 'now') "
                "WHERE cache_type = ? AND cache_key = ?",
                (cache_type, key),
            )
            conn.commit()
            return row[0]
    except Exception as e:
        print(f"[cache] ⚠️ SQLite 查询失败 ({cache_type}): {e}")

    return None


def _db_put(cache_type: str, key: str, value: str) -> None:
    """
    写入或更新 SQLite 缓存（UPSERT）。

    同时执行容量淘汰：超出 MAX_CACHE_SIZE 时删除最久未访问的条目。

    Args:
        cache_type: 'image' 或 'pdf'
        key: 缓存键
        value: 缓存值
    """
    try:
        conn = _get_connection(cache_type)
        now = _db_now(conn)
        size = len(value.encode("utf-8"))

        # UPSERT
        conn.execute("""
            INSERT INTO attachment_cache (cache_type, cache_key, value, created_at, accessed_at, size_bytes)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT (cache_type, cache_key) DO UPDATE SET
                value = excluded.value,
                accessed_at = excluded.accessed_at,
                size_bytes = excluded.size_bytes
        """, (cache_type, key, value, now, now, size))

        # LRU 淘汰：保留最近访问的 MAX_CACHE_SIZE 条
        conn.execute("""
            DELETE FROM attachment_cache
            WHERE cache_type = ?
              AND cache_key NOT IN (
                  SELECT cache_key FROM attachment_cache
                  WHERE cache_type = ?
                  ORDER BY accessed_at DESC
                  LIMIT ?
              )
        """, (cache_type, cache_type, MAX_CACHE_SIZE))

        conn.commit()

    except Exception as e:
        print(f"[cache] ⚠️ SQLite 写入失败 ({cache_type}): {e}")


def _db_delete(cache_type: str, key: Optional[str] = None) -> int:
    """
    删除缓存条目。

    Args:
        cache_type: 'image' 或 'pdf'
        key: 删除指定键；None 表示清空该类型的所有条目

    Returns:
        删除的行数
    """
    try:
        conn = _get_connection(cache_type)
        if key:
            cursor = conn.execute(
                "DELETE FROM attachment_cache WHERE cache_type = ? AND cache_key = ?",
                (cache_type, key),
            )
        else:
            cursor = conn.execute(
                "DELETE FROM attachment_cache WHERE cache_type = ?",
                (cache_type,),
            )
        conn.commit()
        return cursor.rowcount
    except Exception as e:
        print(f"[cache] ⚠️ SQLite 删除失败 ({cache_type}): {e}")
        return 0


def _db_count(cache_type: str) -> int:
    """统计指定类型的缓存条目数量。"""
    try:
        conn = _get_connection(cache_type)
        cursor = conn.execute(
            "SELECT COUNT(*) FROM attachment_cache WHERE cache_type = ?",
            (cache_type,),
        )
        return cursor.fetchone()[0]
    except Exception:
        return 0


def _db_size_bytes(cache_type: str) -> int:
    """统计指定类型的缓存数据总大小（字节）。"""
    try:
        conn = _get_connection(cache_type)
        cursor = conn.execute(
            "SELECT COALESCE(SUM(size_bytes), 0) FROM attachment_cache WHERE cache_type = ?",
            (cache_type,),
        )
        return cursor.fetchone()[0]
    except Exception:
        return 0


def _db_now(conn: sqlite3.Connection) -> float:
    """获取当前时间戳（通过 SQLite 函数保证一致性）。"""
    row = conn.execute("SELECT strftime('%s', 'now')").fetchone()
    return float(row[0])


# ============================================================
# 内容哈希计算（保持不变）
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
# LRU 缓存操作（双层：内存 + SQLite）
# ============================================================

def lru_get(cache: OrderedDict, key: str, cache_type: str = "") -> Optional[str]:
    """
    双层缓存读取：内存 L1 → SQLite L2 → 回填内存。

    Args:
        cache: OrderedDict 内存 LRU 缓存实例
        key: 缓存键（MD5 哈希值或路径键）
        cache_type: 缓存类型标识（'image' 或 'pdf'），用于查 SQLite

    Returns:
        命中的缓存值；未命中返回 None
    """
    # L1: 内存缓存
    if key in cache:
        cache.move_to_end(key)
        return cache[key]

    # L2: SQLite 回填
    if cache_type:
        value = _db_get(cache_type, key)
        if value is not None:
            # 回填到内存（LRU 末尾标记为最近使用）
            cache[key] = value
            print(f"[cache] 💾 SQLite→内存回填 {cache_type}: {key[:24]}...")
            return value

    return None


def lru_put(cache: OrderedDict, key: str, value: str, max_size: int = MAX_CACHE_SIZE, cache_type: str = "") -> None:
    """
    双层缓存写入：内存 L1 + SQLite L2（write-through）。

    Args:
        cache: OrderedDict 缓存实例
        key: 缓存键
        value: 缓存值（分析/解析结果文本）
        max_size: 最大容量限制
        cache_type: 缓存类型标识，用于写 SQLite
    """
    # L1: 写入内存 LRU
    if key in cache:
        cache.move_to_end(key)
        cache[key] = value
    else:
        cache[key] = value
        while len(cache) > max_size:
            evicted_key, _ = cache.popitem(last=False)
            print(f"[cache] 🗑️ 内存缓存已满，淘汰: {evicted_key[:24]}...")

    # L2: write-through 写入 SQLite
    if cache_type:
        _db_put(cache_type, key, value)


# ============================================================
# 便捷接口函数（与之前完全一致，无需修改调用方）
# ============================================================

def get_image_cached(key: str) -> Optional[str]:
    """查询图片分析缓存（内存 → SQLite 自动回填）。"""
    return lru_get(_image_cache, key, cache_type="image")


def put_image_cache(key: str, value: str) -> None:
    """写入图片分析缓存（自动同步到 SQLite）。"""
    lru_put(_image_cache, key, value, cache_type="image")


def get_pdf_cached(key: str) -> Optional[str]:
    """查询 PDF 解析缓存（内存 → SQLite 自动回填）。"""
    return lru_get(_pdf_cache, key, cache_type="pdf")


def put_pdf_cache(key: str, value: str) -> None:
    """写入 PDF 解析缓存（自动同步到 SQLite）。"""
    lru_put(_pdf_cache, key, value, cache_type="pdf")


def get_cache_stats() -> dict:
    """
    返回当前缓存统计信息（内存 + SQLite 双层数据）。
    """
    image_count = _db_count("image")
    pdf_count = _db_count("pdf")
    image_bytes = _db_size_bytes("image")
    pdf_bytes = _db_size_bytes("pdf")

    return {
        # 内存层
        "image_memory_size": len(_image_cache),
        "pdf_memory_size": len(_pdf_cache),
        "max_memory_size": MAX_CACHE_SIZE,
        # SQLite 持久层
        "image_total_entries": image_count,
        "pdf_total_entries": pdf_count,
        "image_total_bytes": image_bytes,
        "pdf_total_bytes": pdf_bytes,
        # 存储
        "storage_engine": "SQLite (WAL mode)",
        "db_file": _DB_FILE,
    }


def clear_all_caches(clear_persistent: bool = True) -> None:
    """
    清空所有附件处理缓存。

    Args:
        clear_persistent: 是否同时清理 SQLite 中的持久化数据（默认 True）
    """
    global _image_cache, _pdf_cache
    _image_cache.clear()
    _pdf_cache.clear()

    if clear_persistent:
        deleted_img = _db_delete("image")
        deleted_pdf = _db_delete("pdf")

    print(
        f"[cache] 🧹 缓存已全部清空"
        f"（内存已清除"
        f"{f' + SQLite 清除 {deleted_img + deleted_pdf} 条' if clear_persistent else ''}）"
    )


# ============================================================
# 统一缓存访问类（面向对象风格，可选使用）
# ============================================================

class AttachmentCache:
    """
    统一的附件处理结果缓存管理类（SQLite 版）。

    将图片和 PDF 的缓存操作封装为统一接口，
    内部维护内存 LRU + SQLite 持久化的双层架构。

    用法（与之前完全一致）：
        cache = AttachmentCache()
        cache.put("image", hash_key, vision_result)
        result = cache.get("image", hash_key)

    特性：
        - 对外接口不变，底层从 JSON 升级为 SQLite
        - 支持多进程/多线程安全（WAL 模式）
        - 崩溃安全（原子事务提交）
        - LRU 淘汰在 SQLite 层自动执行
    """

    CACHE_TYPES = ("image", "pdf")

    def __init__(self):
        self._caches = {
            "image": _image_cache,
            "pdf": _pdf_cache,
        }

    def get(self, cache_type: str, key: str) -> Optional[str]:
        """从指定类型的缓存中获取值（内存 → SQLite 回填）。"""
        if cache_type not in self._caches:
            raise ValueError(f"不支持的缓存类型: {cache_type}，可选: {self.CACHE_TYPES}")
        return lru_get(self._caches[cache_type], key, cache_type=cache_type)

    def put(self, cache_type: str, key: str, value: str) -> None:
        """向指定类型的缓存写入值（自动同步到 SQLite）。"""
        if cache_type not in self._caches:
            raise ValueError(f"不支持的缓存类型: {cache_type}，可选: {self.CACHE_TYPES}")
        lru_put(self._caches[cache_type], key, value, cache_type=cache_type)

    def stats(self) -> dict:
        """返回缓存统计信息。"""
        return {
            ctype: len(c) for ctype, c in self._caches.items()
        }

    def clear(self, cache_type: Optional[str] = None, clear_persistent: bool = True) -> None:
        """
        清空指定类型或全部缓存。

        Args:
            cache_type: 要清空的缓存类型；None 表示清空全部
            clear_persistent: 是否同时清理 SQLite 中的持久化数据
        """
        if cache_type is None:
            for c in self._caches.values():
                c.clear()
            if clear_persistent:
                for ct in self.CACHE_TYPES:
                    _db_delete(ct)
        elif cache_type in self._caches:
            self._caches[cache_type].clear()
            if clear_persistent:
                _db_delete(cache_type)
        else:
            raise ValueError(f"不支持的缓存类型: {cache_type}，可选: {self.CACHE_TYPES}")

    def persist_all(self) -> None:
        """
        手动将当前内存中未持久化的条目强制写入 SQLite。
        
        注：正常流程中每次 put 已自动 write-through，
        此方法仅在特殊场景下使用（如内存被外部修改后需要同步）。
        """
        for cache_type in self.CACHE_TYPES:
            cache = self._caches[cache_type]
            for key, value in cache.items():
                _db_put(cache_type, key, value)
        print("[cache] 💾 内存缓存已手动同步至 SQLite")

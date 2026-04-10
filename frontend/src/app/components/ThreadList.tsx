"use client";
// FIXME  MC80OmFIVnBZMlhvaklQb3RvVTZOVFZIZWc9PTpmMDNhZGFhNA==

import { useEffect, useMemo, useState, useRef, useCallback, Fragment } from "react";
import { Loader2, MessageSquare, Trash2, Pin, PinOff, Pencil, MoreHorizontal, Check, SquarePen } from "lucide-react";
import { useQueryState } from "nuqs";
import { Button } from "@/components/ui/button";

import { Skeleton } from "@/components/ui/skeleton";
import { cn } from "@/lib/utils";
import type { ThreadItem } from "@/app/hooks/useThreads";
import { useThreads } from "@/app/hooks/useThreads";
import { useClient } from "@/providers/ClientProvider";

const GROUP_LABELS = {
  pinned: "📌 置顶",
  interrupted: "需要关注",
  today: "今天",
  yesterday: "昨天",
  week: "本周",
  older: "更早",
} as const;

const STATUS_COLORS: Record<ThreadItem["status"], string> = {
  idle: "bg-green-500",
  busy: "bg-blue-500",
  interrupted: "bg-orange-500",
  error: "bg-red-600",
};
//  MS80OmFIVnBZMlhvaklQb3RvVTZOVFZIZWc9PTpmMDNhZGFhNA==

function getThreadColor(status: ThreadItem["status"]): string {
  return STATUS_COLORS[status] ?? "bg-gray-400";
}

// FIXME  Mi80OmFIVnBZMlhvaklQb3RvVTZOVFZIZWc9PTpmMDNhZGFhNA==

function ErrorState({ message }: { message: string }) {
  return (
    <div className="flex flex-col items-center justify-center p-8 text-center">
      <p className="text-sm text-red-600">加载对话列表失败</p>
      <p className="mt-1 text-xs text-muted-foreground">{message}</p>
    </div>
  );
}

function LoadingState() {
  return (
    <div className="space-y-2 p-4">
      {Array.from({ length: 5 }).map((_, i) => (
        <Skeleton
          key={i}
          className="h-16 w-full"
        />
      ))}
    </div>
  );
}

function EmptyState() {
  return (
    <div className="flex flex-col items-center justify-center p-8 text-center">
      <MessageSquare className="mb-2 h-12 w-12 text-gray-300" />
      <p className="text-sm text-muted-foreground">暂无对话</p>
    </div>
  );
}

interface ThreadListProps {
  onThreadSelect: (id: string) => void;
  onMutateReady?: (mutate: () => void) => void;
  onInterruptCountChange?: (count: number) => void;
  onNewThread?: () => void;
}

export function ThreadList({
  onThreadSelect,
  onMutateReady,
  onInterruptCountChange,
  onNewThread,
}: ThreadListProps) {
  const [currentThreadId, setCurrentThreadId] = useQueryState("threadId");
  const [deletingThreadId, setDeletingThreadId] = useState<string | null>(null);
  const client = useClient();

  // Pinned threads (stored in localStorage)
  const [pinnedThreadIds, setPinnedThreadIds] = useState<string[]>(() => {
    if (typeof window === "undefined") return [];
    try {
      return JSON.parse(localStorage.getItem("pinnedThreads") || "[]");
    } catch { return []; }
  });

  // Renamed threads (stored in localStorage)
  const [renamedThreads, setRenamedThreads] = useState<Record<string, string>>(() => {
    if (typeof window === "undefined") return {};
    try {
      return JSON.parse(localStorage.getItem("renamedThreads") || "{}");
    } catch { return {}; }
  });

  // Inline rename state
  const [renamingThreadId, setRenamingThreadId] = useState<string | null>(null);
  const [renameValue, setRenameValue] = useState("");
  const renameInputRef = useRef<HTMLInputElement>(null);

  // Context menu state
  const [contextMenu, setContextMenu] = useState<{
    threadId: string;
    x: number;
    y: number;
  } | null>(null);
  const contextMenuRef = useRef<HTMLDivElement>(null);

  const threads = useThreads({
    status: undefined,
    limit: 20,
  });

  const flattened = useMemo(() => {
    return threads.data?.flat() ?? [];
  }, [threads.data]);

  const isLoadingMore =
    threads.size > 0 && threads.data?.[threads.size - 1] == null;
  const isEmpty = threads.data?.at(0)?.length === 0;
  const isReachingEnd = isEmpty || (threads.data?.at(-1)?.length ?? 0) < 20;

  // Group threads by time and status
  const grouped = useMemo(() => {
    const now = new Date();
    const groups: Record<keyof typeof GROUP_LABELS, ThreadItem[]> = {
      pinned: [],
      interrupted: [],
      today: [],
      yesterday: [],
      week: [],
      older: [],
    };

    flattened.forEach((thread) => {
      // Pinned threads go to pinned group first
      if (pinnedThreadIds.includes(thread.id)) {
        groups.pinned.push(thread);
        return;
      }

      if (thread.status === "interrupted") {
        groups.interrupted.push(thread);
        return;
      }

      const diff = now.getTime() - thread.updatedAt.getTime();
      const days = Math.floor(diff / (1000 * 60 * 60 * 24));

      if (days === 0) {
        groups.today.push(thread);
      } else if (days === 1) {
        groups.yesterday.push(thread);
      } else if (days < 7) {
        groups.week.push(thread);
      } else {
        groups.older.push(thread);
      }
    });

    return groups;
  }, [flattened, pinnedThreadIds]);

  const interruptedCount = useMemo(() => {
    return flattened.filter((t) => t.status === "interrupted").length;
  }, [flattened]);

  // Expose thread list revalidation to parent component
  // Use refs to create a stable callback that always calls the latest mutate function
  const onMutateReadyRef = useRef(onMutateReady);
  const mutateRef = useRef(threads.mutate);

  useEffect(() => {
    onMutateReadyRef.current = onMutateReady;
  }, [onMutateReady]);

  useEffect(() => {
    mutateRef.current = threads.mutate;
  }, [threads.mutate]);

  const mutateFn = useCallback(() => {
    mutateRef.current();
  }, []);

  useEffect(() => {
    onMutateReadyRef.current?.(mutateFn);
    // Only run once on mount to avoid infinite loops
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleDeleteThread = useCallback(
    async (threadId: string, e: React.MouseEvent) => {
      e.stopPropagation();

      if (!confirm("确定要删除这条对话吗？此操作无法撤销。")) {
        return;
      }

      setDeletingThreadId(threadId);
      try {
        await client.threads.delete(threadId);

        if (currentThreadId === threadId) {
          setCurrentThreadId(null);
        }

        threads.mutate();
      } catch (error) {
        console.error("Failed to delete thread:", error);
        alert("删除失败，请重试。");
      } finally {
        setDeletingThreadId(null);
      }
    },
    [client, currentThreadId, setCurrentThreadId, threads]
  );

  // Toggle pin
  const handleTogglePin = useCallback((threadId: string) => {
    setPinnedThreadIds((prev) => {
      const next = prev.includes(threadId)
        ? prev.filter((id) => id !== threadId)
        : [threadId, ...prev];
      localStorage.setItem("pinnedThreads", JSON.stringify(next));
      return next;
    });
    setContextMenu(null);
  }, []);

  // Start rename
  const handleStartRename = useCallback((threadId: string, currentTitle: string) => {
    setRenamingThreadId(threadId);
    setRenameValue(currentTitle);
    setContextMenu(null);
    setTimeout(() => renameInputRef.current?.focus(), 50);
  }, []);

  // Confirm rename
  const handleConfirmRename = useCallback((threadId: string) => {
    const trimmed = renameValue.trim();
    if (trimmed) {
      setRenamedThreads((prev) => {
        const next = { ...prev, [threadId]: trimmed };
        localStorage.setItem("renamedThreads", JSON.stringify(next));
        return next;
      });
    }
    setRenamingThreadId(null);
    setRenameValue("");
  }, [renameValue]);

  // Close context menu on outside click
  useEffect(() => {
    if (!contextMenu) return;
    const handleClick = (e: MouseEvent) => {
      if (contextMenuRef.current && !contextMenuRef.current.contains(e.target as Node)) {
        setContextMenu(null);
      }
    };
    document.addEventListener("mousedown", handleClick);
    return () => document.removeEventListener("mousedown", handleClick);
  }, [contextMenu]);

  // Notify parent of interrupt count changes
  useEffect(() => {
    onInterruptCountChange?.(interruptedCount);
  }, [interruptedCount, onInterruptCountChange]);

  return (
    <div className="absolute inset-0 flex flex-col bg-card/50">
      {/* Header with title and new thread button */}
      <div className="flex flex-shrink-0 items-center justify-between border-b border-border/60 px-4 py-3">
        {/* <h2 className="text-sm font-semibold tracking-tight">对话历史</h2> */}
        <div className="flex items-center gap-1.5">
          {onNewThread && (
            <Button
              variant="default"
              size="sm"
              onClick={onNewThread}
              className="h-7 rounded-md bg-gradient-to-r from-[#2F6868] to-[#1c3c3c] px-2.5 text-xs text-white shadow-sm transition-all hover:from-[#2F6868]/90 hover:to-[#1c3c3c]/90 hover:shadow-md"
            >
              <SquarePen className="mr-1 h-3.5 w-3.5" />
              新建对话
            </Button>
          )}
        </div>
      </div>

      <div className="h-0 flex-1 overflow-y-auto">
        {threads.error && <ErrorState message={threads.error.message} />}

        {!threads.error && !threads.data && threads.isLoading && (
          <LoadingState />
        )}

        {!threads.error && !threads.isLoading && isEmpty && <EmptyState />}

        {!threads.error && !isEmpty && (
          <div className="box-border w-full px-2 py-1">
            {(
              Object.keys(GROUP_LABELS) as Array<keyof typeof GROUP_LABELS>
            ).map((group) => {
              const groupThreads = grouped[group];
              if (groupThreads.length === 0) return null;

              return (
                <div
                  key={group}
                  className="mb-3"
                >
                  <h4 className="m-0 px-3 py-1.5 text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">
                    {GROUP_LABELS[group]}
                  </h4>
                  <div className="flex flex-col gap-0.5">
                    {groupThreads.map((thread) => {
                      const displayTitle = renamedThreads[thread.id] || thread.title;
                      const isPinned = pinnedThreadIds.includes(thread.id);
                      const isRenaming = renamingThreadId === thread.id;

                      return (
                      <div
                        key={thread.id}
                        className="group flex items-center rounded-lg transition-all duration-150 hover:bg-accent/80"
                      >
                        <div
                          role="button"
                          tabIndex={0}
                          onClick={() => onThreadSelect(thread.id)}
                          onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") onThreadSelect(thread.id); }}
                          className={cn(
                            "flex min-w-0 flex-1 cursor-pointer items-center gap-1.5 overflow-hidden px-3 py-2 text-left",
                            currentThreadId === thread.id
                              ? "bg-accent shadow-sm ring-1 ring-primary/20 rounded-lg"
                              : "bg-transparent"
                          )}
                          aria-current={currentThreadId === thread.id}
                        >
                          {isPinned && <Pin className="h-3 w-3 flex-shrink-0 text-[#2F6868]" />}
                          {isRenaming ? (
                            <div className="flex min-w-0 flex-1 items-center gap-1" onClick={(e) => e.stopPropagation()}>
                              <input
                                ref={renameInputRef}
                                type="text"
                                value={renameValue}
                                onChange={(e) => setRenameValue(e.target.value)}
                                onKeyDown={(e) => {
                                  e.stopPropagation();
                                  if (e.key === "Enter") handleConfirmRename(thread.id);
                                  if (e.key === "Escape") { setRenamingThreadId(null); setRenameValue(""); }
                                }}
                                onBlur={() => handleConfirmRename(thread.id)}
                                className="h-6 flex-1 rounded border border-primary/40 bg-background px-1.5 text-sm font-medium outline-none focus:ring-1 focus:ring-primary/30"
                              />
                            </div>
                          ) : (
                            <span className="min-w-0 flex-1 truncate text-[13px] font-medium">
                              {displayTitle}
                            </span>
                          )}
                          <div
                            className={cn(
                              "h-1.5 w-1.5 flex-shrink-0 rounded-full",
                              getThreadColor(thread.status)
                            )}
                          />
                        </div>
                        <button
                          type="button"
                          onClick={(e) => {
                            e.stopPropagation();
                            const rect = e.currentTarget.getBoundingClientRect();
                            setContextMenu({ threadId: thread.id, x: rect.left, y: rect.bottom + 4 });
                          }}
                          className={cn(
                            "flex-shrink-0 rounded p-1.5 text-muted-foreground transition-opacity hover:bg-accent hover:text-foreground",
                            contextMenu?.threadId === thread.id ? "opacity-100" : "opacity-0 group-hover:opacity-100"
                          )}
                          title="更多操作"
                        >
                          <MoreHorizontal className="h-4 w-4" />
                        </button>
                      </div>
                      );
                    })}
                  </div>
                </div>
              );
            })}

            {!isReachingEnd && (
              <div className="flex justify-center py-3">
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => threads.setSize(threads.size + 1)}
                  disabled={isLoadingMore}
                  className="h-8 text-xs text-muted-foreground"
                >
                  {isLoadingMore ? (
                    <>
                      <Loader2 className="mr-1.5 h-3.5 w-3.5 animate-spin" />
                      加载中...
                    </>
                  ) : (
                    "加载更多"
                  )}
                </Button>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Context Menu */}
      {contextMenu && (
        <div
          ref={contextMenuRef}
          className="fixed z-50 min-w-[140px] rounded-lg border border-border/60 bg-popover p-1 shadow-lg"
          style={{ left: contextMenu.x, top: contextMenu.y }}
        >
          <button
            type="button"
            className="flex w-full items-center gap-2 rounded-md px-2.5 py-1.5 text-xs text-foreground transition-colors hover:bg-accent"
            onClick={() => {
              const thread = flattened.find((t) => t.id === contextMenu.threadId);
              if (thread) handleStartRename(thread.id, renamedThreads[thread.id] || thread.title);
            }}
          >
            <Pencil className="h-3.5 w-3.5" />
            重命名
          </button>
          <button
            type="button"
            className="flex w-full items-center gap-2 rounded-md px-2.5 py-1.5 text-xs text-foreground transition-colors hover:bg-accent"
            onClick={() => handleTogglePin(contextMenu.threadId)}
          >
            {pinnedThreadIds.includes(contextMenu.threadId) ? (
              <><PinOff className="h-3.5 w-3.5" /><span>取消置顶</span></>
            ) : (
              <><Pin className="h-3.5 w-3.5" /><span>置顶</span></>
            )}
          </button>
          <div className="my-1 h-px bg-border/60" />
          <button
            type="button"
            className="flex w-full items-center gap-2 rounded-md px-2.5 py-1.5 text-xs text-destructive transition-colors hover:bg-destructive/10"
            onClick={(e) => {
              handleDeleteThread(contextMenu.threadId, e);
              setContextMenu(null);
            }}
          >
            {deletingThreadId === contextMenu.threadId ? (
              <Loader2 className="h-3.5 w-3.5 animate-spin" />
            ) : (
              <Trash2 className="h-3.5 w-3.5" />
            )}
            删除
          </button>
        </div>
      )}
    </div>
  );
}
// FIXME  My80OmFIVnBZMlhvaklQb3RvVTZOVFZIZWc9PTpmMDNhZGFhNA==

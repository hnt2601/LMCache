# SPDX-License-Identifier: Apache-2.0
"""Idle-TTL tracking per chunk, shared by the L1 and L2 eviction controllers.

A *chunk* is every object that shares ``(model_name, chunk_hash, cache_salt)``
across all kv_ranks and object groups. A chunk is only useful when all of its
stored objects are present, so idle expiry works on whole chunks.

The tracker keeps chunks in an ``OrderedDict`` ordered by last access, oldest
first. Accesses arrive in request order (prefix first), and are applied in
reverse like :class:`LRUEvictionPolicy`, so within one access the prefix head
ends up most recent. Expiry therefore always removes a child before its
parent: a parent is touched whenever any of its children is.

The tracker does not decide anything by itself and does no I/O. The eviction
controllers ask it for expired chunks and delete them through the tier's
normal delete path (L1 lock checks, L2 tombstones and pin checks).
"""

# Future
from __future__ import annotations

# Standard
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
import threading
import time

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.internal_api import (
    L1ManagerListener,
    L2AdapterListener,
)

ChunkId = tuple[str, bytes, str]


def chunk_id_of(key: ObjectKey) -> ChunkId:
    """Return the chunk a key belongs to (all kv_ranks and object groups)."""
    return (key.model_name, key.chunk_hash, key.cache_salt)


def _group_by_chunk(keys: list[ObjectKey]) -> list[tuple[ChunkId, list[ObjectKey]]]:
    """Group ``keys`` by chunk, keeping the first-seen chunk order."""
    groups: dict[ChunkId, list[ObjectKey]] = {}
    for key in keys:
        groups.setdefault(chunk_id_of(key), []).append(key)
    return list(groups.items())


@dataclass
class _ChunkEntry:
    last_access: float
    members: set[ObjectKey] = field(default_factory=set)


class IdleTTLTracker:
    """Track the last access time of every chunk of one cache tier.

    Thread safety: all methods take an internal lock. Listener callbacks
    arrive from several threads (L1 manager callers, the L2 adapter loop,
    the eviction thread).

    Args:
        clock: Monotonic time source in seconds; injectable for tests.
    """

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self._chunks: OrderedDict[ChunkId, _ChunkEntry] = OrderedDict()

    # ------------------------------------------------------------------
    # Event inputs
    # ------------------------------------------------------------------

    def on_keys_created(self, keys: list[ObjectKey]) -> None:
        """Record new objects; their chunks become the most recent."""
        self._touch(keys, create=True)

    def on_keys_touched(self, keys: list[ObjectKey]) -> None:
        """Refresh chunks that are already tracked; unknown keys are ignored."""
        self._touch(keys, create=False)

    def on_keys_removed(self, keys: list[ObjectKey]) -> None:
        """Forget objects; a chunk with no member left is dropped."""
        if not keys:
            return
        with self._lock:
            for key in keys:
                cid = chunk_id_of(key)
                entry = self._chunks.get(cid)
                if entry is None:
                    continue
                entry.members.discard(key)
                if not entry.members:
                    del self._chunks[cid]

    def _touch(self, keys: list[ObjectKey], create: bool) -> None:
        if not keys:
            return
        now = self._clock()
        groups = _group_by_chunk(keys)
        with self._lock:
            # Reverse request order: the first chunk of the request (the
            # prefix head) is moved to the end last, so it is the most
            # recent and is expired after every chunk that extends it.
            for cid, members in reversed(groups):
                entry = self._chunks.get(cid)
                if entry is None:
                    if not create:
                        continue
                    entry = _ChunkEntry(last_access=now)
                    self._chunks[cid] = entry
                else:
                    entry.last_access = now
                    self._chunks.move_to_end(cid)
                # Members come only from creation. A touch may name keys the
                # tier does not hold (lookups pass the whole request), and a
                # member that is never removed would keep the chunk tracked
                # after the tier has dropped it.
                if create:
                    entry.members.update(members)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def expired_chunks(
        self,
        ttl_seconds: float,
        limit: int,
        now: float | None = None,
    ) -> list[tuple[ChunkId, list[ObjectKey]]]:
        """Return up to ``limit`` chunks idle for longer than ``ttl_seconds``.

        Oldest first. The scan stops at the first chunk that is still fresh:
        access times only grow along the order.

        Args:
            ttl_seconds: Idle time after which a chunk expires; ``<= 0``
                returns nothing.
            limit: Maximum number of chunks to return.
            now: Current time; defaults to the tracker clock.

        Returns:
            ``(chunk_id, members)`` pairs. Members is a snapshot.
        """
        if ttl_seconds <= 0 or limit <= 0:
            return []
        if now is None:
            now = self._clock()
        out: list[tuple[ChunkId, list[ObjectKey]]] = []
        with self._lock:
            for cid, entry in self._chunks.items():
                if now - entry.last_access <= ttl_seconds:
                    break
                out.append((cid, list(entry.members)))
                if len(out) >= limit:
                    break
        return out

    def oldest_age_seconds(self, now: float | None = None) -> float:
        """Idle time of the least recently used chunk, 0 when empty."""
        if now is None:
            now = self._clock()
        with self._lock:
            for entry in self._chunks.values():
                return max(0.0, now - entry.last_access)
        return 0.0

    def last_access(self, cid: ChunkId) -> float | None:
        """Last access time of ``cid``, or ``None`` when it is not tracked."""
        with self._lock:
            entry = self._chunks.get(cid)
            return None if entry is None else entry.last_access

    def chunk_order(self) -> list[ChunkId]:
        """Chunk ids oldest first (tests and debugging)."""
        with self._lock:
            return list(self._chunks.keys())

    def __len__(self) -> int:
        with self._lock:
            return len(self._chunks)


class L1IdleTTLListener(L1ManagerListener):
    """Feed L1 manager events into an :class:`IdleTTLTracker`.

    Mirrors :class:`~lmcache.v1.distributed.eviction.L1EvictionPolicy`: a
    reservation counts as creation so an abandoned reservation expires like
    any other unlocked key; releasing a read lock is not an access.
    """

    def __init__(self, tracker: IdleTTLTracker) -> None:
        self._tracker = tracker

    def on_l1_keys_reserved_read(self, keys: list[ObjectKey]) -> None:
        pass

    def on_l1_keys_read_finished(self, keys: list[ObjectKey]) -> None:
        pass

    def on_l1_keys_reserved_write(self, keys: list[ObjectKey]) -> None:
        self._tracker.on_keys_created(keys)

    def on_l1_keys_write_finished(self, keys: list[ObjectKey]) -> None:
        self._tracker.on_keys_created(keys)

    def on_l1_keys_finish_write_and_reserve_read(self, keys: list[ObjectKey]) -> None:
        self._tracker.on_keys_created(keys)

    def on_l1_keys_deleted_by_manager(self, keys: list[ObjectKey]) -> None:
        self._tracker.on_keys_removed(keys)

    def on_l1_keys_accessed(self, keys: list[ObjectKey]) -> None:
        self._tracker.on_keys_touched(keys)


class L2IdleTTLListener(L2AdapterListener):
    """Feed L2 adapter events into an :class:`IdleTTLTracker`."""

    def __init__(self, tracker: IdleTTLTracker) -> None:
        self._tracker = tracker

    def on_l2_keys_stored(self, keys: list[ObjectKey], sizes: list[int]) -> None:
        self._tracker.on_keys_created(keys)

    def on_l2_keys_accessed(self, keys: list[ObjectKey]) -> None:
        self._tracker.on_keys_touched(keys)

    def on_l2_keys_deleted(self, keys: list[ObjectKey]) -> None:
        self._tracker.on_keys_removed(keys)

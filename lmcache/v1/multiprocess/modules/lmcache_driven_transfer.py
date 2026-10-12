# SPDX-License-Identifier: Apache-2.0
"""LMCache-driven KV cache transfer operations for the MPCacheServer."""

# Standard
from dataclasses import dataclass
from typing import Any, Literal, Sequence
import os
import queue
import threading
import time

# First Party
from lmcache import torch_dev
from lmcache.logging import init_logger
from lmcache.utils import (
    EngineType,
    _lmcache_nvtx_annotate,
)
from lmcache.v1.distributed.api import (
    MemoryLayoutDesc,
    ObjectKey,
)
from lmcache.v1.distributed.storage_manager import L1WriteCompletion
from lmcache.v1.gpu_connector.utils import LayoutHints
from lmcache.v1.kv_layer_groups import ObjectGroupInfo
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.mp_observability.event import Event, EventType, next_transfer_key
from lmcache.v1.multiprocess.custom_types import (
    IPCCacheServerKey,
    KVCache,
)
from lmcache.v1.multiprocess.engine_context import MPCacheServerContext
from lmcache.v1.multiprocess.engine_module import InstanceLivenessTarget
from lmcache.v1.multiprocess.group_view import EngineGroupInfo
from lmcache.v1.multiprocess.modules.lookup import resolve_prefetched_obj_keys
from lmcache.v1.multiprocess.native_completion import (
    DeviceHostFuncDispatcher,
    submit_callback_to_stream,
)
from lmcache.v1.multiprocess.object_group_transfer import (
    downsample_and_stage_block_ids,
    transfer_kv_per_object_group,
)
from lmcache.v1.multiprocess.request_handler import HandlerType, request_handler
from lmcache.v1.periodic_thread import (
    PeriodicThread,
    ThreadLevel,
    ThreadRunSummary,
    create_periodic_thread,
)
from lmcache.v1.platform.base.cache_context import (
    BaseCacheContext,
    set_device_alloc_guard,
    set_failed_init_closer,
)
from lmcache.v1.platform.base.event_ipc import (
    EventIPCBackend,
    get_event_ipc_backend,
)
from lmcache.v1.platform.cache_context import create_cache_context
import lmcache.lmcache_native as lmcache_native

logger = init_logger(__name__)


def get_layout_desc(
    cache_context: BaseCacheContext,
    num_tokens: int,
    object_group_id: int,
) -> MemoryLayoutDesc:
    """Get the memory layout description for a specific object group.

    The returned layout describes the single memory object that backs
    ``object_group_id``: one (shape, dtype) entry per kernel group in that
    object group, in the kernel groups' declared layout order. Kernel groups
    may have different shapes and dtypes.

    Args:
        cache_context: The cache context containing the KV cache information.
        num_tokens: The number of tokens to determine the layout for.
        object_group_id: Index of the object group whose layout to build.

    Returns:
        MemoryLayoutDesc: The memory layout description containing shapes and
        dtypes, one entry per kernel group in the object group.
    """
    object_group = cache_context.kv_layer_groups_manager.object_groups[object_group_id]
    shapes_and_dtypes = [
        cache_context.get_kernel_group_shape_dtype(num_tokens, kernel_group_idx)
        for kernel_group_idx in object_group.kernel_group_indices
    ]
    shapes, dtypes = zip(*shapes_and_dtypes, strict=False)
    return MemoryLayoutDesc(shapes=list(shapes), dtypes=list(dtypes))


def all_null_chunk_masks(
    block_ids: Sequence[Sequence[int]],
    object_groups: Sequence[ObjectGroupInfo],
    blocks_per_chunk: Sequence[int],
    num_chunks: int,
    null_block_id: int = 0,
) -> list[list[bool]]:
    """Mark, per object group, the chunks whose engine block ids are all null.

    A chunk is null for an object group when every block ID of every kernel
    group equals the server's null marker. Align-mode Mamba/linear
    layers produce such chunks: only the block holding the last recurrent state
    is real, so every earlier chunk is null. These chunks must not be stored --
    the null block carries no valid KV, and object keys are content hashes, so
    committing them would serve garbage to a later prefix hit.

    Args:
        block_ids: Raw per-kernel-group engine block ids (before any downsample),
            indexed by kernel-group index.
        object_groups: The object groups, indexed by object-group id.
        blocks_per_chunk: Blocks in one chunk per kernel group, indexed by
            kernel-group index.
        num_chunks: Number of chunks in the request.
        null_block_id: Server-wide block ID denoting absent data. Defaults to
            the historical vLLM null block zero.

    Returns:
        ``mask[g][i]`` is True iff chunk ``i`` is all-null for object group ``g``.
    """
    masks: list[list[bool]] = []
    for group in object_groups:
        chunk_null: list[bool] = []
        for i in range(num_chunks):
            is_null = True
            for kg in group.kernel_group_indices:
                bpc = blocks_per_chunk[kg]
                if any(
                    block != null_block_id
                    for block in block_ids[kg][i * bpc : (i + 1) * bpc]
                ):
                    is_null = False
                    break
            chunk_null.append(is_null)
        masks.append(chunk_null)
    return masks


@dataclass
class ContextEntry:
    """Registered cache context metadata for a single worker instance.

    The concrete type is whatever :func:`create_cache_context` returned
    for the wrapper list at registration time -- a
    :class:`GPUCacheContext` for CUDA-IPC wrappers, a
    :class:`CPUCacheContext` for POSIX-SHM wrappers. Both expose
    the same ``kv_tensors`` / ``engine_kv_format`` / ``num_layers`` / ...
    duck-typed surface, so downstream consumers stay agnostic.

    Args:
        cache_context: Platform cache context (GPU or CPU) managing
            shape and pointers to the registered KV cache tensors.
        model_name: The name of the model associated with this KV cache.
        world_size: The world size associated with this KV cache.
        last_seen: ``time.monotonic()`` of the most recent activity from
            this instance (register, PING, store, or retrieve). Drives reaping.
        has_liveness_signal: True once the instance has sent at least one
            PING. Selects the reap window (timeout vs registration grace).
            Latched only by PING, never by traffic.
        event_backend: Cached event backend selected for this context's device.
        generation: Unique, increasing id of this registration. A worker can
            re-register under the same instance ID, so per-registration state
            (store-commit poisoning) is keyed by this instead.
    """

    cache_context: BaseCacheContext
    model_name: str
    world_size: int
    last_seen: float = 0.0
    has_liveness_signal: bool = False
    event_backend: EventIPCBackend | None = None
    generation: int = 0


#: Exit code used by ``stuck_action="exit"`` (EX_TEMPFAIL): the container
#: runtime restarts the server and the driver reclaims every mapping.
STUCK_RELEASE_EXIT_CODE = 75

#: How long a force-completed registration stays poisoned after its release
#: finished, so commit callbacks still buffered in the host-func dispatcher
#: are dropped rather than committed.
_POISON_RETENTION_SECONDS = 300.0


#: Free device memory kept on top of a guarded allocation (see
#: _check_device_alloc), for other processes allocating concurrently.
_ALLOC_GUARD_MARGIN_BYTES = 64 << 20

#: A registered instance's wait pending longer than this makes its device
#: unsafe for the allocator's OOM path (see _check_device_alloc). Shorter than
#: stale_wait_seconds: that one only delays unmaps, this one guards the MQ loop.
_ALLOC_GUARD_WAIT_SECONDS = 1.0


def _allocator_segment_bytes(nbytes: int) -> int:
    """Return the size of the segment the caching allocator would cudaMalloc.

    Mirrors PyTorch's CUDACachingAllocator: requests up to 1 MiB come from a
    2 MiB small-pool segment, requests under 10 MiB from a 20 MiB segment,
    larger ones are rounded up to a multiple of 2 MiB.

    Args:
        nbytes: The requested size in bytes.

    Returns:
        The segment size in bytes.
    """
    mib = 1 << 20
    if nbytes <= mib:
        return 2 * mib
    if nbytes < 10 * mib:
        return 20 * mib
    return -(-nbytes // (2 * mib)) * (2 * mib)


@dataclass(frozen=True)
class IPCReleaseConfig:
    """How an unregistered or reaped instance's CUDA IPC mappings are released.

    Releasing unmaps the worker's KV pool, which first needs the context
    stream drained. A stream waiting on an event from a dead worker never
    drains on its own (LMCache#4676), so the release runs in stages instead of
    blocking: wait for a natural drain, then force-complete the pending
    imported events, then treat the instance as stuck.

    Args:
        drain_grace_seconds: How long to wait for the stream to drain on its
            own before force-completing its imported events.
        stuck_timeout_seconds: How long to wait after forcing (or after the
            grace, when forcing is impossible or unsafe) before the instance
            counts as stuck.
        stuck_action: ``"quarantine"`` keeps a stuck instance mapped and
            reports it; ``"exit"`` terminates the server so it restarts and
            the driver reclaims the memory.
        poll_interval_seconds: How often pending releases are re-checked.
        max_force_wait_age_seconds: Never force when the oldest pending wait
            is older than this. A forced STORE copy writes into the L1 buffer
            it reserved; once the reservation's write lock has expired that
            buffer may belong to another key, so forcing would corrupt the
            shared cache. None derives half the smallest L1 write TTL, and
            disables forcing when no TTL can be read.
        stale_wait_seconds: A wait pending longer than this on a registered
            instance marks its device as having a stuck stream; nothing is
            unmapped on that device until it clears. Live waits complete
            within one engine step.
    """

    drain_grace_seconds: float = 30.0
    stuck_timeout_seconds: float = 60.0
    stuck_action: Literal["quarantine", "exit"] = "quarantine"
    poll_interval_seconds: float = 1.0
    max_force_wait_age_seconds: float | None = None
    stale_wait_seconds: float = 10.0


@dataclass
class _PendingRelease:
    """A popped registration whose IPC mappings are not released yet.

    Holding ``entry`` keeps the IPC mappings open on purpose: unmapping memory
    that queued GPU work still references is what hangs the driver.
    """

    instance_id: int
    entry: ContextEntry
    started: float
    forced_at: float | None = None
    stuck_since: float | None = None
    stuck: bool = False
    #: Why the release cannot be forced, once known ("unforceable" backend,
    #: "too_old" waits, "nothing_held"), or None.
    force_blocked: str | None = None
    #: Set once "nothing_held" has been reported, to log and count it once.
    reported_nothing_held: bool = False
    #: Drained, but held back by a stuck stream on the same device.
    held_back: bool = False


def _min_l1_write_ttl_seconds(storage_manager: Any) -> float | None:
    """Return the smallest L1 write-lock TTL of the storage manager.

    Args:
        storage_manager: The server's storage manager.

    Returns:
        The TTL in seconds, or None when no L1 manager exposes one.
    """
    managers = getattr(storage_manager, "_l1_managers_by_id", None)
    ttls: list[float] = []
    if isinstance(managers, dict):
        for manager in managers.values():
            config = getattr(manager, "config", None)
            ttl = getattr(config, "write_ttl_seconds", None)
            if not isinstance(ttl, (int, float)):
                ttl = getattr(manager, "_write_ttl_seconds", None)
            if isinstance(ttl, (int, float)) and ttl > 0:
                ttls.append(float(ttl))
    return min(ttls) if ttls else None


class LMCacheDrivenTransferModule(InstanceLivenessTarget):
    """Handles LMCache-driven KV cache transfer operations.

    Owns GPU context registrations and provides handlers for
    register, unregister, store, and retrieve of GPU KV caches.

    Args:
        ctx: The shared engine context.
        release_config: How unregistered or reaped instances release their
            IPC mappings; see :class:`IPCReleaseConfig`.
    """

    def __init__(
        self,
        ctx: MPCacheServerContext,
        release_config: IPCReleaseConfig | None = None,
    ) -> None:
        self._ctx = ctx
        self._init_release_state(release_config)
        self._cache_contexts: dict[int, ContextEntry] = {}
        # Guards all reads/writes of _cache_contexts. The reaper mutates it
        # off the MQ main loop, so register/unregister/store/retrieve and
        # report_status all serialize through this lock. Held only for dict
        # ops -- never across context creation, layout-registry calls, or
        # empty_cache (leaf-lock invariant: no thread holds two locks).
        self._lock = threading.Lock()

        # Route finish_write / finish_read_prefetched through a C++ host
        # callback so the driver thread doesn't acquire the GIL.
        self._device_host_func_dispatcher = DeviceHostFuncDispatcher()
        self._device_host_func_dispatcher.register(
            "finish_write",
            self._ctx.storage_manager.finish_write,
            payload_type=list[ObjectKey],
        )
        self._device_host_func_dispatcher.register(
            "finish_write_by_owner",
            self._ctx.storage_manager.finish_write_by_owner,
            payload_type=L1WriteCompletion,
        )
        self._device_host_func_dispatcher.register(
            "finish_read_prefetched",
            self._ctx.storage_manager.finish_read_prefetched,
            payload_type=list[ObjectKey],
        )
        self._device_host_func_dispatcher.register(
            "finish_read_by_owner",
            self._ctx.storage_manager.finish_read_by_owner,
            payload_type=L1WriteCompletion,
        )
        self._device_host_func_dispatcher.register(
            "release_imported_event",
            self._release_imported_event,
            payload_type=tuple[int, int],
        )
        self._device_host_func_dispatcher.register(
            "finish_write_unless_poisoned",
            self._finish_write_unless_poisoned,
            payload_type=tuple[int, L1WriteCompletion],
        )
        self._device_host_func_dispatcher.start()
        # Kept so close() removes exactly the hooks this module installed.
        self._installed_hooks = (self._submit_rollback_close, self._check_device_alloc)
        set_failed_init_closer(self._installed_hooks[0])
        set_device_alloc_guard(self._installed_hooks[1])

    def _init_release_state(
        self, release_config: IPCReleaseConfig | None = None
    ) -> None:
        """Initialize the staged IPC release state (see _release_entries).

        Args:
            release_config: Release timing and stuck policy; defaults apply
                when None.
        """
        cfg = release_config or IPCReleaseConfig()
        self._release_config = cfg
        if cfg.max_force_wait_age_seconds is not None:
            self._max_force_wait_age = cfg.max_force_wait_age_seconds
        else:
            ttl = _min_l1_write_ttl_seconds(getattr(self._ctx, "storage_manager", None))
            # Unknown TTL: never force rather than guess a limit that could
            # outlive a reservation.
            self._max_force_wait_age = 0.5 * ttl if ttl is not None else 0.0
        # Popped registrations whose mappings are not released yet, by a
        # unique token (one instance ID can be released more than once).
        # Guarded by _release_lock, a leaf lock like _lock.
        self._pending_releases: dict[int, _PendingRelease] = {}
        self._next_release_token = 0
        # Force-completed registrations (by generation) -> release finish time
        # (None while pending). Their store commits are dropped: the copies
        # ran without the worker's data.
        self._poisoned: dict[int, float | None] = {}
        self._next_generation = 1
        self._release_counters = {
            "released": 0,
            "deferred": 0,
            "forced": 0,
            "force_blocked": 0,
            "release_errors": 0,
            "dropped_commits": 0,
            "alloc_refused": 0,
        }
        self._release_lock = threading.Lock()
        # Serializes release cycles (release thread vs. shutdown).
        self._release_cycle_lock = threading.Lock()
        self._release_thread: PeriodicThread | None = None
        self._closing = False
        # One unmapper thread and job queue per device key; _unmapping counts
        # entries queued or being unmapped.
        self._unmap_queues: dict[str, queue.SimpleQueue] = {}
        self._unmapping = 0
        self._unmap_stopped = False
        self._query_failures_logged: set[int] = set()

    def _finish_write_unless_poisoned(
        self, payload: tuple[int, L1WriteCompletion]
    ) -> None:
        """Commit a store, unless its registration's events were forced.

        A forced event lets the queued D2H copy run whether or not the worker
        finished writing those blocks, so the copied KV may be partial.
        Committing it would serve garbage to every reader of the shared pool.
        Dropped reservations expire under the existing write-TTL rules.

        Args:
            payload: ``(generation, completion)`` of the store.
        """
        generation, completion = payload
        with self._release_lock:
            poisoned = generation in self._poisoned
            if poisoned:
                self._release_counters["dropped_commits"] += 1
        if poisoned:
            logger.warning(
                "Dropping store commit of force-released registration %d",
                generation,
            )
            return
        self._ctx.storage_manager.finish_write_by_owner(completion)

    def _release_imported_event(self, payload: tuple[int, int]) -> None:
        """Drop an imported worker event; the stream wait queued on it has run.

        Args:
            payload: ``(instance_id, import_token)`` of the imported event.
        """
        instance_id, import_token = payload
        entry = self.get_and_touch_context_entry(instance_id)
        if entry is not None and entry.cache_context.release_imported_event(
            import_token
        ):
            return
        # Unregistered or reaped meanwhile: the context waits in a pending
        # release, whose drain check relies on its holds being dropped.
        with self._release_lock:
            contexts = [
                p.entry.cache_context
                for p in self._pending_releases.values()
                if p.instance_id == instance_id
            ]
        for context in contexts:
            if context.release_imported_event(import_token):
                return

    def register_host_func(self, kind: str, handler: Any, payload_type: Any) -> None:
        """Register *handler* for *kind* on the per-process device host-func
        dispatcher (stream-ordered callbacks without a driver-thread GIL
        acquire); pair with ``submit_callback_to_stream``."""
        self._device_host_func_dispatcher.register(kind, handler, payload_type)

    @property
    def context(self) -> MPCacheServerContext:
        """Return the shared engine context. Exposed for testing only."""
        return self._ctx

    def get_and_touch_context_entry(self, instance_id: int) -> ContextEntry | None:
        """Return the entry for ``instance_id``, refreshing its last-seen time.

        The refresh keeps an actively transferring worker from being reaped
        even if its PINGs are briefly delayed. Does not latch the
        ping-proven flag -- only PINGs do that.

        Args:
            instance_id: The worker instance ID.

        Returns:
            The entry, or None if the instance is not (or no longer) tracked.
        """
        now = time.monotonic()
        with self._lock:
            entry = self._cache_contexts.get(instance_id)
            if entry is not None:
                entry.last_seen = now
            return entry

    def _release_failed_retrieve_locks(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
    ) -> None:
        """Release only one failed instance's unconsumed lookup locks.

        The lookup session is the ownership record.  If it is absent or does
        not match the RETRIEVE range, no release is attempted: L1 locks are
        anonymous refcounts, so guessing could consume a concurrent reader's
        lock.  ``claim_failed_retrieve_release`` also makes duplicate failure
        responses idempotent.
        """
        session = self._ctx.session_manager.get(key.request_id)
        if session is None:
            logger.warning(
                "Cannot release RETRIEVE locks for unregistered instance %d: "
                "request %s has no lookup session",
                instance_id,
                key.request_id,
            )
            return

        lock_state = session.prepare_failed_retrieve_release(key)
        if lock_state is None:
            return
        hit_chunks, locked_gids, group_windows, lookup_generation = lock_state
        obj_keys = resolve_prefetched_obj_keys(
            self._ctx,
            key,
            hit_chunks,
            locked_gids,
            group_windows=group_windows,
        )
        if not session.claim_failed_retrieve_release(
            instance_id, key, lookup_generation
        ):
            return
        if obj_keys:
            # One failed RETRIEVE owns one read lock per key.  In
            # particular, do not release the scheduler's whole MLA
            # reservation here: the remaining TP workers and concurrent
            # requests still own their independent read locks.
            self._ctx.storage_manager.finish_read_prefetched(
                obj_keys,
                read_locks=1,
                l1_owners=self._ctx.get_read_owners(key.request_id),
            )

    def context_entries_snapshot(self) -> dict[int, ContextEntry]:
        """Return a shallow copy of the registry for iteration or status.

        Returns:
            A new dict mapping instance ID to entry; does not refresh
            last-seen times.
        """
        with self._lock:
            return dict(self._cache_contexts)

    def touch_instance(self, instance_id: int) -> bool:
        """Refresh the worker's last-seen time and mark it ping-proven.

        A no-op if the instance is not tracked.

        Args:
            instance_id: The worker instance ID.

        Returns:
            Whether the instance is tracked.
        """
        now = time.monotonic()
        with self._lock:
            entry = self._cache_contexts.get(instance_id)
            if entry is None:
                return False
            entry.last_seen = now
            entry.has_liveness_signal = True
            return True

    def tracked_instance_count(self) -> int:
        """Return the number of currently registered instances."""
        with self._lock:
            return len(self._cache_contexts)

    def reap_stale_instances(
        self, reap_timeout_s: float, registration_grace_s: float
    ) -> list[int]:
        """Reap GPU registrations that have gone silent.

        A ping-proven instance is judged against ``reap_timeout_s``; one
        that has never pinged against the larger ``registration_grace_s``.

        Args:
            reap_timeout_s: Silence budget for ping-proven instances.
            registration_grace_s: Silence budget for never-pinged instances.

        Returns:
            The instance IDs reaped this scan.
        """
        now = time.monotonic()
        reaped: list[tuple[int, ContextEntry]] = []
        with self._lock:
            stale_ids = [
                iid
                for iid, entry in self._cache_contexts.items()
                if now - entry.last_seen
                > (
                    reap_timeout_s
                    if entry.has_liveness_signal
                    else registration_grace_s
                )
            ]
            for iid in stale_ids:
                reaped.append((iid, self._cache_contexts.pop(iid)))
        reaped_ids: list[int] = []
        for iid, e in reaped:
            logger.warning(
                "Reaped GPU instance %d: silent for %.1fs (pinged=%s)",
                iid,
                now - e.last_seen,
                e.has_liveness_signal,
            )
            reaped_ids.append(iid)
        if reaped:
            del e  # a bound name would pin the final entry (see _release_entries)
            self._release_entries(reaped)
        return reaped_ids

    def _release_entries(self, released: list[tuple[int, ContextEntry]]) -> None:
        """Hand popped registrations to the release thread; never unmap here.

        Unmapping (``cudaIpcCloseMemHandle``) waits for the whole device to go
        idle. If any stream on that device is stuck on a dead worker's event,
        it waits until that wait is forced -- minutes, or forever. The callers
        here are the MQ main loop (UNREGISTER) and the reaper; blocking either
        stalls the server for every worker on the node. So every release,
        drained or not, goes through the release thread, and the unmap itself
        runs on a separate unmapper thread (see _release_cycle_locked).

        Args:
            released: ``(instance_id, entry)`` pairs already popped from the
                registry. They must be the only remaining references; the
                list is cleared before this returns.
        """
        if not released:
            return
        now = time.monotonic()
        for instance_id, entry in released:
            self._ctx.layout_desc_registry.unregister(
                entry.model_name, entry.world_size
            )
            with self._release_lock:
                token = self._next_release_token
                self._next_release_token += 1
                self._pending_releases[token] = _PendingRelease(
                    instance_id=instance_id, entry=entry, started=now
                )
                self._release_counters["deferred"] += 1
        del entry
        released.clear()
        thread = self._ensure_release_thread()
        if thread is not None:
            thread.wake()

    def _stream_drained(self, entry: ContextEntry) -> bool:
        """Return whether the entry's transfer stream has no queued work.

        A query that raises (for example a sticky CUDA error) counts as not
        drained, so the entry goes through the stuck path instead of being
        unmapped under a broken context; it is logged once per entry.

        Args:
            entry: A registered or released entry.

        Returns:
            True when it is safe to unmap the entry's mappings.
        """
        stream = getattr(entry.cache_context, "stream", None)
        if stream is None:
            return True
        # A held import means a wait on a worker event is still queued on the
        # stream. Never query it then: behind a wait that never completes the
        # stream's work queue fills up (about 1k ops), and cudaStreamQuery on
        # it then blocks -- holding the GIL, which freezes the whole server,
        # including the release thread that would force the wait.
        events, _ = self._held_imports(entry)
        if events:
            return False
        try:
            return bool(stream.query())
        except Exception:
            key = id(entry)
            with self._release_lock:
                first = key not in self._query_failures_logged
                if first:
                    if len(self._query_failures_logged) > 10_000:
                        self._query_failures_logged.clear()
                    self._query_failures_logged.add(key)
            if first:
                logger.exception("Querying a context stream failed")
            return False

    @staticmethod
    def _entry_device_key(entry: ContextEntry) -> str:
        """Return a key for the device an entry's mappings live on."""
        return str(getattr(entry.cache_context, "device", None))

    @staticmethod
    def _held_imports(entry: ContextEntry) -> tuple[list[object], float | None]:
        """Return the entry's held imported events and oldest hold time.

        Contexts without hold tracking report nothing held.
        """
        snapshot = getattr(entry.cache_context, "held_imports_snapshot", None)
        if snapshot is None:
            return [], None
        try:
            events, oldest = snapshot()
        except (TypeError, ValueError):
            return [], None
        if not isinstance(oldest, (int, float)):
            oldest = None
        return list(events), oldest

    def _stale_registered_devices(
        self, now: float, limit: float | None = None
    ) -> dict[str, list[int]]:
        """Return devices where a registered instance has a stale wait.

        A wait older than ``stale_wait_seconds`` on a still-registered
        instance belongs to a worker that is dead or hung but not reaped yet.
        Its stream will not drain until the reaper hands it over and it is
        forced, so nothing may be unmapped on that device meanwhile.

        Args:
            now: Current ``time.monotonic()``.
            limit: Age that makes a wait stale; None uses
                ``stale_wait_seconds``.

        Returns:
            Device key (see _entry_device_key) -> instance IDs with stale waits.
        """
        with self._lock:
            entries = list(self._cache_contexts.items())
        stale: dict[str, list[int]] = {}
        if limit is None:
            limit = self._release_config.stale_wait_seconds
        for instance_id, entry in entries:
            _, oldest = self._held_imports(entry)
            if (
                oldest is not None
                and now - oldest > limit
                and not self._stream_drained(entry)
            ):
                stale.setdefault(self._entry_device_key(entry), []).append(instance_id)
        return stale

    def _blocked_devices(self, now: float) -> set[str]:
        """Return devices where an unmap could block right now.

        A device is blocked while a registered instance on it has a stale
        wait or a pending release on it is not drained.

        Args:
            now: Current ``time.monotonic()``.

        Returns:
            Device keys.
        """
        blocked = set(self._stale_registered_devices(now))
        with self._release_lock:
            pending = [p.entry for p in self._pending_releases.values()]
        for entry in pending:
            if not self._stream_drained(entry):
                blocked.add(self._entry_device_key(entry))
        del pending
        return blocked

    def _check_device_alloc(self, device: Any, nbytes: int) -> None:
        """Refuse a registration's allocation that could hang the MQ loop.

        Installed as the platform's device allocation guard; it runs before
        each GPU allocation of a cache context built by REGISTER, on the MQ
        loop. When cudaMalloc fails, PyTorch's caching allocator cudaFrees
        its cached blocks and retries, and cudaFree waits for the whole
        device: behind a stream stuck on a dead worker's event it never
        returns, stalling the server for every worker on the node. So when
        the device lacks room for a fresh segment and has a possibly stuck
        stream, fail the registration instead (the worker sees the error).
        With room, or with every stream on the device live, allocate as usual:
        the allocator may still reclaim cached memory, but any wait it
        blocks on completes.

        Args:
            device: The device about to be allocated on.
            nbytes: The requested size in bytes.

        Raises:
            RuntimeError: The allocation is refused.
        """
        if getattr(device, "type", None) != "cuda":
            return
        need = _allocator_segment_bytes(nbytes) + _ALLOC_GUARD_MARGIN_BYTES
        free, _ = torch_dev.mem_get_info(device)
        if free >= need:
            return
        reason = self._device_alloc_risk(str(device), time.monotonic())
        if reason is None:
            logger.warning(
                "Allocating %d bytes on %s with only %d bytes free; no stream "
                "there looks stuck, so the allocator may reclaim cached memory",
                nbytes,
                device,
                free,
            )
            return
        with self._release_lock:
            self._release_counters["alloc_refused"] += 1
        raise RuntimeError(
            f"Refusing to allocate {nbytes} bytes on {device} during "
            f"registration: {free} bytes free and {reason}, so an out-of-memory "
            "retry could block the server behind a stuck stream; retry once "
            "memory is freed or the stuck release completes"
        )

    def _device_alloc_risk(self, device_key: str, now: float) -> str | None:
        """Return why the allocator's OOM path could block on a device.

        Args:
            device_key: The device (see _entry_device_key).
            now: Current ``time.monotonic()``.

        Returns:
            A reason, or None when every stream on the device is live.
        """
        stale = self._stale_registered_devices(now, _ALLOC_GUARD_WAIT_SECONDS)
        if device_key in stale:
            return f"instance(s) {sorted(stale[device_key])} have a wait pending"
        with self._release_lock:
            pending = [
                (p.instance_id, p.entry)
                for p in self._pending_releases.values()
                if self._entry_device_key(p.entry) == device_key
            ]
        busy = sorted(iid for iid, entry in pending if not self._stream_drained(entry))
        del pending
        if busy:
            return f"the release of instance(s) {busy} has not drained"
        return None

    def _finalize_releases(
        self,
        entries: list[ContextEntry],
        closers: Sequence[Any] = (),
    ) -> None:
        """Unmap drained entries and reclaim their device memory.

        Runs on an unmapper thread only. ``empty_cache`` frees cached device
        memory on every device and can block like an unmap, so it runs only
        when no release is pending, no device is blocked and no other unmap is
        in progress.

        Args:
            entries: Entries whose streams have drained. They must be the only
                remaining references; the list is cleared before memory is
                reclaimed.
            closers: Extra objects with ``close()`` to close first (wrappers
                left open by a failed registration).
        """
        if not entries and not closers:
            return
        errors = 0
        for closer in closers:
            try:
                closer.close()
            except Exception:
                errors += 1
                logger.exception("Closing a rolled-back KV wrapper failed")
        for entry in entries:
            try:
                entry.cache_context.close()
            except Exception:
                # Keep going: one failed close must not strand the others.
                errors += 1
                logger.exception("Closing a released cache context failed")
        if entries:
            del entry
        count = len(entries)
        entries.clear()
        with self._release_lock:
            self._release_counters["released"] += count
            self._release_counters["release_errors"] += errors
            others_busy = bool(self._pending_releases) or self._unmapping > count
        # ipc_collect() only unmaps a CUDA-IPC-imported segment once its last
        # tensor reference is gone (LMCache#4014), hence the clear() above.
        if not others_busy and not self._blocked_devices(time.monotonic()):
            torch_dev.empty_cache()
        ipc_collect = getattr(torch_dev, "ipc_collect", None)
        if ipc_collect is not None:
            # Backends without IPC collection omit this optional operation.
            ipc_collect()

    def _submit_unmap(self, entries: list[ContextEntry]) -> None:
        """Queue drained entries on their devices' unmapper threads.

        Args:
            entries: Entries to unmap; ownership moves to the unmappers (the
                caller must drop its references). The list is cleared.
        """
        by_device: dict[str, list[ContextEntry]] = {}
        for entry in entries:
            by_device.setdefault(self._entry_device_key(entry), []).append(entry)
        if entries:
            del entry
        entries.clear()
        for device_key, batch in by_device.items():
            self._enqueue_unmap(device_key, batch, ())
        by_device.clear()

    def _submit_rollback_close(self, wrappers: list[Any], device_key: str) -> None:
        """Close wrappers a failed registration left open, off the caller.

        Installed as the platform's failed-construction closer: the caller is
        REGISTER on the MQ loop, and the unmap could block behind a stuck
        stream on the same device.

        Args:
            wrappers: The opened wrappers.
            device_key: Their device, or "None" when unknown.
        """
        self._enqueue_unmap(device_key, [], tuple(wrappers))

    def _enqueue_unmap(
        self, device_key: str, entries: list[ContextEntry], closers: tuple
    ) -> None:
        """Put one job on the unmapper thread of ``device_key``."""
        with self._release_lock:
            if self._unmap_stopped:
                # Shutting down: process exit reclaims these mappings.
                logger.warning(
                    "Dropping an unmap job for %s queued after shutdown", device_key
                )
                return
            self._unmapping += len(entries)
            unmap_queue = self._unmap_queues.get(device_key)
            if unmap_queue is None:
                unmap_queue = queue.SimpleQueue()
                self._unmap_queues[device_key] = unmap_queue
                threading.Thread(
                    target=self._unmap_loop,
                    args=(device_key, unmap_queue),
                    name=f"lmcache-mp-ipc-unmap-{device_key}",
                    daemon=True,
                ).start()
        unmap_queue.put((entries, closers))

    def _unmap_loop(self, device_key: str, unmap_queue: queue.SimpleQueue) -> None:
        """Unmapper thread of one device: finalize jobs until told to stop.

        An unmap blocks while any stream on its device is stuck, so each job
        waits until the device is no longer blocked (re-checked here, since a
        stream can get stuck after the job was queued), and each device has
        its own thread so a stuck GPU never delays another GPU's unmaps. The
        release thread, which does the forcing, never waits on this.
        """
        poll = self._release_config.poll_interval_seconds
        while True:
            job = unmap_queue.get()
            if job is None:
                return
            entries, closers = job
            del job
            count = len(entries)
            try:
                while True:
                    blocked = self._blocked_devices(time.monotonic())
                    # A rollback with an unknown device waits for every device.
                    if not blocked or (
                        device_key != "None" and device_key not in blocked
                    ):
                        break
                    time.sleep(poll)
                self._finalize_releases(entries, closers)
            except Exception:
                logger.exception("Unmapping released IPC mappings failed")
            finally:
                del entries, closers
                with self._release_lock:
                    self._unmapping -= count

    def _ensure_release_thread(self) -> PeriodicThread | None:
        """Start the release thread on the first release.

        Returns:
            The release thread, or None while shutting down.
        """
        with self._release_lock:
            if self._closing:
                return None
            if self._release_thread is not None:
                return self._release_thread
            self._release_thread = create_periodic_thread(
                name="lmcache-mp-ipc-release",
                interval=self._release_config.poll_interval_seconds,
                execute_fn=self._release_cycle,
                level=ThreadLevel.MEDIUM,
            )
            thread = self._release_thread
        thread.start()
        return thread

    def _release_cycle(
        self, allow_exit: bool = True, lock_timeout: float = -1
    ) -> ThreadRunSummary:
        """Advance every pending release by one step.

        A drained entry is queued for unmapping, unless its device still has a
        busy pending release or a registered instance with a stale wait. A
        busy one past the drain grace has its pending imported events
        force-completed, every cycle (new waits can still be queued by a
        transfer that took the entry before it was popped). One still busy
        after the stuck timeout is reported and, with ``stuck_action="exit"``,
        terminates the server. Never blocks on the GPU.

        Args:
            allow_exit: False during shutdown, which must never exit early.
            lock_timeout: How long to wait for a concurrent cycle; -1 waits.

        Returns:
            A summary of this cycle.
        """
        if not self._release_cycle_lock.acquire(timeout=lock_timeout):
            return ThreadRunSummary(success=False, message="cycle busy")
        try:
            return self._release_cycle_locked(allow_exit)
        finally:
            self._release_cycle_lock.release()

    def _release_cycle_locked(self, allow_exit: bool) -> ThreadRunSummary:
        """One release cycle; the caller holds ``_release_cycle_lock``."""
        cfg = self._release_config
        now = time.monotonic()
        with self._release_lock:
            pending = list(self._pending_releases.items())
        drained = {token: self._stream_drained(p.entry) for token, p in pending}
        blocked_devices = set(self._stale_registered_devices(now)) | {
            self._entry_device_key(p.entry)
            for token, p in pending
            if not drained[token]
        }
        ready_tokens: list[int] = []
        newly_stuck: list[tuple[int, float]] = []
        num_stuck = 0
        for token, item in pending:
            if drained[token]:
                if self._entry_device_key(item.entry) in blocked_devices:
                    item.held_back = True
                else:
                    ready_tokens.append(token)
                continue
            if now - item.started < cfg.drain_grace_seconds:
                continue
            if item.force_blocked is None:
                self._force_release(item, now)
            if item.stuck_since is None:
                item.stuck_since = (
                    item.forced_at
                    if item.forced_at is not None
                    else item.started + cfg.drain_grace_seconds
                )
            if now - item.stuck_since >= cfg.stuck_timeout_seconds:
                num_stuck += 1
                if not item.stuck:
                    item.stuck = True
                    newly_stuck.append((item.instance_id, now - item.stuck_since))
        if pending:
            del item
        del pending, drained
        for instance_id, waited in newly_stuck:
            logger.error(
                "IPC release of instance %d is stuck: its stream is still busy "
                "%.0fs after it was forced (or found unforceable). Its KV "
                "mappings, and every later release on the same GPU, stay held "
                "until the server restarts.",
                instance_id,
                waited,
            )
        with self._release_lock:
            popped = [self._pending_releases.pop(t, None) for t in ready_tokens]
            ready = [p.entry for p in popped if p is not None]
            for gen in [p.entry.generation for p in popped if p is not None]:
                if gen in self._poisoned:
                    self._poisoned[gen] = now
            # No name may stay bound to a released entry past this point:
            # ipc_collect() only unmaps segments nothing references.
            del popped
            # Commit callbacks of a finished release are long drained by now.
            for generation, done in list(self._poisoned.items()):
                if done is not None and now - done > _POISON_RETENTION_SECONDS:
                    del self._poisoned[generation]
            remaining = len(self._pending_releases)
        released = len(ready)
        self._submit_unmap(ready)
        del ready
        if num_stuck and allow_exit and cfg.stuck_action == "exit":
            self._exit_for_stuck_release(num_stuck)
        return ThreadRunSummary(
            success=num_stuck == 0,
            message=f"released={released} pending={remaining} stuck={num_stuck}",
        )

    def _force_release(self, item: _PendingRelease, now: float) -> None:
        """Force-complete the imported events a pending release waits on.

        The registration is poisoned first, so any store whose copy the force
        lets run is not committed. Forcing is refused (and the release left to
        the stuck path) when the backend cannot force, or when the oldest
        pending wait is older than the force age limit: its STORE's L1
        reservation may have expired and been handed to another key.

        Args:
            item: The pending release.
            now: Current ``time.monotonic()``.
        """
        force = getattr(item.entry.event_backend, "force_complete_events", None)
        snapshot = getattr(item.entry.cache_context, "held_imports_snapshot", None)
        events, oldest = snapshot() if snapshot is not None else ([], None)
        reason: str | None = None
        if force is None:
            reason = "unforceable"
        elif oldest is None:
            reason = "nothing_held"
        elif now - oldest > self._max_force_wait_age:
            reason = "too_old"
        if reason is not None:
            # A wait can be queued a moment before it is held, so an empty
            # snapshot is retried next cycle; the others never get better.
            if reason == "nothing_held":
                if item.reported_nothing_held:
                    return
                item.reported_nothing_held = True
            else:
                item.force_blocked = reason
            with self._release_lock:
                self._release_counters["force_blocked"] += 1
            logger.error(
                "IPC release of instance %d: stream still busy after %.0fs and "
                "cannot be forced (%s). unforceable: the %s event backend "
                "cannot complete events for a dead peer -- run server and "
                "workers with isolated IPC (--isolated-ipc / "
                "lmcache.mp.isolated_ipc). too_old: a pending wait is older than "
                "%.0fs, so its L1 reservation may already belong to another key.",
                item.instance_id,
                now - item.started,
                reason,
                type(item.entry.event_backend).__name__,
                self._max_force_wait_age,
            )
            return
        with self._release_lock:
            self._poisoned.setdefault(item.entry.generation, None)
        try:
            slots = force(events)
        except Exception:
            logger.exception(
                "Force-completing events of instance %d failed", item.instance_id
            )
            item.force_blocked = "force_failed"
            return
        if item.forced_at is None:
            item.forced_at = now
            # The stuck timeout counts from the first real force, even when
            # earlier cycles found nothing to force.
            item.stuck_since = now
            with self._release_lock:
                self._release_counters["forced"] += 1
            logger.warning(
                "IPC release of instance %d: stream still busy after %.0fs; "
                "force-completed %d semaphore slot(s) of the dead worker. Its "
                "in-flight stores are dropped, not committed.",
                item.instance_id,
                now - item.started,
                slots,
            )

    @staticmethod
    def _exit_for_stuck_release(num_stuck: int) -> None:
        """Terminate the server so a restart reclaims stuck mappings.

        Process exit is the one thing that always unmaps IPC memory. Uses
        ``os._exit`` because a normal shutdown would block on the same stuck
        streams, and writes its last line straight to fd 2 because a logging
        handler lock may itself be wedged.

        Args:
            num_stuck: Number of stuck releases, for the log line.
        """
        try:
            os.write(
                2,
                (
                    f"LMCache CRITICAL: exiting with code {STUCK_RELEASE_EXIT_CODE}: "
                    f"{num_stuck} IPC release(s) stuck and "
                    "--ipc-release-stuck-action=exit. The restart frees their "
                    "GPU memory.\n"
                ).encode(),
            )
        finally:
            os._exit(STUCK_RELEASE_EXIT_CODE)

    def release_status(self) -> dict:
        """Return counters and pending state of the IPC release path.

        Returns:
            A dict with the pending/stuck/held-back instance IDs, the
            registered instances blocking unmaps with a stale wait, and the
            lifetime counters.
        """
        with self._release_lock:
            status: dict[str, Any] = dict(self._release_counters)
            pending = list(self._pending_releases.values())
            status["pending_instance_ids"] = [p.instance_id for p in pending]
            status["stuck_instance_ids"] = [p.instance_id for p in pending if p.stuck]
            status["held_back_instance_ids"] = [
                p.instance_id for p in pending if p.held_back
            ]
            status["force_blocked_reasons"] = {
                str(p.instance_id): p.force_blocked for p in pending if p.force_blocked
            }
            del pending
            status["unmapping"] = self._unmapping
            status["stuck_action"] = self._release_config.stuck_action
            status["max_force_wait_age_seconds"] = self._max_force_wait_age
        status["stale_wait_instance_ids"] = sorted(
            iid
            for ids in self._stale_registered_devices(time.monotonic()).values()
            for iid in ids
        )
        return status

    def report_status(self) -> dict:
        """Return GPU transfer module status information.

        Returns:
            A dict containing registered GPU instance IDs and
            per-instance KV cache layout metadata.
        """
        registered_gpu_ids: list[int] = []
        cache_context_meta: dict[str, dict] = {}

        for instance_id, entry in self.context_entries_snapshot().items():
            registered_gpu_ids.append(instance_id)
            ctx = entry.cache_context
            cache_context_meta[str(instance_id)] = {
                "model_name": entry.model_name,
                "world_size": entry.world_size,
                "kv_cache_layout": ctx.report_status(),
            }

        return {
            "registered_gpu_ids": registered_gpu_ids,
            "cache_context_meta": cache_context_meta,
            "ipc_release": self.release_status(),
        }

    def close(self) -> None:
        """Release GPU resources owned by this module.

        Pending releases get one bounded attempt: their imported events are
        forced right away (the server is going down, so no grace), and any
        stream still busy after the stuck timeout is left mapped for process
        exit to reclaim instead of hanging shutdown.
        """
        with self._release_lock:
            self._closing = True
            release_thread = self._release_thread
        if release_thread is not None:
            release_thread.stop()

        with self._lock:
            released = list(self._cache_contexts.items())
            self._cache_contexts.clear()
        self._release_entries(released)

        cfg = self._release_config
        with self._release_lock:
            for item in self._pending_releases.values():
                item.started -= cfg.drain_grace_seconds
        deadline = time.monotonic() + cfg.stuck_timeout_seconds
        while True:
            remaining_s = max(0.0, deadline - time.monotonic())
            # A release thread wedged mid-cycle must not wedge shutdown too.
            self._release_cycle(allow_exit=False, lock_timeout=remaining_s)
            with self._release_lock:
                leftover = sorted(
                    p.instance_id for p in self._pending_releases.values()
                )
                unmapping = self._unmapping
            if (not leftover and not unmapping) or time.monotonic() >= deadline:
                break
            time.sleep(cfg.poll_interval_seconds)
        # Stop the drain thread only now: its callbacks drop the imported-event
        # holds the drain checks above rely on. Still before
        # storage_manager.close(), so in-flight completions reach a live
        # storage manager.
        self._device_host_func_dispatcher.stop()
        # Stop the unmapper after its queued work; never join it, since an
        # unmap stuck on the GPU must not hang shutdown.
        # Clear the rollback hook first, then refuse new unmap jobs, so no job
        # lands on a queue after its stop sentinel.
        set_failed_init_closer(None, replacing=self._installed_hooks[0])
        set_device_alloc_guard(None, replacing=self._installed_hooks[1])
        with self._release_lock:
            self._unmap_stopped = True
            unmap_queues = list(self._unmap_queues.values())
        for unmap_queue in unmap_queues:
            unmap_queue.put(None)
        if leftover or unmapping:
            logger.error(
                "Shutting down with IPC release(s) still pending for instance(s) "
                "%s and %d unmap(s) in progress; process exit reclaims their "
                "mappings.",
                leftover,
                unmapping,
            )

    @request_handler()
    def register_kv_cache(
        self,
        instance_id: int,
        kv_caches: KVCache,
        model_name: str,
        world_size: int,
        engine_type: EngineType,
        layout_hints: LayoutHints,
        engine_group_infos: list[EngineGroupInfo],
    ) -> None:
        """Register the KV cache tensors for a given GPU instance ID.

        Args:
            instance_id: The GPU instance ID (such as PID).
            kv_caches: The KV cache tensor wrappers from the
                serving engine.
            model_name: The name of the model associated with this KV cache.
            world_size: The world size associated with this KV cache.
            engine_type: Which serving engine produced the caches.
                Forwarded to GPUCacheContext for format detection.
            layout_hints: See LayoutHints.  Forwarded to
                GPUCacheContext for GPU KV format detection.
            engine_group_infos: Engine-neutral KV cache group metadata
                (already msgspec-decoded by the message queue).
        """
        now = time.monotonic()
        # NOOP-register: an already-registered instance (e.g. a recovering
        # worker re-registering on its first ping) refreshes its last-seen
        # time so a stale entry is not reaped right after recovery. REGISTER
        # is SYNC-serialized on the MQ main loop, so it is the sole inserter.
        with self._lock:
            existing = self._cache_contexts.get(instance_id)
            if existing is not None:
                existing.last_seen = now
                logger.info(
                    "Instance %d already registered; refreshing liveness",
                    instance_id,
                )
                return

        # Build the context and layout descriptor outside the lock.
        cache_context = create_cache_context(
            kv_caches,
            self._ctx.chunk_size,
            layout_hints=layout_hints or None,
            engine_group_infos=engine_group_infos,
            engine_type=engine_type,
            separate_object_groups=self._ctx.separate_object_groups,
            full_sw_kv=self._ctx.full_sw_kv,
        )
        try:
            event_backend = self._register_layout(cache_context, model_name, world_size)
        except BaseException:
            # The context holds the worker's IPC mappings; a failed REGISTER
            # must not pin its KV pool. Unmap off the MQ loop, like a release.
            self._submit_unmap(
                [
                    ContextEntry(
                        cache_context=cache_context,
                        model_name=model_name,
                        world_size=world_size,
                    )
                ]
            )
            del cache_context
            raise

        with self._release_lock:
            generation = self._next_generation
            self._next_generation += 1
        with self._lock:
            self._cache_contexts[instance_id] = ContextEntry(
                cache_context=cache_context,
                model_name=model_name,
                world_size=world_size,
                last_seen=now,
                has_liveness_signal=False,
                event_backend=event_backend,
                generation=generation,
            )

        logger.info(
            "Registered KV cache for GPU ID %d with %d layers",
            instance_id,
            cache_context.num_layers,
        )

    def _register_layout(
        self, cache_context: BaseCacheContext, model_name: str, world_size: int
    ) -> EventIPCBackend:
        """Check event support and register the layout of a new context.

        Args:
            cache_context: The context just built by REGISTER.
            model_name: The model of the registration.
            world_size: The world size of the registration.

        Returns:
            The event backend of the context's device.
        """
        kv_groups_manager = cache_context.kv_layer_groups_manager
        num_object_groups = kv_groups_manager.num_object_groups
        event_backend = get_event_ipc_backend(cache_context.device)
        event_backend.check_event_support(cache_context.device)
        layout_desc = get_layout_desc(
            cache_context, self._ctx.chunk_size, object_group_id=0
        )
        # One layout per object group, also in the single-group case: no
        # None special-casing downstream (group 0 maps to the merged layout).
        group_layout_descs = {
            gid: get_layout_desc(
                cache_context, self._ctx.chunk_size, object_group_id=gid
            )
            for gid in range(num_object_groups)
        }
        attn_desc = kv_groups_manager.get_attn_desc()
        self._ctx.layout_desc_registry.register(
            model_name,
            world_size,
            layout_desc,
            attn_desc,
            group_layout_descs=group_layout_descs,
        )
        return event_backend

    @request_handler()
    def unregister_kv_cache(self, instance_id: int) -> None:
        """Unregister the KV cache tensors for a given GPU instance ID.

        Args:
            instance_id: The GPU instance ID (such as PID).
        """
        with self._lock:
            popped = [
                e
                for e in (self._cache_contexts.pop(instance_id, None),)
                if e is not None
            ]
        if not popped:
            logger.warning(
                "No registered GPU context found for instance ID %d", instance_id
            )
            return

        # No scalar binding: `popped` must stay the only reference so
        # _release_entries' reclaim actually unmaps the IPC segments.
        self._release_entries([(instance_id, popped.pop())])
        logger.info("Unregistered KV cache for GPU ID %d", instance_id)

    @request_handler(
        HandlerType.BLOCKING,
        requires_client_affinity=True,
    )
    @_lmcache_nvtx_annotate
    def store(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
        gpu_block_ids: list[list[int]],
        event_ipc_handle: bytes,
    ) -> tuple[bytes, bool]:
        """Store the GPU KV cache blocks to CPU; see store_with_chunk_mask."""
        handle, ok, _ = self.store_with_chunk_mask(
            key, instance_id, gpu_block_ids, event_ipc_handle
        )
        return handle, ok

    def store_with_chunk_mask(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
        gpu_block_ids: list[list[int]],
        event_ipc_handle: bytes,
    ) -> tuple[bytes, bool, list[bool]]:
        """Store the GPU KV cache blocks to CPU.

        Args:
            key: The IPC key for the KV cache blocks.
                Must have worker_id != None (worker store operation).
            instance_id: The GPU instance ID (such as PID).
            gpu_block_ids: GPU block IDs to store, indexed by LMCache KV
                group index.
            event_ipc_handle: The IPC handle of the event to wait on.

        Returns:
            A tuple where the first element is the IPC handle of the event
            that signals the completion of the store operation, and the second
            element indicates whether the store operation completed without a
            fatal error (not whether every requested chunk was stored; see
            Notes). The event handle is empty when no device work was submitted.
            The third element marks per chunk whether every object group
            committed it.

        Raises:
            RuntimeError: If the backend does not support IPC event handles.

        Notes:
            All-or-nothing. If ``gpu_block_ids`` do not fully cover every chunk
            ``key`` resolves to for every LMCache group (e.g. a caller/protocol
            bug), a copy fails, or completion ownership is invalid, the whole
            store is skipped and nothing is committed; a subsequent retrieve misses
            and the engine recomputes. The boolean result reports whether the
            store completed without such a failure.
            Failed copies or completion preparation retain staging reservations
            under the existing write-TTL rules; queued GPU writes may still
            reference those buffers.
        """
        st = time.perf_counter()

        entry = self.get_and_touch_context_entry(instance_id)
        if entry is None:
            # The worker can reconnect to a replacement server before its next
            # registration probe. No device work was submitted in that window,
            # so return an empty completion-event handle and a terminal False
            # response instead of leaving the MQ future unanswered. Echoing the
            # producer handle would make the originating process import its own
            # IPC event, which is invalid on HIP.
            logger.warning(
                "Rejecting STORE for unregistered GPU instance ID %d",
                instance_id,
            )
            return b"", False, []
        cache_context = entry.cache_context
        model_name = entry.model_name
        event_backend = entry.event_backend
        if event_backend is None:
            raise RuntimeError("Registered cache context has no event backend")

        num_object_groups = cache_context.kv_layer_groups_manager.num_object_groups
        obj_keys_per_obj_group = self._ctx.resolve_obj_keys(
            key, list(range(num_object_groups))
        )
        num_chunks = len(obj_keys_per_obj_group[0])

        # NOTE: different engine groups may have different block sizes, so
        # ``blocks_per_chunk[i]`` is the number of blocks in one chunk for
        # group ``i``.
        blocks_per_chunk = [
            cache_context.calculate_num_blocks(self._ctx.chunk_size, group_idx)
            for group_idx in range(
                cache_context.kv_layer_groups_manager.num_kernel_groups
            )
        ]

        with (
            torch_dev.device(cache_context.device),
            torch_dev.stream(cache_context.stream),
        ):
            event = event_backend.create_event(cache_context.device)

            # Fail closed: every LMCache group must have block IDs covering all
            # chunks. A short list (e.g. a caller/protocol bug) would otherwise
            # drive the transfer kernel to read out-of-bounds GPU memory, so skip
            # the whole store and commit nothing rather than caching a partial or
            # garbage entry. A later request can store it once the block IDs are
            # complete. Checked on the raw block ids, before cutting drops the
            # per-chunk blocks that sliding-window groups do not need.
            if any(
                len(group_block_ids) < num_chunks * bpc
                for group_block_ids, bpc in zip(
                    gpu_block_ids, blocks_per_chunk, strict=True
                )
            ):
                logger.warning(
                    "STORE block ID underflow for request_id=%s: each group needs "
                    "num_chunks * blocks_per_chunk block IDs for %d chunks "
                    "(per-group blocks_per_chunk=%s); skipping the store.",
                    key.request_id,
                    num_chunks,
                    blocks_per_chunk,
                )
                event_backend.record_event(event, cache_context.stream)
                return (
                    event_backend.export_event(event, cache_context.device),
                    False,
                    [],
                )

            # Chunks whose block ids are all the null block (e.g. align-mode
            # Mamba chunks holding no real state) carry no valid KV and must not
            # be committed. Computed on the raw block ids before downsampling
            # mutates them.
            null_block_id = self._ctx.null_block_id
            skipped_chunks = all_null_chunk_masks(
                gpu_block_ids,
                cache_context.kv_layer_groups_manager.object_groups,
                blocks_per_chunk,
                num_chunks,
                null_block_id,
            )

            block_ids_per_group_gpu = downsample_and_stage_block_ids(
                cache_context, gpu_block_ids
            )

            producer_event = event_backend.import_event(
                event_ipc_handle, cache_context.device
            )
            event_backend.wait_event(producer_event, cache_context.stream)
            import_token = cache_context.hold_imported_event(producer_event)
            submit_callback_to_stream(
                cache_context.cupy_stream,
                "release_imported_event",
                (instance_id, import_token),
            )

            # CPU-synchronous sentinel: a GPU store is about to be enqueued.
            # Must be published via publish() (not publish_on_stream) so the
            # drain thread sees it before MP_REQUEST_END can race MP_STORE_END.
            self._ctx.event_bus.publish(
                Event(
                    event_type=EventType.MP_STORE_SUBMITTED,
                    session_id=key.request_id,
                    metadata={"device": str(cache_context.device)},
                )
            )

            # Worker 0 only: bindings depend on token content alone, so one
            # report covers every rank's keys. Published before finish_write
            # is enqueued so the token bindings precede the write-finished
            # events on the bus.
            if key.worker_id == 0 and self._ctx.event_bus.has_subscribers(
                EventType.MP_TOKENS
            ):
                self._publish_token_bindings(key, obj_keys_per_obj_group[0])

            transfer_key = next_transfer_key(key.request_id)
            self._ctx.event_bus.publish_on_stream(
                cache_context.cupy_stream,
                Event(
                    event_type=EventType.MP_STORE_START,
                    session_id=key.request_id,
                    metadata={
                        "device": str(cache_context.device),
                        "engine_id": instance_id,
                        "model_name": model_name,
                        "transfer_key": transfer_key,
                    },
                ),
            )

            reserved_dict: dict[ObjectKey, MemoryObj] = {}
            all_dict: dict[ObjectKey, MemoryObj] = {}
            total_bytes: int = 0
            store_succeeded = False
            try:
                for obj_group_id in range(num_object_groups):
                    obj_keys = obj_keys_per_obj_group[obj_group_id]
                    skip_mask = skipped_chunks[obj_group_id]
                    keys_to_reserve = [
                        k for i, k in enumerate(obj_keys) if not skip_mask[i]
                    ]
                    layout_desc = get_layout_desc(
                        cache_context,
                        self._ctx.chunk_size,
                        object_group_id=obj_group_id,
                    )
                    reserved_dict = self._ctx.storage_manager.reserve_write(
                        keys_to_reserve, layout_desc
                    )
                    all_dict.update(reserved_dict)
                    if reserved_dict:
                        total_bytes += next(
                            iter(reserved_dict.values())
                        ).get_size() * len(reserved_dict)

                    # Keys not in reserved_dict (all-null chunks skipped above, or
                    # skipped by the storage manager) become None entries; the
                    # helper skips them for D2H.
                    memory_objs: list[MemoryObj | None] = [
                        reserved_dict.get(obj_key) for obj_key in obj_keys
                    ]

                    # NOTE: batch_size must stay 1 for store.
                    transfer_kv_per_object_group(
                        cache_context,
                        block_ids_per_group_gpu,
                        memory_objs,
                        object_group_id=obj_group_id,
                        batch_size=1,
                        skip_first_n_tokens=0,
                        direction=lmcache_native.TransferDirection.D2H,
                        transfer_key=transfer_key,
                        block_ids_host=gpu_block_ids,
                    )

                completion = (
                    self._ctx.storage_manager.prepare_write_completion(all_dict)
                    if all_dict
                    else []
                )
                store_succeeded = True
            except Exception:
                logger.exception("Cannot store keys due to exception")
            finally:
                event_backend.record_event(event, cache_context.stream)
                # Fail closed: commit the reserved objects only when every chunk
                # copied successfully; otherwise the whole store is skipped.
                stored_count = len(all_dict) if store_succeeded else 0
                if stored_count:
                    submit_callback_to_stream(
                        cache_context.cupy_stream,
                        "finish_write_unless_poisoned",
                        (entry.generation, completion),
                    )
                else:
                    total_bytes = 0
                num_tokens = num_chunks * self._ctx.chunk_size if stored_count else 0
                self._ctx.event_bus.publish_on_stream(
                    cache_context.cupy_stream,
                    Event(
                        event_type=EventType.MP_STORE_END,
                        session_id=key.request_id,
                        metadata={
                            "stored_count": stored_count,
                            "device": str(cache_context.device),
                            "engine_id": instance_id,
                            "model_name": model_name,
                            "total_bytes": total_bytes,
                            "num_tokens": num_tokens,
                            "transfer_key": transfer_key,
                        },
                    ),
                )

        ed = time.perf_counter()
        if stored_count:
            logger.info(
                "Stored %d tokens in %.3f seconds",
                num_chunks * self._ctx.chunk_size,
                ed - st,
            )

        # A chunk is stored only when every object group committed its key.
        stored_mask = [
            store_succeeded
            and all(keys[i] in all_dict for keys in obj_keys_per_obj_group)
            for i in range(num_chunks)
        ]
        return (
            event_backend.export_event(event, cache_context.device),
            store_succeeded,
            stored_mask,
        )

    @request_handler(
        HandlerType.BLOCKING,
        requires_client_affinity=True,
    )
    @_lmcache_nvtx_annotate
    def retrieve(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
        gpu_block_ids: list[list[int]],
        event_ipc_handle: bytes,
        skip_first_n_tokens: int = 0,
    ) -> tuple[bytes, bool]:
        """Retrieve the CPU KV cache and put into GPU blocks.

        Args:
            key: The IPC key for the KV cache blocks.
                Must have worker_id != None (worker retrieve operation).
            instance_id: The GPU instance ID (such as PID).
            gpu_block_ids: GPU block IDs to retrieve into, indexed by LMCache
                KV group index.
            event_ipc_handle: The IPC handle of the event to wait on.
            skip_first_n_tokens: Number of tokens to skip writing at
                the start of the retrieve range. This avoids overwriting
                APC-shared GPU blocks that may be read concurrently by other
                requests.

        Returns:
            A tuple where the first element is the IPC handle of the event
            that signals the completion of the retrieve operation, and the
            second element indicates whether the key was successfully retrieved.
            The event handle is empty when no device work was submitted.

        Raises:
            RuntimeError: If the backend does not support IPC event handles.
        """
        st = time.perf_counter()

        entry = self.get_and_touch_context_entry(instance_id)
        if entry is None:
            # See store(): there is no completion event because no device work
            # was submitted. The False result lets the caller recover or
            # recompute without importing its own producer event.
            logger.warning(
                "Rejecting RETRIEVE for unregistered GPU instance ID %d",
                instance_id,
            )
            try:
                self._release_failed_retrieve_locks(key, instance_id)
            except Exception:
                # A cleanup failure must never suppress the terminal response:
                # the client otherwise waits forever because blocking-handler
                # exceptions are only logged by the MQ server.
                logger.exception(
                    "Failed to release RETRIEVE locks for unregistered "
                    "GPU instance ID %d",
                    instance_id,
                )
            return b"", False
        cache_context = entry.cache_context
        model_name = entry.model_name
        event_backend = entry.event_backend
        if event_backend is None:
            raise RuntimeError("Registered cache context has no event backend")

        num_object_groups = cache_context.kv_layer_groups_manager.num_object_groups
        obj_keys_per_obj_group = self._ctx.resolve_obj_keys(
            key, list(range(num_object_groups))
        )
        num_chunks = len(obj_keys_per_obj_group[0])

        # CPU-synchronous sentinel: a GPU retrieve is about to be enqueued.
        # Must be published via publish() (not publish_on_stream) so the
        # drain thread sees it before MP_REQUEST_END can race MP_RETRIEVE_END.
        self._ctx.event_bus.publish(
            Event(
                event_type=EventType.MP_RETRIEVE_SUBMITTED,
                session_id=key.request_id,
                metadata={"device": str(cache_context.device)},
            )
        )

        transfer_key = next_transfer_key(key.request_id)
        self._ctx.event_bus.publish_on_stream(
            cache_context.cupy_stream,
            Event(
                event_type=EventType.MP_RETRIEVE_START,
                session_id=key.request_id,
                metadata={
                    "device": str(cache_context.device),
                    "engine_id": instance_id,
                    "model_name": model_name,
                    "transfer_key": transfer_key,
                },
            ),
        )

        blocks_per_chunk = [
            cache_context.calculate_num_blocks(self._ctx.chunk_size, group_idx)
            for group_idx in range(
                cache_context.kv_layer_groups_manager.num_kernel_groups
            )
        ]

        with (
            torch_dev.device(cache_context.device),
            torch_dev.stream(cache_context.stream),
        ):
            event = event_backend.create_event(cache_context.device)

            # Fail closed: a short block-id list would drive the transfer
            # kernel to write out-of-bounds GPU memory. Checked on the raw
            # block ids, before cutting drops the per-chunk blocks that
            # sliding-window groups do not need.
            if any(
                len(group_block_ids) < num_chunks * bpc
                for group_block_ids, bpc in zip(
                    gpu_block_ids, blocks_per_chunk, strict=True
                )
            ):
                logger.error(
                    "RETRIEVE block ID underflow for request_id=%s: each group "
                    "needs num_chunks * blocks_per_chunk block IDs for %d "
                    "chunks (per-group blocks_per_chunk=%s); skipping the "
                    "retrieve.",
                    key.request_id,
                    num_chunks,
                    blocks_per_chunk,
                )
                event_backend.record_event(event, cache_context.stream)
                return event_backend.export_event(event, cache_context.device), False

            # Cut and stage all block_ids to GPU once before the transfer
            block_ids_per_group_gpu = downsample_and_stage_block_ids(
                cache_context, gpu_block_ids
            )
            producer_event = event_backend.import_event(
                event_ipc_handle, cache_context.device
            )
            event_backend.wait_event(producer_event, cache_context.stream)
            import_token = cache_context.hold_imported_event(producer_event)
            submit_callback_to_stream(
                cache_context.cupy_stream,
                "release_imported_event",
                (instance_id, import_token),
            )

            # Per object group, the prefetch only locked the in-window suffix
            # (the last ``num_chunks_in_sw`` chunks; the whole prefix for full
            # attention, where the value is < 0). Read and transfer only those.
            # Aux (connector-private) groups are never served by the
            # std retrieve: the lookup does not lock their keys and their
            # block-id entry is a placeholder -- reading them would be an
            # unlocked read of a plane nobody consumes here.
            attn_desc = cache_context.kv_layer_groups_manager.get_attn_desc()
            skipped_groups = {
                g for g, kind in enumerate(attn_desc.group_kinds) if kind == "aux"
            }
            group_skips = [
                0 if window < 0 else max(0, num_chunks - window)
                for window in attn_desc.num_chunks_in_sw
            ]
            expected_retained = sum(
                num_chunks - skip
                for g, skip in enumerate(group_skips)
                if g not in skipped_groups
            )

            prefetched_keys: list[ObjectKey] = []
            read_owners = self._ctx.get_read_owners(key.request_id)
            retained_keys = [
                obj_key
                for group, keys in enumerate(obj_keys_per_obj_group)
                if group not in skipped_groups
                for obj_key in keys[group_skips[group] :]
            ]
            # Capture every retained lock before enqueuing GPU work, including
            # groups that a later transfer failure may leave unvisited.
            read_completion = self._ctx.storage_manager.prepare_read_completion(
                retained_keys, read_owners
            )
            total_bytes = 0
            retrieve_succeeded = True
            try:
                for obj_group_id in range(num_object_groups):
                    if obj_group_id in skipped_groups:
                        continue
                    skip = group_skips[obj_group_id]
                    in_window_keys = obj_keys_per_obj_group[obj_group_id][skip:]
                    found_keys, window_objs = self._ctx.storage_manager.unsafe_read(
                        in_window_keys, l1_owners=read_owners
                    )
                    if not window_objs or found_keys != in_window_keys:
                        logger.error("Some keys not found during retrieve!")
                        retrieve_succeeded = False
                        break

                    total_bytes += sum(mo.get_size() for mo in window_objs)
                    # None-pad the skipped prefix so block-id slicing is unchanged.
                    memory_objs: list[MemoryObj | None] = [None] * skip + window_objs
                    transfer_kv_per_object_group(
                        cache_context,
                        block_ids_per_group_gpu,
                        memory_objs,
                        object_group_id=obj_group_id,
                        batch_size=cache_context.max_batch_size,
                        skip_first_n_tokens=skip_first_n_tokens,
                        direction=lmcache_native.TransferDirection.H2D,
                        transfer_key=transfer_key,
                        block_ids_host=gpu_block_ids,
                    )
                    prefetched_keys.extend(in_window_keys)
            except Exception:
                logger.exception("Cannot retrieve keys due to exception")
                retrieve_succeeded = False
            finally:
                event_backend.record_event(event, cache_context.stream)
                if retained_keys:
                    submit_callback_to_stream(
                        cache_context.cupy_stream,
                        "finish_read_by_owner",
                        read_completion,
                    )
                num_tokens = (
                    num_chunks * self._ctx.chunk_size
                    if len(prefetched_keys) == expected_retained
                    else 0
                )
                self._ctx.event_bus.publish_on_stream(
                    cache_context.cupy_stream,
                    Event(
                        event_type=EventType.MP_RETRIEVE_END,
                        session_id=key.request_id,
                        metadata={
                            "retrieved_count": len(prefetched_keys),
                            "device": str(cache_context.device),
                            "engine_id": instance_id,
                            "model_name": model_name,
                            "cache_salt": key.cache_salt,
                            "total_bytes": total_bytes,
                            "num_tokens": num_tokens,
                            "transfer_key": transfer_key,
                        },
                    ),
                )
        if retrieve_succeeded:
            tokens_retrieved = num_chunks * self._ctx.chunk_size
            ed = time.perf_counter()
            logger.info(
                "Retrieved %d tokens in %.3f seconds",
                tokens_retrieved,
                ed - st,
            )

        return (
            event_backend.export_event(event, cache_context.device),
            retrieve_succeeded,
        )

    def _publish_token_bindings(
        self, key: IPCCacheServerKey, obj_keys: list[ObjectKey]
    ) -> None:
        """Publish one ``MP_TOKENS`` event for ``key``'s chunks.

        Pairs each complete chunk in ``[key.start, key.end)`` with its
        ObjectKey chunk hash and token position. Must be called at store
        submission, before the write-finished events reach the bus, so the
        cache-event subscriber can stamp them onto the STORE entries. A
        store that later fails leaves only unused cache entries.

        Args:
            key: The IPC key of the store being submitted.
            obj_keys: One ObjectKey per complete chunk, in chunk order.
        """
        # Complete chunks in [key.start, key.end) paired with the absolute
        # position of each chunk's first token. Prefix-chained chunk hashes
        # imply a position without revealing it, so it is reported here. A
        # trailing partial chunk has no stored KV to bind to.
        chunk_size = self._ctx.chunk_size
        token_ids = list(key.token_ids)
        effective_len = min(len(token_ids), key.end)
        num_complete = effective_len - effective_len % chunk_size
        token_offsets = list(range(key.start, num_complete, chunk_size))
        token_chunks = [
            token_ids[offset : offset + chunk_size] for offset in token_offsets
        ]
        if not token_chunks:
            return
        if len(obj_keys) != len(token_chunks):
            logger.warning(
                "Skipping token bindings for request %s: %d resolved keys "
                "vs %d complete chunks in [%d, %d)",
                key.request_id,
                len(obj_keys),
                len(token_chunks),
                key.start,
                key.end,
            )
            return
        self._ctx.event_bus.publish(
            Event(
                event_type=EventType.MP_TOKENS,
                session_id=key.request_id,
                metadata={
                    "chunk_hashes": [obj_key.chunk_hash for obj_key in obj_keys],
                    "token_chunks": token_chunks,
                    "token_offsets": token_offsets,
                },
            )
        )

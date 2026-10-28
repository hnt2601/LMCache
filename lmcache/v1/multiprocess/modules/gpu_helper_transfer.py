# SPDX-License-Identifier: Apache-2.0
"""LMCache-driven KV transfer with the device work in per-GPU helper processes.

The control plane stays here: object keys, L1 reservations and read locks, the
layout registry, liveness and metrics. The data plane -- the worker's KV IPC
mappings, event import/export and the copy kernels -- lives in one helper
process per registration (see :mod:`lmcache.v1.multiprocess.gpu_helper`),
restricted to the worker's GPU. The server process itself never opens a CUDA
context on a worker's GPU, and a helper's exit gives the whole GPU back.

Completion flows back as tokens: every transfer registers a pending record
(the write commit or the read-lock release to run) before the helper is asked
to queue the device work; the helper reports the token when the stream reaches
the end of that work, and the record is finalized here. A helper that dies or
stops answering is killed and its pending records are resolved the fail-closed
way: reads release their locks, writes are never committed.
"""

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass, field
from typing import Any, Sequence
import multiprocessing
import os
import threading
import time

# First Party
from lmcache.logging import init_logger
from lmcache.utils import EngineType, _lmcache_nvtx_annotate
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.storage_manager import L1WriteCompletion
from lmcache.v1.gpu_connector.utils import LayoutHints
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.mp_observability.event import Event, EventType, next_transfer_key
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey, KVCache
from lmcache.v1.multiprocess.engine_context import MPCacheServerContext
from lmcache.v1.multiprocess.engine_module import InstanceLivenessTarget
from lmcache.v1.multiprocess.gpu_helper import (
    HelperInit,
    ObjectRef,
    RegisterReply,
    RegisterRequest,
    TransferRequest,
    helper_main,
)
from lmcache.v1.multiprocess.group_view import EngineGroupInfo
from lmcache.v1.multiprocess.modules.lmcache_driven_transfer import (
    _isolated_ipc_mismatch,
    all_null_chunk_masks,
)
from lmcache.v1.multiprocess.modules.lookup import resolve_prefetched_obj_keys
from lmcache.v1.multiprocess.request_handler import HandlerType, request_handler
from lmcache.v1.platform.ipc_policy import is_isolated_ipc, is_use_vmm_api

logger = init_logger(__name__)


class HelperError(RuntimeError):
    """The helper of a registration died or stopped answering."""


@dataclass
class _Pending:
    """A transfer whose stream completion is still owed by the helper."""

    kind: str  # "write" | "read"
    completion: L1WriteCompletion
    end_event: EventType
    metadata: dict[str, Any]
    ok: bool | None = None
    done: bool = False
    finalized: bool = False
    created: float = field(default_factory=time.monotonic)


@dataclass
class HelperEntry:
    """One registration and the helper process that serves it."""

    instance_id: int
    model_name: str
    world_size: int
    device_uuid: str
    process: Any
    cmd_conn: Any
    evt_conn: Any
    generation: int
    reply: RegisterReply | None = None
    last_seen: float = 0.0
    has_liveness_signal: bool = False
    call_lock: threading.Lock = field(default_factory=threading.Lock)
    pending_lock: threading.Lock = field(default_factory=threading.Lock)
    pending: dict[int, _Pending] = field(default_factory=dict)
    gone: bool = False
    exiting: bool = False
    reader: threading.Thread | None = None


def _normalize_shm_name(shm_name: str) -> str:
    bare = shm_name.lstrip("/")
    if not bare.startswith("lmcache_l1_pool_"):
        bare = f"lmcache_l1_pool_{bare}"
    return bare


def l1_shm_pool_info(storage_manager: Any) -> tuple[str, int, int]:
    """Return ``(shm_name, base_ptr, size)`` of the single shm-backed L1 pool.

    Raises:
        ValueError: The L1 is not a single, eagerly allocated, shm-backed DRAM
            pool, which is what the helpers map.
    """
    managers = getattr(storage_manager, "_l1_managers_by_id", None) or {}
    if len(managers) != 1:
        raise ValueError(
            f"--gpu-helper needs exactly one L1 manager, got {len(managers)}"
        )
    l1 = next(iter(managers.values()))
    config = l1._config
    mem = config.memory_config
    reasons = []
    if getattr(config, "gds_l1_config", None) is not None:
        reasons.append("GDS L1")
    if mem.use_lazy:
        reasons.append("lazy allocation")
    if mem.devdax_path:
        reasons.append("Device-DAX")
    if mem.use_hugepages:
        reasons.append("hugepages")
    if not mem.shm_name:
        reasons.append("no shm_name")
    if reasons:
        raise ValueError(
            "--gpu-helper needs an eagerly allocated, shm-backed DRAM L1; "
            "this L1 has: " + ", ".join(reasons)
        )
    desc = l1.get_l1_memory_desc()
    return _normalize_shm_name(mem.shm_name), int(desc.ptr), int(desc.size)


class GPUHelperTransferModule(InstanceLivenessTarget):
    """Server side of the GPU-helper transfer path.

    Args:
        ctx: The shared engine context.
        call_timeout_seconds: How long a STORE/RETRIEVE waits for the helper to
            queue its device work before the helper is treated as stuck.
        register_timeout_seconds: How long REGISTER waits for the helper to
            start, map and pin the L1 pool, and build the cache context.
        exit_timeout_seconds: How long an orderly exit may take before the
            helper is killed.
        stuck_transfer_seconds: A transfer still incomplete after this long
            marks the helper's stream as stuck; the next STORE/RETRIEVE kills
            the helper. Defaults to 4x the call timeout, at least 60 s.
    """

    def __init__(
        self,
        ctx: MPCacheServerContext,
        *,
        call_timeout_seconds: float = 30.0,
        register_timeout_seconds: float = 600.0,
        exit_timeout_seconds: float = 10.0,
        stuck_transfer_seconds: float | None = None,
    ) -> None:
        self._ctx = ctx
        if is_use_vmm_api():
            raise ValueError(
                "--gpu-helper does not support the VMM IPC mode (use_vmm_api): "
                "its file descriptors cannot be forwarded to the helper processes"
            )
        shm_name, base_ptr, pool_size = l1_shm_pool_info(ctx.storage_manager)
        self._shm_name = shm_name
        self._base_ptr = base_ptr
        self._pool_size = pool_size
        self._call_timeout = call_timeout_seconds
        self._register_timeout = register_timeout_seconds
        self._exit_timeout = exit_timeout_seconds
        # A transfer whose completion is owed longer than this means the
        # helper's stream is stuck (a worker died mid-transfer and its event
        # never fires); the next request to that helper kills it instead of
        # queueing behind the stuck work.
        self._stuck_transfer_seconds = (
            stuck_transfer_seconds
            if stuck_transfer_seconds is not None
            else max(4.0 * call_timeout_seconds, 60.0)
        )
        self._entries: dict[int, HelperEntry] = {}
        self._lock = threading.Lock()
        self._spawn_lock = threading.Lock()
        self._next_token = 1
        self._next_generation = 1
        self._counters: dict[str, int] = {
            "helpers_started": 0,
            "helpers_exited": 0,
            "helpers_killed": 0,
            "helpers_died": 0,
            "released": 0,
            "forced": 0,
            "dropped_commits": 0,
            "failed_transfers": 0,
            "stuck_streams": 0,
        }
        self._mp = multiprocessing.get_context("spawn")
        logger.info(
            "GPU helper transfer: L1 pool %s (%d bytes at %#x) will be mapped by "
            "one helper process per registration",
            shm_name,
            pool_size,
            base_ptr,
        )

    # -- factory hooks (tests replace these) ----------------------------------

    def _spawn_process(
        self, instance_id: int, device_uuid: str, init: HelperInit
    ) -> tuple[Any, Any, Any]:
        """Start a helper restricted to ``device_uuid``; return (proc, cmd, evt)."""
        cmd_parent, cmd_child = self._mp.Pipe(duplex=True)
        evt_parent, evt_child = self._mp.Pipe(duplex=False)
        with self._spawn_lock:
            saved = os.environ.get("CUDA_VISIBLE_DEVICES")
            os.environ["CUDA_VISIBLE_DEVICES"] = f"GPU-{device_uuid}"
            try:
                proc = self._mp.Process(
                    target=helper_main,
                    args=(cmd_child, evt_child, init),
                    name=f"lmcache-gpu-helper-{instance_id}",
                    daemon=True,
                )
                proc.start()
            finally:
                if saved is None:
                    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
                else:
                    os.environ["CUDA_VISIBLE_DEVICES"] = saved
        cmd_child.close()
        evt_child.close()
        return proc, cmd_parent, evt_parent

    # -- helper lifecycle ------------------------------------------------------

    def _oldest_pending_age(self, entry: HelperEntry) -> float:
        now = time.monotonic()
        with entry.pending_lock:
            if not entry.pending:
                return 0.0
            return now - min(p.created for p in entry.pending.values())

    def _call(self, entry: HelperEntry, msg: tuple, timeout: float) -> tuple:
        with entry.call_lock:
            if entry.gone:
                raise HelperError("helper is gone")
            if msg[0] == "transfer":
                age = self._oldest_pending_age(entry)
                if age > self._stuck_transfer_seconds:
                    self._counters["stuck_streams"] += 1
                    raise HelperError(
                        f"a transfer has been incomplete for {age:.0f}s; the helper's "
                        "stream is stuck"
                    )
            try:
                entry.cmd_conn.send(msg)
                if not entry.cmd_conn.poll(timeout):
                    raise HelperError(
                        f"helper did not answer {msg[0]!r} within {timeout:.0f}s"
                    )
                reply = entry.cmd_conn.recv()
            except (EOFError, OSError, BrokenPipeError) as exc:
                raise HelperError(f"helper pipe failed: {exc}") from exc
        if reply[0] == "err":
            raise RuntimeError(reply[1])
        return reply

    def _reader_loop(self, entry: HelperEntry) -> None:
        while True:
            try:
                msg = entry.evt_conn.recv()
            except (EOFError, OSError):
                break
            if msg[0] == "done":
                self._on_done(entry, msg[1])
        if not entry.exiting:
            with self._lock:
                live = self._entries.get(entry.instance_id) is entry
            if live:
                logger.error(
                    "GPU helper of instance %d (pid %s) died; its worker must "
                    "re-register",
                    entry.instance_id,
                    getattr(entry.process, "pid", None),
                )
                self._counters["helpers_died"] += 1
                self._remove(entry)
                self._drop(entry, kill=True)

    def _remove(self, entry: HelperEntry) -> bool:
        with self._lock:
            if self._entries.get(entry.instance_id) is entry:
                del self._entries[entry.instance_id]
                return True
        return False

    def _drop(self, entry: HelperEntry, *, kill: bool) -> None:
        """Stop the helper (orderly unless ``kill``) and settle its pendings."""
        entry.exiting = True
        orderly = False
        if not kill and not entry.gone:
            try:
                reply = self._call(entry, ("exit",), self._exit_timeout)
                orderly = reply[0] == "bye"
            except Exception as exc:
                logger.warning(
                    "GPU helper of instance %d did not exit cleanly: %s",
                    entry.instance_id,
                    exc,
                )
        entry.gone = True
        proc = entry.process
        try:
            if orderly:
                proc.join(self._exit_timeout)
            if proc.is_alive():
                proc.kill()
                proc.join(self._exit_timeout)
                self._counters["helpers_killed"] += 1
            elif orderly:
                self._counters["helpers_exited"] += 1
        except Exception:
            logger.exception("Stopping GPU helper of instance %d failed", entry.instance_id)
        for conn in (entry.cmd_conn, entry.evt_conn):
            try:
                conn.close()
            except Exception:
                pass
        self._ctx.layout_desc_registry.unregister(entry.model_name, entry.world_size)
        self._counters["released"] += 1
        with entry.pending_lock:
            pendings = list(entry.pending.values())
            entry.pending.clear()
        for pending in pendings:
            self._settle(pending, committed=False)

    def _settle(self, pending: _Pending, *, committed: bool) -> None:
        """Run a pending transfer's completion exactly once."""
        if pending.finalized:
            return
        pending.finalized = True
        if pending.kind == "read":
            if pending.completion:
                self._ctx.storage_manager.finish_read_by_owner(pending.completion)
            retrieved = pending.metadata.get("expected_retained", 0) if committed else 0
            self._ctx.event_bus.publish(
                Event(
                    event_type=pending.end_event,
                    session_id=pending.metadata["request_id"],
                    metadata={**pending.metadata["event"], "retrieved_count": retrieved},
                )
            )
            return
        if committed and pending.completion:
            self._ctx.storage_manager.finish_write_by_owner(pending.completion)
            stored = pending.metadata["stored_count"]
            total_bytes = pending.metadata["total_bytes"]
            num_tokens = pending.metadata["num_tokens"]
        else:
            if pending.completion:
                self._counters["dropped_commits"] += 1
            stored, total_bytes, num_tokens = 0, 0, 0
        self._ctx.event_bus.publish(
            Event(
                event_type=pending.end_event,
                session_id=pending.metadata["request_id"],
                metadata={
                    **pending.metadata["event"],
                    "stored_count": stored,
                    "total_bytes": total_bytes,
                    "num_tokens": num_tokens,
                },
            )
        )

    def _on_done(self, entry: HelperEntry, token: int) -> None:
        with entry.pending_lock:
            pending = entry.pending.get(token)
            if pending is None:
                return
            pending.done = True
            ready = pending.ok is not None
            if ready:
                del entry.pending[token]
        if ready:
            self._settle(pending, committed=bool(pending.ok))

    def _set_ok(self, entry: HelperEntry, token: int, ok: bool) -> None:
        with entry.pending_lock:
            pending = entry.pending.get(token)
            if pending is None:
                return
            pending.ok = ok
            ready = pending.done
            if ready:
                del entry.pending[token]
        if ready:
            self._settle(pending, committed=ok)

    def _add_pending(self, entry: HelperEntry, pending: _Pending) -> int:
        with entry.pending_lock:
            token = self._next_token
            self._next_token += 1
            entry.pending[token] = pending
        return token

    def _fail_pending(self, entry: HelperEntry, token: int) -> None:
        with entry.pending_lock:
            pending = entry.pending.pop(token, None)
        if pending is not None:
            self._settle(pending, committed=False)

    def _helper_failed(self, entry: HelperEntry, what: str, exc: Exception) -> None:
        logger.error(
            "GPU helper of instance %d failed during %s: %s; killing it, the worker "
            "must re-register",
            entry.instance_id,
            what,
            exc,
        )
        self._counters["failed_transfers"] += 1
        if self._remove(entry):
            self._drop(entry, kill=True)

    # -- liveness --------------------------------------------------------------

    def get_and_touch_context_entry(self, instance_id: int) -> HelperEntry | None:
        now = time.monotonic()
        with self._lock:
            entry = self._entries.get(instance_id)
            if entry is not None:
                entry.last_seen = now
            return entry

    def touch_instance(self, instance_id: int) -> bool:
        now = time.monotonic()
        with self._lock:
            entry = self._entries.get(instance_id)
            if entry is None:
                return False
            entry.last_seen = now
            entry.has_liveness_signal = True
            return True

    def tracked_instance_count(self) -> int:
        with self._lock:
            return len(self._entries)

    def reap_stale_instances(
        self, reap_timeout_s: float, registration_grace_s: float
    ) -> list[int]:
        now = time.monotonic()
        with self._lock:
            stale = [
                entry
                for entry in self._entries.values()
                if now - entry.last_seen
                > (reap_timeout_s if entry.has_liveness_signal else registration_grace_s)
            ]
            for entry in stale:
                del self._entries[entry.instance_id]
        reaped: list[int] = []
        for entry in stale:
            logger.warning(
                "Reaped GPU instance %d: silent for %.1fs (pinged=%s); stopping its "
                "helper",
                entry.instance_id,
                now - entry.last_seen,
                entry.has_liveness_signal,
            )
            reaped.append(entry.instance_id)
            self._drop(entry, kill=False)
        return reaped

    def context_entries_snapshot(self) -> dict:
        """No in-process cache contexts exist on this path."""
        return {}

    def stuck_release_seconds(self) -> float | None:
        """Nothing is ever stuck here: a stuck helper is killed."""
        return None

    @property
    def context(self) -> MPCacheServerContext:
        return self._ctx

    # -- status ----------------------------------------------------------------

    def release_status(self) -> dict:
        with self._lock:
            entries = list(self._entries.values())
        status: dict[str, Any] = dict(self._counters)
        status.update(
            {
                "pending_instance_ids": [],
                "stuck_instance_ids": [],
                "stuck": 0,
                "held_back_instance_ids": [],
                "force_blocked_reasons": {},
                "unmapping": 0,
                "stuck_action": "kill_helper",
                "stuck_transfer_seconds": self._stuck_transfer_seconds,
                "max_force_wait_age_seconds": 0.0,
                "stuck_seconds": None,
                "stale_wait_instance_ids": [],
                "helpers": {
                    str(e.instance_id): {
                        "pid": getattr(e.process, "pid", None),
                        "device_uuid": e.device_uuid,
                        "pending_transfers": len(e.pending),
                        "oldest_pending_seconds": round(self._oldest_pending_age(e), 1),
                    }
                    for e in entries
                },
            }
        )
        return status

    def report_status(self) -> dict:
        with self._lock:
            entries = list(self._entries.values())
        return {
            "registered_gpu_ids": [e.instance_id for e in entries],
            "cache_context_meta": {
                str(e.instance_id): {
                    "model_name": e.model_name,
                    "world_size": e.world_size,
                    "helper_pid": getattr(e.process, "pid", None),
                    "device": e.reply.device if e.reply else None,
                    "kv_cache_layout": e.reply.kv_cache_layout if e.reply else {},
                }
                for e in entries
            },
            "ipc_release": self.release_status(),
        }

    def close(self) -> None:
        with self._lock:
            entries = list(self._entries.values())
            self._entries.clear()
        for entry in entries:
            self._drop(entry, kill=False)

    # -- handlers --------------------------------------------------------------

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
        """Start a helper on the worker's GPU and register its KV cache there."""
        now = time.monotonic()
        with self._lock:
            existing = self._entries.get(instance_id)
            if existing is not None:
                existing.last_seen = now
                logger.info(
                    "Instance %d already registered; refreshing liveness", instance_id
                )
                return
        mismatch = _isolated_ipc_mismatch(kv_caches, engine_type)
        if mismatch is not None:
            raise RuntimeError(f"Refusing to register instance {instance_id}: {mismatch}")
        if not kv_caches:
            raise RuntimeError(f"Instance {instance_id} registered no KV cache")
        device_uuid = str(kv_caches[0].device_uuid)
        init = HelperInit(
            instance_id=instance_id,
            shm_name=self._shm_name,
            pool_size=self._pool_size,
            chunk_size=self._ctx.chunk_size,
            separate_object_groups=self._ctx.separate_object_groups,
            full_sw_kv=self._ctx.full_sw_kv,
            isolated_ipc=is_isolated_ipc(),
            use_vmm_api=is_use_vmm_api(),
        )
        t0 = time.perf_counter()
        process, cmd_conn, evt_conn = self._spawn_process(instance_id, device_uuid, init)
        self._counters["helpers_started"] += 1
        with self._lock:
            generation = self._next_generation
            self._next_generation += 1
        entry = HelperEntry(
            instance_id=instance_id,
            model_name=model_name,
            world_size=world_size,
            device_uuid=device_uuid,
            process=process,
            cmd_conn=cmd_conn,
            evt_conn=evt_conn,
            generation=generation,
            last_seen=now,
        )
        try:
            if not evt_conn.poll(self._register_timeout):
                raise HelperError(
                    f"helper did not start within {self._register_timeout:.0f}s"
                )
            hello = evt_conn.recv()
            if hello[0] != "hello":
                raise HelperError(f"unexpected first helper message {hello!r}")
            reply = self._call(
                entry,
                (
                    "register",
                    RegisterRequest(
                        kv_caches=kv_caches,
                        model_name=model_name,
                        world_size=world_size,
                        engine_type=engine_type,
                        layout_hints=layout_hints,
                        engine_group_infos=engine_group_infos,
                    ),
                ),
                self._register_timeout,
            )
        except Exception as exc:
            logger.error(
                "GPU helper registration of instance %d failed: %s", instance_id, exc
            )
            entry.exiting = True
            entry.gone = True
            try:
                process.kill()
                process.join(self._exit_timeout)
            except Exception:
                pass
            for conn in (cmd_conn, evt_conn):
                try:
                    conn.close()
                except Exception:
                    pass
            self._counters["helpers_killed"] += 1
            raise
        entry.reply = reply[1]
        self._ctx.layout_desc_registry.register(
            model_name,
            world_size,
            entry.reply.group_layout_descs[0],
            entry.reply.attn_desc,
            group_layout_descs=entry.reply.group_layout_descs,
        )
        entry.reader = threading.Thread(
            target=self._reader_loop,
            args=(entry,),
            name=f"lmcache-gpu-helper-events-{instance_id}",
            daemon=True,
        )
        with self._lock:
            self._entries[instance_id] = entry
        entry.reader.start()
        logger.info(
            "Registered KV cache for GPU ID %d with %d layers in helper pid %s on %s "
            "(%.2fs)",
            instance_id,
            entry.reply.num_layers,
            getattr(process, "pid", None),
            entry.reply.device,
            time.perf_counter() - t0,
        )

    @request_handler()
    def unregister_kv_cache(self, instance_id: int) -> None:
        """Stop the registration's helper; its exit returns the GPU memory."""
        with self._lock:
            entry = self._entries.pop(instance_id, None)
        if entry is None:
            logger.warning(
                "No registered GPU context found for instance ID %d", instance_id
            )
            return
        self._drop(entry, kill=False)
        logger.info("Unregistered KV cache for GPU ID %d", instance_id)

    @request_handler(HandlerType.BLOCKING, requires_client_affinity=True)
    @_lmcache_nvtx_annotate
    def store(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
        gpu_block_ids: list[list[int]],
        event_ipc_handle: bytes,
    ) -> tuple[bytes, bool]:
        handle, ok, _ = self.store_with_chunk_mask(
            key, instance_id, gpu_block_ids, event_ipc_handle
        )
        return handle, ok

    @staticmethod
    def _object_ref(obj: MemoryObj) -> ObjectRef:
        meta = obj.meta
        return (int(meta.address), int(meta.phy_size), int(meta.fmt.value))

    def store_with_chunk_mask(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
        gpu_block_ids: list[list[int]],
        event_ipc_handle: bytes,
    ) -> tuple[bytes, bool, list[bool]]:
        """Store the worker's KV blocks into L1 through its helper.

        Semantics follow ``LMCacheDrivenTransferModule.store_with_chunk_mask``:
        all-or-nothing per store, reservations committed only when the helper
        reports that every copy was queued and the stream reached their end.
        """
        st = time.perf_counter()
        entry = self.get_and_touch_context_entry(instance_id)
        if entry is None or entry.reply is None:
            logger.warning("Rejecting STORE for unregistered GPU instance ID %d", instance_id)
            return b"", False, []
        reply = entry.reply
        num_object_groups = reply.num_object_groups
        obj_keys_per_obj_group = self._ctx.resolve_obj_keys(
            key, list(range(num_object_groups))
        )
        num_chunks = len(obj_keys_per_obj_group[0])
        blocks_per_chunk = reply.blocks_per_chunk
        if any(
            len(group_block_ids) < num_chunks * bpc
            for group_block_ids, bpc in zip(gpu_block_ids, blocks_per_chunk, strict=True)
        ):
            logger.warning(
                "STORE block ID underflow for request_id=%s: each group needs "
                "num_chunks * blocks_per_chunk block IDs for %d chunks "
                "(per-group blocks_per_chunk=%s); skipping the store.",
                key.request_id,
                num_chunks,
                blocks_per_chunk,
            )
            return b"", False, []
        skipped_chunks = all_null_chunk_masks(
            gpu_block_ids,
            reply.object_groups,
            blocks_per_chunk,
            num_chunks,
            self._ctx.null_block_id,
        )
        self._ctx.event_bus.publish(
            Event(
                event_type=EventType.MP_STORE_SUBMITTED,
                session_id=key.request_id,
                metadata={"device": reply.device},
            )
        )
        if key.worker_id == 0 and self._ctx.event_bus.has_subscribers(EventType.MP_TOKENS):
            self._publish_token_bindings(key, obj_keys_per_obj_group[0])
        transfer_key = next_transfer_key(key.request_id)
        event_meta = {
            "device": reply.device,
            "engine_id": instance_id,
            "model_name": entry.model_name,
            "transfer_key": transfer_key,
        }
        self._ctx.event_bus.publish(
            Event(
                event_type=EventType.MP_STORE_START,
                session_id=key.request_id,
                metadata=dict(event_meta),
            )
        )

        all_dict: dict[ObjectKey, MemoryObj] = {}
        groups: list[tuple[int, list[ObjectRef | None]]] = []
        total_bytes = 0
        for obj_group_id in range(num_object_groups):
            obj_keys = obj_keys_per_obj_group[obj_group_id]
            skip_mask = skipped_chunks[obj_group_id]
            keys_to_reserve = [k for i, k in enumerate(obj_keys) if not skip_mask[i]]
            reserved = self._ctx.storage_manager.reserve_write(
                keys_to_reserve, reply.group_layout_descs[obj_group_id]
            )
            all_dict.update(reserved)
            if reserved:
                total_bytes += next(iter(reserved.values())).get_size() * len(reserved)
            groups.append(
                (
                    obj_group_id,
                    [
                        None if (obj := reserved.get(k)) is None else self._object_ref(obj)
                        for k in obj_keys
                    ],
                )
            )
        completion = (
            self._ctx.storage_manager.prepare_write_completion(all_dict) if all_dict else []
        )
        pending = _Pending(
            kind="write",
            completion=completion,
            end_event=EventType.MP_STORE_END,
            metadata={
                "request_id": key.request_id,
                "event": event_meta,
                "stored_count": len(all_dict),
                "total_bytes": total_bytes,
                "num_tokens": num_chunks * self._ctx.chunk_size if all_dict else 0,
            },
        )
        token = self._add_pending(entry, pending)
        try:
            status = self._call(
                entry,
                (
                    "transfer",
                    TransferRequest(
                        token=token,
                        direction="store",
                        gpu_block_ids=gpu_block_ids,
                        event_ipc_handle=event_ipc_handle,
                        transfer_key=transfer_key,
                        groups=groups,
                    ),
                ),
                self._call_timeout,
            )
        except HelperError as exc:
            self._helper_failed(entry, "STORE", exc)
            return b"", False, []
        except Exception as exc:
            logger.error("STORE of request %s failed in the helper: %s", key.request_id, exc)
            self._counters["failed_transfers"] += 1
            self._fail_pending(entry, token)
            return b"", False, []
        _, handle, ok = status
        self._set_ok(entry, token, ok)
        if ok and all_dict:
            logger.info(
                "Stored %d tokens in %.3f seconds",
                num_chunks * self._ctx.chunk_size,
                time.perf_counter() - st,
            )
        stored_mask = [
            ok and all(keys[i] in all_dict for keys in obj_keys_per_obj_group)
            for i in range(num_chunks)
        ]
        return handle, ok, stored_mask

    @request_handler(HandlerType.BLOCKING, requires_client_affinity=True)
    @_lmcache_nvtx_annotate
    def retrieve(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
        gpu_block_ids: list[list[int]],
        event_ipc_handle: bytes,
        skip_first_n_tokens: int = 0,
    ) -> tuple[bytes, bool]:
        """Copy L1 objects into the worker's KV blocks through its helper."""
        st = time.perf_counter()
        entry = self.get_and_touch_context_entry(instance_id)
        if entry is None or entry.reply is None:
            logger.warning(
                "Rejecting RETRIEVE for unregistered GPU instance ID %d", instance_id
            )
            try:
                self._release_failed_retrieve_locks(key, instance_id)
            except Exception:
                logger.exception(
                    "Failed to release RETRIEVE locks for unregistered GPU instance ID %d",
                    instance_id,
                )
            return b"", False
        reply = entry.reply
        num_object_groups = reply.num_object_groups
        obj_keys_per_obj_group = self._ctx.resolve_obj_keys(
            key, list(range(num_object_groups))
        )
        num_chunks = len(obj_keys_per_obj_group[0])
        self._ctx.event_bus.publish(
            Event(
                event_type=EventType.MP_RETRIEVE_SUBMITTED,
                session_id=key.request_id,
                metadata={"device": reply.device},
            )
        )
        transfer_key = next_transfer_key(key.request_id)
        event_meta = {
            "device": reply.device,
            "engine_id": instance_id,
            "model_name": entry.model_name,
            "transfer_key": transfer_key,
        }
        self._ctx.event_bus.publish(
            Event(
                event_type=EventType.MP_RETRIEVE_START,
                session_id=key.request_id,
                metadata=dict(event_meta),
            )
        )
        blocks_per_chunk = reply.blocks_per_chunk
        if any(
            len(group_block_ids) < num_chunks * bpc
            for group_block_ids, bpc in zip(gpu_block_ids, blocks_per_chunk, strict=True)
        ):
            logger.error(
                "RETRIEVE block ID underflow for request_id=%s: each group needs "
                "num_chunks * blocks_per_chunk block IDs for %d chunks "
                "(per-group blocks_per_chunk=%s); skipping the retrieve.",
                key.request_id,
                num_chunks,
                blocks_per_chunk,
            )
            return b"", False

        attn_desc = reply.attn_desc
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
        read_owners = self._ctx.get_read_owners(key.request_id)
        retained_keys = [
            obj_key
            for group, keys in enumerate(obj_keys_per_obj_group)
            if group not in skipped_groups
            for obj_key in keys[group_skips[group] :]
        ]
        read_completion = self._ctx.storage_manager.prepare_read_completion(
            retained_keys, read_owners
        )
        pending = _Pending(
            kind="read",
            completion=read_completion,
            end_event=EventType.MP_RETRIEVE_END,
            metadata={
                "request_id": key.request_id,
                "expected_retained": expected_retained,
                "event": {
                    **event_meta,
                    "cache_salt": key.cache_salt,
                    "total_bytes": 0,
                    "num_tokens": num_chunks * self._ctx.chunk_size,
                },
            },
        )
        groups: list[tuple[int, list[ObjectRef | None]]] = []
        total_bytes = 0
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
                # Nothing was queued: release the captured locks right away.
                pending.metadata["event"]["total_bytes"] = total_bytes
                self._settle(pending, committed=False)
                return b"", False
            total_bytes += sum(mo.get_size() for mo in window_objs)
            groups.append(
                (
                    obj_group_id,
                    [None] * skip + [self._object_ref(mo) for mo in window_objs],
                )
            )
        pending.metadata["event"]["total_bytes"] = total_bytes
        token = self._add_pending(entry, pending)
        try:
            status = self._call(
                entry,
                (
                    "transfer",
                    TransferRequest(
                        token=token,
                        direction="retrieve",
                        gpu_block_ids=gpu_block_ids,
                        event_ipc_handle=event_ipc_handle,
                        transfer_key=transfer_key,
                        groups=groups,
                        skip_first_n_tokens=skip_first_n_tokens,
                    ),
                ),
                self._call_timeout,
            )
        except HelperError as exc:
            self._helper_failed(entry, "RETRIEVE", exc)
            return b"", False
        except Exception as exc:
            logger.error(
                "RETRIEVE of request %s failed in the helper: %s", key.request_id, exc
            )
            self._counters["failed_transfers"] += 1
            self._fail_pending(entry, token)
            return b"", False
        _, handle, ok = status
        self._set_ok(entry, token, ok)
        if ok:
            logger.info(
                "Retrieved %d tokens in %.3f seconds",
                num_chunks * self._ctx.chunk_size,
                time.perf_counter() - st,
            )
        return handle, ok

    # -- helpers copied from the in-process module ------------------------------

    def _release_failed_retrieve_locks(
        self, key: IPCCacheServerKey, instance_id: int
    ) -> None:
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
            self._ctx, key, hit_chunks, locked_gids, group_windows=group_windows
        )
        if not session.claim_failed_retrieve_release(instance_id, key, lookup_generation):
            return
        if obj_keys:
            self._ctx.storage_manager.finish_read_prefetched(
                obj_keys,
                read_locks=1,
                l1_owners=self._ctx.get_read_owners(key.request_id),
            )

    def _publish_token_bindings(
        self, key: IPCCacheServerKey, obj_keys: Sequence[ObjectKey]
    ) -> None:
        chunk_size = self._ctx.chunk_size
        token_ids = list(key.token_ids)
        effective_len = min(len(token_ids), key.end)
        num_complete = effective_len - effective_len % chunk_size
        token_offsets = list(range(key.start, num_complete, chunk_size))
        token_chunks = [token_ids[o : o + chunk_size] for o in token_offsets]
        if not token_chunks:
            return
        if len(obj_keys) != len(token_chunks):
            logger.warning(
                "Skipping token bindings for request %s: %d resolved keys vs %d "
                "complete chunks in [%d, %d)",
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

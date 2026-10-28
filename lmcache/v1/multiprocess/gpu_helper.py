# SPDX-License-Identifier: Apache-2.0
"""Per-registration GPU helper process for the multiprocess server.

With ``--gpu-helper`` the server process never opens a CUDA context on a
worker's GPU. Everything that touches the worker's device -- importing its
KV-cache IPC handles, importing and recording events, the staging buffers and
the D2H / H2D copy kernels -- runs here, in a process spawned for that
registration and restricted to that one GPU through ``CUDA_VISIBLE_DEVICES``.
The helper maps the server's L1 pool (a POSIX shared-memory segment) and pins
it, so the copy kernels read and write the host memory the server allocates.

When the registration ends the helper exits and the driver reclaims its
context, so the GPU holds no memory of ours afterwards. A stuck stream (a
worker that died mid-transfer, LMCache#4676) is handled the same way: the
server kills the helper instead of waiting for the device to drain.

Protocol, pickled objects over :mod:`multiprocessing` pipes:

command pipe (server -> helper, exactly one reply per command)::

    ("register", RegisterRequest) -> ("ok", RegisterReply) | ("err", str)
    ("transfer", TransferRequest) -> ("ok", event_handle, ok) | ("err", str)
    ("exit",)                     -> ("bye",)

event pipe (helper -> server, asynchronous)::

    ("hello", pid)   once, after CUDA and the L1 mapping are ready
    ("done", token)  when a transfer's device work has completed
"""

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass, field
from multiprocessing import shared_memory
from multiprocessing.connection import Connection
from multiprocessing.resource_tracker import unregister
from typing import Any
import contextlib
import ctypes
import os
import threading
import time

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import AttnWindowDesc, MemoryLayoutDesc
from lmcache.v1.kv_layer_groups import ObjectGroupInfo

logger = init_logger(__name__)


# ----------------------------------------------------------------------------
# Wire types
# ----------------------------------------------------------------------------


@dataclass
class HelperInit:
    """Process-wide settings a helper needs before its first command."""

    instance_id: int
    shm_name: str
    pool_size: int
    chunk_size: int
    separate_object_groups: bool
    full_sw_kv: bool
    isolated_ipc: bool
    use_vmm_api: bool


@dataclass
class RegisterRequest:
    """The worker's REGISTER payload, forwarded verbatim."""

    kv_caches: Any
    model_name: str
    world_size: int
    engine_type: Any
    layout_hints: Any
    engine_group_infos: Any


@dataclass
class RegisterReply:
    """What the server needs from the helper's cache context."""

    helper_pid: int
    device: str
    num_layers: int
    num_object_groups: int
    num_kernel_groups: int
    group_layout_descs: dict[int, MemoryLayoutDesc]
    attn_desc: AttnWindowDesc
    blocks_per_chunk: list[int]
    object_groups: list[ObjectGroupInfo]
    max_batch_size: int
    kv_cache_layout: dict = field(default_factory=dict)
    init_seconds: float = 0.0
    l1_pinned: bool | None = None  # None while the background pin is running


#: One L1 object of a transfer: ``(address, phy_size, fmt_value)``; ``address``
#: is the offset in the L1 pool.
ObjectRef = tuple[int, int, int]


@dataclass
class TransferRequest:
    """One STORE (``store``) or RETRIEVE (``retrieve``)."""

    token: int
    direction: str
    gpu_block_ids: list[list[int]]
    event_ipc_handle: bytes
    transfer_key: str
    #: ``(object_group_id, objects)``; ``objects`` has one entry per chunk,
    #: ``None`` where the chunk is skipped.
    groups: list[tuple[int, list[ObjectRef | None]]]
    skip_first_n_tokens: int = 0


# ----------------------------------------------------------------------------
# Helper process
# ----------------------------------------------------------------------------


class _L1PoolView:
    """The server's L1 pool, mapped and pinned in this process."""

    def __init__(self, shm_name: str, pool_size: int) -> None:
        # First Party
        from lmcache.v1.platform import current_device_spec

        self._shm = shared_memory.SharedMemory(name=shm_name, create=False)
        # The server owns the segment; this process must neither unlink it
        # nor let the resource tracker count it.
        unregister(f"/{self._shm.name}", "shared_memory")
        if self._shm.size < pool_size:
            raise RuntimeError(
                f"L1 segment {shm_name} is {self._shm.size} bytes, "
                f"the server reports {pool_size}"
            )
        # Third Party
        import torch

        self._array = (ctypes.c_uint8 * pool_size).from_buffer(self._shm.buf)
        self.ptr = ctypes.addressof(self._array)
        self.tensor = torch.frombuffer(self._array, dtype=torch.uint8)
        self.size = pool_size
        self.pinned = False
        self.pin_seconds = 0.0
        # Pinning runs at ~7 GiB/s (measured: 32 GiB in 4.65 s), so a large
        # pool would hold REGISTER -- and with it the server's request loop --
        # for tens of seconds. It runs in the background instead; transfers
        # wait for it (see wait_pinned), REGISTER does not.
        self._ready = threading.Event()
        self._pin_thread = threading.Thread(
            target=self._pin, args=(shm_name, current_device_spec), daemon=True
        )
        self._pin_thread.start()

    def _pin(self, shm_name: str, device_spec: Any) -> None:
        t0 = time.perf_counter()
        try:
            self.pinned = bool(device_spec.pin_memory(self.ptr, self.size))
        except Exception:
            logger.exception("Pinning L1 segment %s failed", shm_name)
        self.pin_seconds = time.perf_counter() - t0
        if self.pinned:
            logger.info(
                "L1 segment %s pinned: %d bytes in %.2fs", shm_name, self.size, self.pin_seconds
            )
        else:
            logger.warning(
                "L1 segment %s could not be pinned; copies will be slow", shm_name
            )
        self._ready.set()

    def wait_pinned(self, timeout: float | None = None) -> bool:
        """Block until the pin attempt finished; True when the pool is pinned."""
        self._ready.wait(timeout)
        return self.pinned


class _Helper:
    def __init__(self, cmd: Connection, evt: Connection, init: HelperInit) -> None:
        self._cmd = cmd
        self._evt = evt
        self._evt_lock = threading.Lock()
        self._init = init
        self._pool: _L1PoolView | None = None
        self._cc: Any = None
        self._eb: Any = None
        self._layouts: dict[int, MemoryLayoutDesc] = {}
        self._dispatcher: Any = None

    # -- startup -------------------------------------------------------------

    def start(self) -> None:
        # First Party
        from lmcache.v1.multiprocess.native_completion import (
            DeviceHostFuncDispatcher,
        )
        from lmcache.v1.platform.ipc_policy import set_ipc_policy

        set_ipc_policy(
            isolated_ipc=self._init.isolated_ipc, use_vmm_api=self._init.use_vmm_api
        )
        self._pool = _L1PoolView(self._init.shm_name, self._init.pool_size)
        self._dispatcher = DeviceHostFuncDispatcher()
        self._dispatcher.register("notify", self._notify, payload_type=int)
        self._dispatcher.register(
            "release_imported_event", self._release_imported_event, payload_type=int
        )
        self._dispatcher.start()
        logger.info(
            "GPU helper for instance %d ready: pid=%d L1=%s (%d bytes, pinning in "
            "the background) CUDA_VISIBLE_DEVICES=%s",
            self._init.instance_id,
            os.getpid(),
            self._init.shm_name,
            self._pool.size,
            os.environ.get("CUDA_VISIBLE_DEVICES"),
        )
        self._send_event(("hello", os.getpid()))

    def _send_event(self, msg: tuple) -> None:
        with self._evt_lock:
            self._evt.send(msg)

    # -- stream callbacks (dispatcher thread) ----------------------------------

    def _notify(self, token: int) -> None:
        self._send_event(("done", token))

    def _release_imported_event(self, token: int) -> None:
        # First Party
        from lmcache.v1.platform.base.cache_context import (
            release_imported_event_token,
        )

        if release_imported_event_token(token):
            return
        if self._cc is not None:
            self._cc.release_imported_event(token)

    # -- commands --------------------------------------------------------------

    def serve(self) -> None:
        while True:
            try:
                msg = self._cmd.recv()
            except (EOFError, OSError):
                logger.warning("GPU helper command pipe closed; exiting")
                return
            kind = msg[0]
            try:
                if kind == "register":
                    self._cmd.send(("ok", self._register(msg[1])))
                elif kind == "transfer":
                    handle, ok = self._transfer(msg[1])
                    self._cmd.send(("ok", handle, ok))
                elif kind == "exit":
                    self._cmd.send(("bye",))
                    return
                else:
                    self._cmd.send(("err", f"unknown command {kind!r}"))
            except Exception as exc:
                logger.exception("GPU helper command %r failed", kind)
                self._cmd.send(("err", f"{type(exc).__name__}: {exc}"))

    def _register(self, req: RegisterRequest) -> RegisterReply:
        # First Party
        from lmcache.v1.multiprocess.modules.lmcache_driven_transfer import (
            get_layout_desc,
        )
        from lmcache.v1.platform.base.event_ipc import get_event_ipc_backend
        from lmcache.v1.platform.cache_context import create_cache_context

        if self._cc is not None:
            raise RuntimeError("helper already holds a registration")
        t0 = time.perf_counter()
        cc = create_cache_context(
            req.kv_caches,
            self._init.chunk_size,
            layout_hints=req.layout_hints or None,
            engine_group_infos=req.engine_group_infos,
            engine_type=req.engine_type,
            separate_object_groups=self._init.separate_object_groups,
            full_sw_kv=self._init.full_sw_kv,
        )
        eb = get_event_ipc_backend(cc.device)
        eb.check_event_support(cc.device)
        manager = cc.kv_layer_groups_manager
        chunk = self._init.chunk_size
        self._layouts = {
            gid: get_layout_desc(cc, chunk, object_group_id=gid)
            for gid in range(manager.num_object_groups)
        }
        self._cc, self._eb = cc, eb
        reply = RegisterReply(
            helper_pid=os.getpid(),
            device=str(cc.device),
            num_layers=cc.num_layers,
            num_object_groups=manager.num_object_groups,
            num_kernel_groups=manager.num_kernel_groups,
            group_layout_descs=dict(self._layouts),
            attn_desc=manager.get_attn_desc(),
            blocks_per_chunk=[
                cc.calculate_num_blocks(chunk, kg)
                for kg in range(manager.num_kernel_groups)
            ],
            object_groups=list(manager.object_groups),
            max_batch_size=cc.max_batch_size,
            kv_cache_layout=cc.report_status(),
            init_seconds=time.perf_counter() - t0,
            l1_pinned=(self._pool.pinned if self._pool._ready.is_set() else None),
        )
        logger.info(
            "GPU helper registered instance %d on %s: %d layers, %d object groups, "
            "%.2fs",
            self._init.instance_id,
            reply.device,
            reply.num_layers,
            reply.num_object_groups,
            reply.init_seconds,
        )
        return reply

    def _memory_obj(self, object_group_id: int, ref: ObjectRef) -> Any:
        """A view of one server L1 object through this process's mapping."""
        # First Party
        from lmcache.v1.memory_management import (
            MemoryFormat,
            MemoryObjMetadata,
            TensorMemoryObj,
        )

        assert self._pool is not None
        address, phy_size, fmt_value = ref
        if address < 0 or address + phy_size > self._pool.size:
            raise ValueError(
                f"L1 object [{address}, {address + phy_size}) is outside the "
                f"{self._pool.size}-byte pool"
            )
        layout = self._layouts[object_group_id]
        meta = MemoryObjMetadata(
            shape=layout.shapes[0],
            dtype=layout.dtypes[0],
            address=address,
            phy_size=phy_size,
            ref_count=1,
            pin_count=0,
            fmt=MemoryFormat(fmt_value),
            shapes=list(layout.shapes),
            dtypes=list(layout.dtypes),
        )
        raw = self._pool.tensor[address : address + phy_size]
        return TensorMemoryObj(raw_data=raw, metadata=meta, parent_allocator=None)

    def _end_held_section(self, import_token: int) -> None:
        # First Party
        from lmcache.v1.multiprocess.native_completion import (
            submit_callback_to_stream,
        )

        try:
            submit_callback_to_stream(
                self._cc.cupy_stream, "release_imported_event", import_token
            )
        except Exception:
            logger.exception("Queueing the hold release failed")

    def _transfer(self, req: TransferRequest) -> tuple[bytes, bool]:
        # First Party
        from lmcache import torch_dev
        from lmcache.v1.multiprocess.native_completion import (
            submit_callback_to_stream,
        )
        from lmcache.v1.multiprocess.object_group_transfer import (
            downsample_and_stage_block_ids,
            transfer_kv_per_object_group,
        )
        import lmcache.lmcache_native as lmcache_native

        cc, eb = self._cc, self._eb
        if cc is None or eb is None:
            raise RuntimeError("helper has no registration")
        assert self._pool is not None
        if not self._pool._ready.is_set():
            t0 = time.perf_counter()
            self._pool.wait_pinned()
            logger.info(
                "First transfer waited %.2fs for the L1 pin", time.perf_counter() - t0
            )
        is_store = req.direction == "store"
        ok = True
        with (
            torch_dev.device(cc.device),
            torch_dev.stream(cc.stream),
            contextlib.ExitStack() as held_section,
        ):
            event = eb.create_event(cc.device)
            if getattr(eb, "supports_import_users", False):
                producer_event = eb.import_event(
                    req.event_ipc_handle, cc.device, user=cc
                )
            else:
                producer_event = eb.import_event(req.event_ipc_handle, cc.device)
            import_token = cc.hold_imported_event(producer_event)
            held_section.callback(self._end_held_section, import_token)
            block_ids_gpu = downsample_and_stage_block_ids(cc, req.gpu_block_ids)
            eb.wait_event(producer_event, cc.stream)
            try:
                for object_group_id, refs in req.groups:
                    memory_objs = [
                        None if ref is None else self._memory_obj(object_group_id, ref)
                        for ref in refs
                    ]
                    transfer_kv_per_object_group(
                        cc,
                        block_ids_gpu,
                        memory_objs,
                        object_group_id=object_group_id,
                        batch_size=1 if is_store else cc.max_batch_size,
                        skip_first_n_tokens=0 if is_store else req.skip_first_n_tokens,
                        direction=(
                            lmcache_native.TransferDirection.D2H
                            if is_store
                            else lmcache_native.TransferDirection.H2D
                        ),
                        transfer_key=req.transfer_key,
                        block_ids_host=req.gpu_block_ids,
                    )
            except Exception:
                logger.exception("GPU helper %s failed", req.direction)
                ok = False
            finally:
                eb.record_event(event, cc.stream)
                submit_callback_to_stream(cc.cupy_stream, "notify", req.token)
        return eb.export_event(event, cc.device), ok


def helper_main(cmd: Connection, evt: Connection, init: HelperInit) -> None:
    """Entry point of the helper process (``multiprocessing`` spawn target)."""
    helper = _Helper(cmd, evt, init)
    try:
        helper.start()
        helper.serve()
    except Exception:
        logger.exception("GPU helper for instance %d died", init.instance_id)
    finally:
        # Never run Python teardown: pending device work may be waiting on a
        # dead worker's event, and only process exit reclaims the context.
        os._exit(0)

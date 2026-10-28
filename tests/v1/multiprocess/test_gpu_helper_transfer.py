# SPDX-License-Identifier: Apache-2.0
"""Server side of the GPU-helper transfer path, with an in-process fake helper.

The helper process is replaced by a thread speaking the pipe protocol, so these
tests cover the control plane: object refs sent as pool offsets, completion
tokens, fail-closed settlement when a helper dies or stops answering, and the
registration lifecycle.
"""

# Standard
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any
import multiprocessing
import threading
import time

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import EngineType
from lmcache.v1.distributed.api import AttnWindowDesc, MemoryLayoutDesc, ObjectKey
from lmcache.v1.kv_layer_groups import ObjectGroupInfo
from lmcache.v1.memory_management import MemoryFormat
from lmcache.v1.mp_observability.event import EventType
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.gpu_helper import (
    RegisterReply,
    RegisterRequest,
    TransferRequest,
)
from lmcache.v1.multiprocess.modules.gpu_helper_transfer import (
    GPUHelperTransferModule,
    l1_shm_pool_info,
)

CHUNK = 4
LAYOUT = MemoryLayoutDesc(shapes=[torch.Size([2, CHUNK, 8])], dtypes=[torch.bfloat16])


# --------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------


class FakeMemoryObj:
    def __init__(self, address: int, size: int) -> None:
        self.meta = SimpleNamespace(
            address=address, phy_size=size, fmt=MemoryFormat.KV_2LTD
        )
        self._size = size

    def get_size(self) -> int:
        return self._size


class FakeStorageManager:
    def __init__(self) -> None:
        self.reserved: dict[ObjectKey, FakeMemoryObj] = {}
        self.resident: dict[ObjectKey, FakeMemoryObj] = {}
        self.calls: list[tuple[str, Any]] = []
        self._next = 4096
        mem = SimpleNamespace(
            use_lazy=False,
            devdax_path=None,
            use_hugepages=False,
            shm_name="pool-x",
        )
        l1 = SimpleNamespace(
            _config=SimpleNamespace(memory_config=mem, gds_l1_config=None),
            get_l1_memory_desc=lambda: SimpleNamespace(ptr=0x1000, size=1 << 20),
        )
        self._l1_managers_by_id = {7: l1}

    def reserve_write(self, keys, layout_desc):
        self.calls.append(("reserve_write", list(keys), layout_desc))
        out = {}
        for k in keys:
            obj = FakeMemoryObj(self._next, 512)
            self._next += 512
            self.reserved[k] = obj
            out[k] = obj
        return out

    def prepare_write_completion(self, objects):
        return [(7, list(objects))]

    def finish_write_by_owner(self, completion):
        self.calls.append(("finish_write_by_owner", completion))

    def prepare_read_completion(self, keys, l1_owners=None):
        return [(7, list(keys))] if keys else []

    def unsafe_read(self, keys, l1_owners=None):
        found = [k for k in keys if k in self.resident]
        return found, [self.resident[k] for k in found]

    def finish_read_by_owner(self, completion):
        self.calls.append(("finish_read_by_owner", completion))

    def finish_read_prefetched(self, keys, read_locks=1, l1_owners=None):
        self.calls.append(("finish_read_prefetched", list(keys)))


class FakeEventBus:
    def __init__(self) -> None:
        self.events: list[Any] = []

    def publish(self, event) -> None:
        self.events.append(event)

    def has_subscribers(self, event_type) -> bool:
        return False


class FakeRegistry:
    def __init__(self) -> None:
        self.registered: list[tuple] = []
        self.unregistered: list[tuple] = []

    def register(self, model_name, world_size, layout, attn, group_layout_descs=None):
        self.registered.append((model_name, world_size))

    def unregister(self, model_name, world_size):
        self.unregistered.append((model_name, world_size))


def _obj_key(i: int) -> ObjectKey:
    return ObjectKey(
        model_name="m", chunk_hash=bytes([i]) * 4, kv_rank=0, object_group_id=0
    )


class FakeCtx:
    def __init__(self, num_groups: int = 1) -> None:
        self.chunk_size = CHUNK
        self.null_block_id = 0
        self.separate_object_groups = num_groups > 1
        self.full_sw_kv = False
        self.storage_manager = FakeStorageManager()
        self.event_bus = FakeEventBus()
        self.layout_desc_registry = FakeRegistry()
        self.session_manager = SimpleNamespace(get=lambda rid: None)
        self.num_groups = num_groups

    def resolve_obj_keys(self, key, object_group_ids):
        num_chunks = (key.end - key.start) // CHUNK
        return [
            [
                ObjectKey(
                    model_name="m",
                    chunk_hash=bytes([i + 1]) * 4,
                    kv_rank=0,
                    object_group_id=gid,
                )
                for i in range(num_chunks)
            ]
            for gid in object_group_ids
        ]

    def get_read_owners(self, request_id):
        return None


@dataclass
class FakeProcess:
    pid: int = 4242
    alive: bool = True
    killed: bool = False
    joined: list = field(default_factory=list)

    def is_alive(self) -> bool:
        return self.alive

    def kill(self) -> None:
        self.killed = True
        self.alive = False

    def join(self, timeout=None) -> None:
        self.joined.append(timeout)


class FakeHelper:
    """A thread that answers the command pipe like the real helper."""

    def __init__(
        self,
        *,
        num_groups: int = 1,
        windows: list[int] | None = None,
        transfer_ok: bool = True,
        auto_done: bool = True,
        hang: bool = False,
        done_before_reply: bool = False,
    ) -> None:
        ctx = multiprocessing.get_context("spawn")
        self.cmd_parent, self.cmd_child = ctx.Pipe(duplex=True)
        self.evt_parent, self.evt_child = ctx.Pipe(duplex=False)
        self.process = FakeProcess()
        self.num_groups = num_groups
        self.windows = windows or [-1] * num_groups
        self.transfer_ok = transfer_ok
        self.auto_done = auto_done
        self.hang = hang
        self.done_before_reply = done_before_reply
        self.requests: list[Any] = []
        self.exited = threading.Event()
        self._lock = threading.Lock()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self.thread.start()
        self.evt_child.send(("hello", self.process.pid))

    def send_done(self, token: int) -> None:
        with self._lock:
            self.evt_child.send(("done", token))

    def die(self) -> None:
        self.process.alive = False
        self.cmd_child.close()
        self.evt_child.close()

    def _run(self) -> None:
        while True:
            try:
                msg = self.cmd_child.recv()
            except (EOFError, OSError):
                return
            self.requests.append(msg)
            if msg[0] == "register":
                reply = RegisterReply(
                    helper_pid=self.process.pid,
                    device="cuda:0",
                    num_layers=2,
                    num_object_groups=self.num_groups,
                    num_kernel_groups=self.num_groups,
                    group_layout_descs={g: LAYOUT for g in range(self.num_groups)},
                    attn_desc=AttnWindowDesc(num_chunks_in_sw=list(self.windows)),
                    blocks_per_chunk=[1] * self.num_groups,
                    object_groups=[
                        ObjectGroupInfo(kernel_group_indices=[g])
                        for g in range(self.num_groups)
                    ],
                    max_batch_size=4,
                )
                self.cmd_child.send(("ok", reply))
            elif msg[0] == "transfer":
                if self.hang:
                    return  # never answers; the pipe stays open
                req: TransferRequest = msg[1]
                if self.done_before_reply:
                    self.send_done(req.token)
                    time.sleep(0.05)
                self.cmd_child.send(("ok", b"evt-" + str(req.token).encode(), self.transfer_ok))
                if self.auto_done and not self.done_before_reply:
                    self.send_done(req.token)
            elif msg[0] == "exit":
                self.cmd_child.send(("bye",))
                self.process.alive = False
                self.exited.set()
                self.cmd_child.close()
                self.evt_child.close()
                return


class FakeWrapper:
    device_uuid = "11111111-2222-3333-4444-555555555555"


def _key(num_chunks: int, worker_id: int = 1, rid: str = "r1") -> IPCCacheServerKey:
    return IPCCacheServerKey(
        model_name="m",
        world_size=1,
        worker_id=worker_id,
        token_ids=tuple(range(num_chunks * CHUNK)),
        start=0,
        end=num_chunks * CHUNK,
        request_id=rid,
    )


def _make(monkeypatch, helper: FakeHelper, ctx: FakeCtx | None = None):
    ctx = ctx or FakeCtx(helper.num_groups)
    module = GPUHelperTransferModule(
        ctx, call_timeout_seconds=0.5, register_timeout_seconds=2.0, exit_timeout_seconds=0.5
    )
    spawned: list[tuple] = []

    def spawn(instance_id, device_uuid, init):
        spawned.append((instance_id, device_uuid, init))
        helper.start()
        return helper.process, helper.cmd_parent, helper.evt_parent

    monkeypatch.setattr(module, "_spawn_process", spawn)
    monkeypatch.setattr(
        "lmcache.v1.multiprocess.modules.gpu_helper_transfer._isolated_ipc_mismatch",
        lambda kv, et: None,
    )
    module.spawned = spawned  # type: ignore[attr-defined]
    return module, ctx


def _register(module, instance_id: int = 11) -> None:
    module.register_kv_cache(
        instance_id, [FakeWrapper()], "m", 1, EngineType.VLLM, None, []
    )


def _wait(pred, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while not pred():
        if time.monotonic() > deadline:
            raise AssertionError("condition not met in time")
        time.sleep(0.01)


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------


def test_pool_info_requires_eager_shm_l1():
    sm = FakeStorageManager()
    assert l1_shm_pool_info(sm) == ("lmcache_l1_pool_pool-x", 0x1000, 1 << 20)
    sm._l1_managers_by_id[7]._config.memory_config.use_lazy = True
    with pytest.raises(ValueError, match="lazy"):
        l1_shm_pool_info(sm)


def test_register_spawns_helper_on_the_worker_gpu(monkeypatch):
    helper = FakeHelper()
    module, ctx = _make(monkeypatch, helper)
    _register(module)
    assert module.spawned[0][1] == FakeWrapper.device_uuid
    init = module.spawned[0][2]
    assert (init.shm_name, init.pool_size, init.chunk_size) == (
        "lmcache_l1_pool_pool-x",
        1 << 20,
        CHUNK,
    )
    assert isinstance(helper.requests[0][1], RegisterRequest)
    assert ctx.layout_desc_registry.registered == [("m", 1)]
    assert module.tracked_instance_count() == 1
    assert module.touch_instance(11) is True
    assert module.touch_instance(12) is False
    status = module.report_status()
    assert status["registered_gpu_ids"] == [11]
    assert status["ipc_release"]["helpers"]["11"]["pid"] == 4242
    # A second REGISTER of the same instance only refreshes liveness.
    _register(module)
    assert len(module.spawned) == 1


def test_store_sends_pool_offsets_and_commits_on_done(monkeypatch):
    helper = FakeHelper()
    module, ctx = _make(monkeypatch, helper)
    _register(module)
    key = _key(3)
    # Chunk 1 is all-null: it must be skipped (None) and not reserved.
    block_ids = [[5, 0, 7]]
    handle, ok, mask = module.store_with_chunk_mask(key, 11, block_ids, b"producer")
    assert ok and handle.startswith(b"evt-")
    assert mask == [True, False, True]
    req: TransferRequest = helper.requests[1][1]
    assert req.direction == "store" and req.event_ipc_handle == b"producer"
    (gid, refs), = req.groups
    assert gid == 0
    assert refs[1] is None
    assert refs[0] == (4096, 512, MemoryFormat.KV_2LTD.value)
    assert refs[2] == (4096 + 512, 512, MemoryFormat.KV_2LTD.value)
    sm = ctx.storage_manager
    _wait(lambda: any(c[0] == "finish_write_by_owner" for c in sm.calls))
    committed = next(c for c in sm.calls if c[0] == "finish_write_by_owner")[1]
    assert [len(keys) for _, keys in committed] == [2]
    assert module.release_status()["dropped_commits"] == 0
    kinds = [e.event_type for e in ctx.event_bus.events]
    assert kinds[-1] == EventType.MP_STORE_END
    assert ctx.event_bus.events[-1].metadata["stored_count"] == 2


def test_store_failure_in_helper_is_not_committed(monkeypatch):
    helper = FakeHelper(transfer_ok=False)
    module, ctx = _make(monkeypatch, helper)
    _register(module)
    _, ok, mask = module.store_with_chunk_mask(_key(2), 11, [[1, 2]], b"p")
    assert ok is False and mask == [False, False]
    _wait(lambda: module.release_status()["dropped_commits"] == 1)
    assert not any(c[0] == "finish_write_by_owner" for c in ctx.storage_manager.calls)


def test_done_before_reply_commits_once(monkeypatch):
    helper = FakeHelper(done_before_reply=True)
    module, ctx = _make(monkeypatch, helper)
    _register(module)
    _, ok, _ = module.store_with_chunk_mask(_key(2), 11, [[1, 2]], b"p")
    assert ok
    sm = ctx.storage_manager
    _wait(lambda: any(c[0] == "finish_write_by_owner" for c in sm.calls))
    time.sleep(0.1)
    assert sum(c[0] == "finish_write_by_owner" for c in sm.calls) == 1


def test_store_block_id_underflow_is_rejected_without_helper(monkeypatch):
    helper = FakeHelper()
    module, _ = _make(monkeypatch, helper)
    _register(module)
    assert module.store_with_chunk_mask(_key(3), 11, [[1, 2]], b"p") == (b"", False, [])
    assert len(helper.requests) == 1  # only the REGISTER


def test_retrieve_pads_sliding_window_and_releases_locks_on_done(monkeypatch):
    helper = FakeHelper(num_groups=2, windows=[-1, 1])
    module, ctx = _make(monkeypatch, helper)
    _register(module)
    key = _key(3)
    keys = ctx.resolve_obj_keys(key, [0, 1])
    sm = ctx.storage_manager
    for gid in (0, 1):
        for i, k in enumerate(keys[gid]):
            sm.resident[k] = FakeMemoryObj(8192 + gid * 10000 + i * 512, 512)
    handle, ok = module.retrieve(key, 11, [[1, 2, 3], [4, 5, 6]], b"p", skip_first_n_tokens=2)
    assert ok and handle.startswith(b"evt-")
    req: TransferRequest = helper.requests[1][1]
    assert req.direction == "retrieve" and req.skip_first_n_tokens == 2
    groups = dict(req.groups)
    assert [r is None for r in groups[0]] == [False, False, False]
    # Window of 1 chunk on group 1: the first two chunks are None-padded.
    assert [r is None for r in groups[1]] == [True, True, False]
    assert groups[1][2][0] == 8192 + 10000 + 2 * 512
    _wait(lambda: any(c[0] == "finish_read_by_owner" for c in sm.calls))
    released = next(c for c in sm.calls if c[0] == "finish_read_by_owner")[1]
    assert [len(k) for _, k in released] == [4]  # 3 (full) + 1 (window)
    assert ctx.event_bus.events[-1].event_type == EventType.MP_RETRIEVE_END
    assert ctx.event_bus.events[-1].metadata["retrieved_count"] == 4


def test_retrieve_miss_releases_locks_without_helper(monkeypatch):
    helper = FakeHelper()
    module, ctx = _make(monkeypatch, helper)
    _register(module)
    assert module.retrieve(_key(2), 11, [[1, 2]], b"p") == (b"", False)
    assert len(helper.requests) == 1
    assert any(c[0] == "finish_read_by_owner" for c in ctx.storage_manager.calls)


def test_helper_death_releases_reads_and_drops_writes(monkeypatch):
    helper = FakeHelper(auto_done=False)
    module, ctx = _make(monkeypatch, helper)
    _register(module)
    sm = ctx.storage_manager
    key = _key(2, rid="w")
    for k in ctx.resolve_obj_keys(key, [0])[0]:
        sm.resident[k] = FakeMemoryObj(8192, 512)
    assert module.store_with_chunk_mask(_key(2, rid="s"), 11, [[1, 2]], b"p")[1]
    assert module.retrieve(key, 11, [[1, 2]], b"p")[1]
    assert module.release_status()["helpers"]["11"]["pending_transfers"] == 2
    helper.die()
    _wait(lambda: module.tracked_instance_count() == 0)
    _wait(lambda: any(c[0] == "finish_read_by_owner" for c in sm.calls))
    assert not any(c[0] == "finish_write_by_owner" for c in sm.calls)
    status = module.release_status()
    assert status["helpers_died"] == 1 and status["dropped_commits"] == 1
    assert ctx.layout_desc_registry.unregistered == [("m", 1)]
    assert module.touch_instance(11) is False
    assert module.store_with_chunk_mask(_key(1), 11, [[1]], b"p") == (b"", False, [])


def test_helper_timeout_kills_it_and_rejects_the_store(monkeypatch):
    helper = FakeHelper(hang=True)
    module, ctx = _make(monkeypatch, helper)
    _register(module)
    assert module.store_with_chunk_mask(_key(1), 11, [[1]], b"p") == (b"", False, [])
    assert helper.process.killed
    assert module.tracked_instance_count() == 0
    assert module.release_status()["failed_transfers"] == 1
    assert module.release_status()["helpers_killed"] == 1


def test_stuck_stream_kills_the_helper_on_the_next_transfer(monkeypatch):
    helper = FakeHelper(auto_done=False)
    module, ctx = _make(monkeypatch, helper)
    module._stuck_transfer_seconds = 0.05
    _register(module)
    assert module.store_with_chunk_mask(_key(1, rid="a"), 11, [[1]], b"p")[1]
    time.sleep(0.1)
    assert module.store_with_chunk_mask(_key(1, rid="b"), 11, [[2]], b"p") == (b"", False, [])
    assert helper.process.killed
    status = module.release_status()
    assert status["stuck_streams"] == 1 and status["dropped_commits"] == 2
    assert module.tracked_instance_count() == 0


def test_vmm_ipc_mode_is_refused(monkeypatch):
    monkeypatch.setattr(
        "lmcache.v1.multiprocess.modules.gpu_helper_transfer.is_use_vmm_api",
        lambda: True,
    )
    with pytest.raises(ValueError, match="VMM"):
        GPUHelperTransferModule(FakeCtx())


def test_unregister_exits_the_helper(monkeypatch):
    helper = FakeHelper()
    module, ctx = _make(monkeypatch, helper)
    _register(module)
    module.unregister_kv_cache(11)
    assert helper.exited.wait(1.0)
    assert helper.requests[-1] == ("exit",)
    assert module.tracked_instance_count() == 0
    assert ctx.layout_desc_registry.unregistered == [("m", 1)]
    status = module.release_status()
    assert status["helpers_exited"] == 1 and status["released"] == 1
    module.unregister_kv_cache(11)  # unknown: no error


def test_reap_stops_the_helper(monkeypatch):
    helper = FakeHelper()
    module, _ = _make(monkeypatch, helper)
    _register(module)
    assert module.reap_stale_instances(1000.0, 1000.0) == []
    module.touch_instance(11)
    time.sleep(0.05)
    assert module.reap_stale_instances(0.01, 1000.0) == [11]
    assert helper.exited.wait(1.0)
    assert module.tracked_instance_count() == 0


def test_close_exits_every_helper(monkeypatch):
    helper = FakeHelper()
    module, _ = _make(monkeypatch, helper)
    _register(module)
    module.close()
    assert helper.exited.wait(1.0)
    assert module.stuck_release_seconds() is None
    assert module.context_entries_snapshot() == {}

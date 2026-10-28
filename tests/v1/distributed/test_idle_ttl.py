# SPDX-License-Identifier: Apache-2.0
"""
Idle-TTL tests (FCI patch 0027).

U5  prefix invariant: expiry never removes a parent chunk while a child of it
    is still tracked (chains, trees, randomized property test)
U6  cross-tier touch: an L1 hit refreshes the L2 idle clock (and, with
    touch_on_l1_hit, the L2 LRU order) without any adapter I/O
U7  TTL with a fake clock: batches, locked/pinned chunks kept, whole chunks
    deleted, TTL off changes nothing
U8  config: new flags distinct from the lock TTLs, validation, defaults off
"""

# Future
from __future__ import annotations

# Standard
from pathlib import Path
import random
import select
import shutil
import tempfile

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.config import EvictionConfig, parse_args
from lmcache.v1.distributed.eviction_policy.idle_ttl import (
    IdleTTLTracker,
    L1IdleTTLListener,
    chunk_id_of,
)
from lmcache.v1.distributed.internal_api import L1ManagerListener, L2AdapterListener
from lmcache.v1.distributed.l2_adapters.config import L2AdapterConfigBase
from lmcache.v1.distributed.storage_controllers.eviction_controller import (
    L1EvictionController,
    L2AdapterEvictionState,
    L2EvictionController,
)
from lmcache.v1.distributed.storage_manager import (
    StorageManager,
    _validate_idle_ttl_tiers,
)


def key(chunk: int, group: int = 0, rank: int = 0, salt: str = "") -> ObjectKey:
    return ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes(chunk),
        model_name="m",
        kv_rank=rank,
        object_group_id=group,
        cache_salt=salt,
    )


def chunk_keys(chunk: int, groups: int = 2) -> list[ObjectKey]:
    return [key(chunk, g) for g in range(groups)]


def request(*chunks: int, groups: int = 2) -> list[ObjectKey]:
    """Keys of one request in request order (prefix first)."""
    return [k for c in chunks for k in chunk_keys(c, groups)]


class FakeClock:
    def __init__(self, t: float = 0.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


def ids(*chunks: int):
    return [chunk_id_of(key(c)) for c in chunks]


# =============================================================================
# U5: prefix invariant
# =============================================================================


def test_u5_chain_expires_tail_first():
    clock = FakeClock()
    tr = IdleTTLTracker(clock=clock)
    tr.on_keys_created(request(0, 1, 2, 3, 4))
    clock.t = 100
    got = [cid for cid, _ in tr.expired_chunks(60, limit=10)]
    assert got == ids(4, 3, 2, 1, 0)
    assert [cid for cid, _ in tr.expired_chunks(60, limit=2)] == ids(4, 3)


def test_u5_touching_a_prefix_keeps_it_and_its_parents():
    clock = FakeClock()
    tr = IdleTTLTracker(clock=clock)
    tr.on_keys_created(request(0, 1, 2, 3, 4))
    clock.t = 50
    tr.on_keys_touched(request(0, 1, 2))  # a later request hit chunks 0..2
    clock.t = 100
    assert [cid for cid, _ in tr.expired_chunks(60, 10)] == ids(4, 3)
    clock.t = 115
    assert [cid for cid, _ in tr.expired_chunks(60, 10)] == ids(4, 3, 2, 1, 0)


def test_u5_tree_parent_outlives_every_branch():
    clock = FakeClock()
    tr = IdleTTLTracker(clock=clock)
    tr.on_keys_created(request(0, 10, 11))  # branch A: p, a1, a2
    clock.t = 30
    tr.on_keys_created(request(0, 20))  # branch B: p, b1 (touches p)
    clock.t = 70
    assert [cid for cid, _ in tr.expired_chunks(50, 10)] == ids(11, 10)


def test_u5_randomized_never_expires_parent_before_child():
    rng = random.Random(20261028)
    for _round in range(30):
        clock = FakeClock()
        tr = IdleTTLTracker(clock=clock)
        parent: dict[int, int | None] = {0: None}
        nxt = 1
        paths: list[list[int]] = [[0]]
        for _step in range(60):
            clock.t += rng.uniform(0, 5)
            base = rng.choice(paths)
            if rng.random() < 0.6:  # extend a path (new child chunk)
                parent[nxt] = base[-1]
                path = base + [nxt]
                nxt += 1
                paths.append(path)
                tr.on_keys_created(request(*path))
            else:  # re-hit a prefix of an existing path
                path = base[: rng.randint(1, len(base))]
                tr.on_keys_touched(request(*path))
        clock.t += 1000
        tracked = {cid for cid in tr.chunk_order()}
        by_cid = {chunk_id_of(key(c)): c for c in parent}
        while True:
            exp = tr.expired_chunks(1, limit=1)
            if not exp:
                break
            cid, members = exp[0]
            c = by_cid[cid]
            children = [x for x, p in parent.items() if p == c]
            alive = [x for x in children if chunk_id_of(key(x)) in tracked]
            assert not alive, f"parent {c} expired before child {alive}"
            tr.on_keys_removed(members)
            tracked.discard(cid)
        assert len(tr) == 0


def test_u5_chunk_removed_only_when_all_members_gone():
    tr = IdleTTLTracker(clock=FakeClock())
    tr.on_keys_created(request(0, groups=2))
    tr.on_keys_removed([key(0, 0)])
    assert len(tr) == 1
    tr.on_keys_removed([key(0, 1)])
    assert len(tr) == 0


def test_u5_touch_does_not_add_members_or_unknown_chunks():
    clock = FakeClock()
    tr = IdleTTLTracker(clock=clock)
    tr.on_keys_created([key(0, 0)])
    tr.on_keys_touched([key(0, 1), key(9, 0)])  # sibling and unknown chunk
    assert len(tr) == 1
    clock.t = 100
    ((_, members),) = tr.expired_chunks(1, 10)
    assert members == [key(0, 0)]


# =============================================================================
# U6: cross-tier touch
# =============================================================================


class FakeL2Adapter:
    """Records every call; delete notifies listeners like a real adapter."""

    def __init__(self) -> None:
        self.listeners: list[L2AdapterListener] = []
        self.keys: set[ObjectKey] = set()
        self.pinned: set[ObjectKey] = set()
        self.calls: list[str] = []

    def register_listener(self, listener: L2AdapterListener) -> None:
        self.listeners.append(listener)

    def store(self, keys: list[ObjectKey]) -> None:
        self.keys.update(keys)
        for li in self.listeners:
            li.on_l2_keys_stored(keys, [1] * len(keys))

    def delete(self, keys: list[ObjectKey]) -> None:
        self.calls.append("delete")
        gone = [k for k in keys if k in self.keys and k not in self.pinned]
        self.keys.difference_update(gone)
        for li in self.listeners:
            li.on_l2_keys_deleted(gone)

    def __getattr__(self, name):
        if name.startswith("submit_") or name in ("get_usage",):

            def _record(*a, **kw):
                self.calls.append(name)

            return _record
        raise AttributeError(name)


def l2_state(clock, **cfg) -> tuple[FakeL2Adapter, L2AdapterEvictionState]:
    adapter = FakeL2Adapter()
    config = EvictionConfig("LRU", 0.9, 0.25, **cfg)
    return adapter, L2AdapterEvictionState(0, adapter, config, clock=clock)  # type: ignore[arg-type]


def test_u6_l1_hit_refreshes_l2_idle_clock_without_io():
    clock = FakeClock()
    adapter, st = l2_state(clock, idle_ttl_seconds=300)
    ctl = L2EvictionController([st])
    adapter.store(request(0, 1))
    clock.t = 200
    ctl.on_l1_keys_touched(request(0, 1))
    assert st.idle_tracker is not None
    assert st.idle_tracker.last_access(chunk_id_of(key(0))) == 200
    assert adapter.calls == []  # no adapter call at all on the touch path
    clock.t = 450  # 450 - 200 = 250 < 300: still fresh thanks to the touch
    assert ctl.run_idle_ttl_pass(st) == 0


def test_u6_touch_on_l1_hit_moves_l2_lru_order_only_when_enabled():
    for enabled in (False, True):
        clock = FakeClock()
        adapter, st = l2_state(clock, touch_on_l1_hit=enabled)
        ctl = L2EvictionController([st])
        adapter.store(chunk_keys(0))
        adapter.store(chunk_keys(1))
        ctl.on_l1_keys_touched(chunk_keys(0))
        first = st.eviction_policy.get_eviction_candidates(1)[0]  # type: ignore[attr-defined]
        assert chunk_id_of(first) == (
            chunk_id_of(key(1)) if enabled else chunk_id_of(key(0))
        )
        assert adapter.calls == []


def test_u6_storage_manager_forwards_l1_touch_to_l2_controller():
    seen: dict[str, list] = {"l1": [], "l2": []}

    class L1:
        def touch_keys(self, keys):
            seen["l1"].append(keys)

    class L2Ctl:
        def on_l1_keys_touched(self, keys):
            seen["l2"].append(keys)

    class Self:
        _l1_managers_by_id = {0: L1()}
        _l2_eviction_controller = L2Ctl()

    keys = request(0, 1)
    StorageManager.touch_l1_keys(Self(), keys)  # type: ignore[arg-type]
    assert seen == {"l1": [keys], "l2": [keys]}


def test_u6_default_config_adds_no_tracker_and_no_listener():
    adapter, st = l2_state(FakeClock())
    assert st.idle_tracker is None
    assert len(adapter.listeners) == 1  # only the eviction-policy bridge


# =============================================================================
# U7: TTL with a fake clock
# =============================================================================


class FakeL1Manager:
    def __init__(self) -> None:
        self.listeners: list[L1ManagerListener] = []
        self.keys: set[ObjectKey] = set()
        self.locked: set[ObjectKey] = set()
        self.deleted: list[list[ObjectKey]] = []

    def register_listener(self, listener: L1ManagerListener) -> None:
        self.listeners.append(listener)

    def write(self, keys: list[ObjectKey]) -> None:
        self.keys.update(keys)
        for li in self.listeners:
            li.on_l1_keys_write_finished(keys)

    def is_key_evictable(self, k: ObjectKey) -> bool:
        return k not in self.locked

    def delete(self, keys: list[ObjectKey]) -> None:
        self.deleted.append(list(keys))
        gone = [k for k in keys if k in self.keys and k not in self.locked]
        self.keys.difference_update(gone)
        for li in self.listeners:
            li.on_l1_keys_deleted_by_manager(gone)

    def get_memory_usage(self):
        return 0, 1


def l1_ctl(clock, **cfg) -> tuple[FakeL1Manager, L1EvictionController]:
    mgr = FakeL1Manager()
    config = EvictionConfig("LRU", 0.9, 0.1, **cfg)
    return mgr, L1EvictionController(mgr, config, clock=clock)  # type: ignore[arg-type]


def test_u7_l1_deletes_expired_whole_chunks_in_batches():
    clock = FakeClock()
    mgr, ctl = l1_ctl(clock, idle_ttl_seconds=60, idle_ttl_batch=2)
    mgr.write(request(0, 1, 2))
    clock.t = 30
    mgr.write(request(5))
    clock.t = 100  # 0..2 idle 100 s, chunk 5 idle 70 s: all expired
    assert ctl.run_idle_ttl_pass() == 2
    # Tail first, whole chunks: chunks 2 and 1, never the prefix head 0.
    assert set(mgr.deleted[-1]) == set(chunk_keys(2) + chunk_keys(1))
    assert ctl.run_idle_ttl_pass() == 2  # chunk 0, then chunk 5
    assert ctl.run_idle_ttl_pass() == 0
    assert mgr.keys == set()
    st = ctl.report_status()["idle_ttl"]
    assert st["evicted_chunks"] == 4 and st["tracked_chunks"] == 0


def test_u7_l1_locked_object_keeps_its_whole_chunk():
    clock = FakeClock()
    mgr, ctl = l1_ctl(clock, idle_ttl_seconds=60, idle_ttl_batch=8)
    mgr.write(request(0, 1))
    mgr.locked.add(key(1, 1))  # one object of chunk 1 is being stored to L2
    clock.t = 100
    assert ctl.run_idle_ttl_pass() == 1  # chunk 0 only
    assert set(chunk_keys(1)) <= mgr.keys
    assert ctl.report_status()["idle_ttl"]["skipped_chunks"] == 1
    mgr.locked.clear()
    assert ctl.run_idle_ttl_pass() == 1
    assert mgr.keys == set()


def test_u7_l1_fresh_chunks_are_not_touched():
    clock = FakeClock()
    mgr, ctl = l1_ctl(clock, idle_ttl_seconds=60)
    mgr.write(request(0, 1))
    clock.t = 59
    assert ctl.run_idle_ttl_pass() == 0
    assert mgr.deleted == []


def test_u7_ttl_off_changes_nothing():
    clock = FakeClock()
    mgr, ctl = l1_ctl(clock)  # idle_ttl_seconds defaults to 0
    assert len(mgr.listeners) == 1  # only the eviction-policy bridge
    mgr.write(request(0))
    clock.t = 10**9
    assert ctl.run_idle_ttl_pass() == 0
    assert mgr.deleted == []
    assert "idle_ttl" not in ctl.report_status()


def test_u7_l2_pinned_chunk_is_kept_and_counted():
    clock = FakeClock()
    adapter, st = l2_state(clock, idle_ttl_seconds=300, idle_ttl_batch=8)
    ctl = L2EvictionController([st])
    adapter.store(request(0, 1))
    adapter.pinned.add(key(1, 0))
    clock.t = 400
    assert ctl.run_idle_ttl_pass(st) == 1
    assert st.idle_skipped_chunks == 1
    adapter.pinned.clear()
    assert ctl.run_idle_ttl_pass(st) == 1
    assert adapter.keys == set()


# ---- U7 against the real nixl_store_dynamic adapter (whole chunk, pin, files)

nixl = pytest.importorskip("nixl")

# First Party
from lmcache.v1.distributed.internal_api import L1MemoryDesc  # noqa: E402
from lmcache.v1.distributed.l2_adapters.config import PersistConfig  # noqa: E402
from lmcache.v1.distributed.l2_adapters.nixl_store_dynamic_l2_adapter import (  # noqa: E402, E501
    DynamicNixlStoreL2Adapter,
    DynamicNixlStoreL2AdapterConfig,
)
from lmcache.v1.memory_management import (  # noqa: E402
    MemoryFormat,
    MemoryObjMetadata,
    TensorMemoryObj,
)
from lmcache.v1.platform import consume_fd  # noqa: E402

PAGE = 4096


def _obj(buf: torch.Tensor, page: int) -> TensorMemoryObj:
    raw = buf[page * PAGE : (page + 1) * PAGE].view(torch.float32)
    raw.fill_(float(page))
    meta = MemoryObjMetadata(
        shape=torch.Size([PAGE // 4]),
        dtype=torch.float32,
        address=page * PAGE,
        phy_size=PAGE,
        fmt=MemoryFormat.KV_2LTD,
        ref_count=1,
    )
    return TensorMemoryObj(raw, meta, parent_allocator=None)


def _wait(fd: int) -> None:
    p = select.poll()
    p.register(fd, select.POLLIN)
    assert p.poll(5000)
    try:
        consume_fd(fd)
    except BlockingIOError:
        pass


@pytest.fixture
def nixl_adapter():
    tmp = tempfile.mkdtemp(prefix="idle_ttl_nixl_")
    buf = torch.empty(PAGE * 16, dtype=torch.uint8)
    cfg = DynamicNixlStoreL2AdapterConfig(
        backend="POSIX",
        backend_params={
            "file_path": tmp,
            "use_direct_io": "false",
            "max_capacity_gb": "0.001",
            "shard_dirs": "true",
        },
    )
    cfg.persist_config = PersistConfig(persist_enabled=True)
    adpt = DynamicNixlStoreL2Adapter(
        cfg, L1MemoryDesc(ptr=buf.data_ptr(), size=buf.numel(), align_bytes=PAGE)
    )
    yield adpt, buf, Path(tmp)
    adpt.close()
    shutil.rmtree(tmp, ignore_errors=True)


def test_u7_l2_real_adapter_whole_chunk_and_pin(nixl_adapter):
    adpt, buf, root = nixl_adapter
    clock = FakeClock()
    st = L2AdapterEvictionState(
        0, adpt, EvictionConfig("LRU", 0.9, 0.25, idle_ttl_seconds=300), clock=clock
    )
    ctl = L2EvictionController([st])
    keys = request(0, 1)  # 2 chunks x 2 groups
    tid = adpt.submit_store_task(keys, [_obj(buf, i) for i in range(len(keys))])
    _wait(adpt.get_store_event_fd())
    assert adpt.pop_completed_store_tasks()[tid].is_successful()
    assert len(list(root.rglob("*.bin"))) == 4

    # Pin one object of chunk 1 through a lookup, like a request in flight.
    tid = adpt.submit_lookup_and_lock_task([key(1, 1)], {0: MemoryLayoutDesc([], [])})
    _wait(adpt.get_lookup_and_lock_event_fd())
    assert adpt.query_lookup_and_lock_result(tid).test(0)

    clock.t = 400
    assert ctl.run_idle_ttl_pass(st) == 1  # chunk 0 deleted, chunk 1 pinned
    files = sorted(p.name for p in root.rglob("*.bin"))
    assert len(files) == 2 and all(ObjectKey.IntHash2Bytes(1).hex() in f for f in files)

    adpt.submit_unlock([key(1, 1)])
    done = __import__("threading").Event()
    adpt._loop.call_soon_threadsafe(done.set)
    assert done.wait(5)
    assert ctl.run_idle_ttl_pass(st) == 1
    assert list(root.rglob("*.bin")) == []
    assert st.idle_tracker is not None and len(st.idle_tracker) == 0


# =============================================================================
# U8: config
# =============================================================================


def test_u8_cli_flags_default_off_and_distinct_from_lock_ttl():
    cfg = parse_args(["--l1-size-gb", "1", "--eviction-policy", "LRU"])
    assert cfg.eviction_config.idle_ttl_seconds == 0
    assert cfg.eviction_config.idle_ttl_batch == 256
    assert cfg.l1_manager_config.read_ttl_seconds == 300  # lock TTLs unchanged
    assert cfg.l1_manager_config.write_ttl_seconds == 600
    cfg = parse_args(
        [
            "--l1-size-gb",
            "1",
            "--eviction-policy",
            "LRU",
            "--l1-idle-ttl-seconds",
            "120",
            "--l1-idle-ttl-batch",
            "8",
        ]
    )
    assert cfg.eviction_config.idle_ttl_seconds == 120
    assert cfg.eviction_config.idle_ttl_batch == 8
    assert cfg.l1_manager_config.read_ttl_seconds == 300


@pytest.mark.parametrize(
    "kwargs",
    [
        {"idle_ttl_seconds": -1},
        {"idle_ttl_batch": 0},
        {"idle_ttl_batch": -5},
        {"idle_ttl_seconds": "60"},
        {"idle_ttl_batch": True},
    ],
)
def test_u8_invalid_values_rejected(kwargs):
    with pytest.raises(ValueError):
        EvictionConfig("LRU", **kwargs)


def test_u8_l2_eviction_block_parsing():
    parse = L2AdapterConfigBase._parse_eviction_config
    base = {"eviction_policy": "LRU", "trigger_watermark": 0.9, "eviction_ratio": 0.25}
    cfg = parse({"eviction": base})
    assert (cfg.idle_ttl_seconds, cfg.idle_ttl_batch, cfg.touch_on_l1_hit) == (
        0,
        256,
        False,
    )
    cfg = parse({"eviction": {**base, "idle_ttl_seconds": None}})
    assert cfg.idle_ttl_seconds == 0
    cfg = parse(
        {
            "eviction": {
                **base,
                "idle_ttl_seconds": 300,
                "idle_ttl_batch": 16,
                "touch_on_l1_hit": True,
            }
        }
    )
    assert (cfg.idle_ttl_seconds, cfg.idle_ttl_batch, cfg.touch_on_l1_hit) == (
        300,
        16,
        True,
    )
    assert cfg.trigger_watermark == 0.9 and cfg.eviction_ratio == 0.25
    with pytest.raises(ValueError):
        parse({"eviction": {**base, "touch_on_l1_hit": "yes"}})
    with pytest.raises(ValueError):
        parse({"eviction": {**base, "idle_ttl_batch": 0}})


def test_u8_l2_ttl_must_not_be_shorter_than_l1():
    l1 = [EvictionConfig("LRU", idle_ttl_seconds=120)]
    _validate_idle_ttl_tiers(l1, [EvictionConfig("LRU", idle_ttl_seconds=300)])
    _validate_idle_ttl_tiers(l1, [EvictionConfig("LRU")])  # L2 TTL off is fine
    _validate_idle_ttl_tiers(
        [EvictionConfig("LRU")], [EvictionConfig("LRU", idle_ttl_seconds=5)]
    )
    with pytest.raises(ValueError, match="must be >="):
        _validate_idle_ttl_tiers(l1, [EvictionConfig("LRU", idle_ttl_seconds=60)])


def test_u8_watermark_and_ratio_untouched():
    cfg = EvictionConfig("LRU", 0.9, 0.25, idle_ttl_seconds=60)
    assert (cfg.trigger_watermark, cfg.eviction_ratio) == (0.9, 0.25)


def test_u8_l1_listener_mirrors_policy_bridge():
    tr = IdleTTLTracker(clock=FakeClock())
    li = L1IdleTTLListener(tr)
    li.on_l1_keys_reserved_write([key(0)])
    li.on_l1_keys_read_finished([key(0)])
    li.on_l1_keys_reserved_read([key(0)])
    assert len(tr) == 1
    li.on_l1_keys_deleted_by_manager([key(0)])
    assert len(tr) == 0

# SPDX-License-Identifier: Apache-2.0
"""Non-blocking IPC release of unregistered / reaped instances (LMCache#4676).

A released instance whose transfer stream still has queued work must not be
unmapped (that hangs the driver) and must not block its caller (that wedges
the reaper or the MQ loop). These tests drive the staged release through the
module's public surface with a fake clock, a stubbed stream and a stubbed
release thread, so each stage is stepped explicitly:

    drained -> unmap now (only when nothing else is pending)
    busy    -> deferred; after the drain grace, force-complete held events
               every cycle (registration poisoned: its store commits are
               dropped), unless the backend cannot force or the oldest wait
               is too old for its L1 reservation to still be valid
    still busy after the stuck timeout -> quarantine, or exit the process
"""

# Standard
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock
import threading
import time

# Third Party
import pytest

# First Party
from lmcache.v1.multiprocess.modules.lmcache_driven_transfer import (
    STUCK_RELEASE_EXIT_CODE,
    IPCReleaseConfig,
    LMCacheDrivenTransferModule,
)
from lmcache.v1.platform.devices.cuda import timeline_semaphore_event_ipc as ts
import lmcache.v1.multiprocess.modules.lmcache_driven_transfer as gpu_mod


class _Clock:
    """Monotonic clock the tests advance by hand."""

    def __init__(self) -> None:
        self.now = 1000.0

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


class _Stream:
    """Transfer stream whose drained state the test controls."""

    def __init__(self, drained: bool) -> None:
        self.drained = drained

    def query(self) -> bool:
        return self.drained


class _ForcingBackend:
    """Event backend with the timeline-semaphore force path."""

    def __init__(self, stream: _Stream, frees_stream: bool = True) -> None:
        self.forced: list[list[object]] = []
        self._stream = stream
        self._frees_stream = frees_stream

    def check_event_support(self, device: object) -> None:
        pass

    def force_complete_events(self, events: list[object]) -> int:
        self.forced.append(list(events))
        if self._frees_stream:
            self._stream.drained = True
        return len(events)


class _LegacyBackend:
    """Event backend without a force path (CUDA interprocess events)."""

    def check_event_support(self, device: object) -> None:
        pass


@pytest.fixture
def clock(monkeypatch) -> _Clock:
    fake = _Clock()
    monkeypatch.setattr(
        gpu_mod,
        "time",
        SimpleNamespace(
            monotonic=fake.monotonic,
            perf_counter=time.perf_counter,
            sleep=fake.sleep,
        ),
    )
    return fake


@pytest.fixture
def dev(monkeypatch) -> MagicMock:
    fake = MagicMock(name="torch_dev")
    monkeypatch.setattr(gpu_mod, "torch_dev", fake)
    return fake


def _module(monkeypatch, **cfg) -> LMCacheDrivenTransferModule:
    monkeypatch.setattr(gpu_mod, "DeviceHostFuncDispatcher", MagicMock())
    # The release thread is stepped by hand through _release_cycle().
    monkeypatch.setattr(gpu_mod, "create_periodic_thread", MagicMock())
    config = IPCReleaseConfig(
        drain_grace_seconds=cfg.get("grace", 30.0),
        stuck_timeout_seconds=cfg.get("stuck", 60.0),
        stuck_action=cfg.get("action", "quarantine"),
        poll_interval_seconds=1.0,
        max_force_wait_age_seconds=cfg.get("max_age", 300.0),
    )
    return LMCacheDrivenTransferModule(MagicMock(name="ctx"), config)


def _register(
    module,
    monkeypatch,
    clock: _Clock,
    instance_id: int,
    stream: _Stream,
    backend,
    device: str = "cuda:0",
):
    cache_context = MagicMock(name=f"cache_context-{instance_id}")
    cache_context.num_layers = 1
    cache_context.stream = stream
    cache_context.device = device
    cache_context.held_imports_snapshot.return_value = (
        [f"event-{instance_id}"],
        clock.now,
    )
    monkeypatch.setattr(gpu_mod, "create_cache_context", lambda *a, **kw: cache_context)
    monkeypatch.setattr(gpu_mod, "get_event_ipc_backend", lambda device: backend)
    monkeypatch.setattr(gpu_mod, "get_layout_desc", lambda *a, **kw: MagicMock())
    module.register_kv_cache(
        instance_id,
        kv_caches=MagicMock(),
        model_name="m",
        world_size=1,
        engine_type=MagicMock(),
        layout_hints=MagicMock(),
        engine_group_infos=[],
    )
    return cache_context


def _generation(module, instance_id: int) -> int:
    return module.context_entries_snapshot()[instance_id].generation


# ----------------------------------------------------------------- staging --


def test_drained_stream_is_unmapped_inline(monkeypatch, clock, dev) -> None:
    module = _module(monkeypatch)
    stream = _Stream(drained=True)
    ctx = _register(module, monkeypatch, clock, 1, stream, _ForcingBackend(stream))

    module.unregister_kv_cache(1)

    ctx.close.assert_called_once()
    dev.empty_cache.assert_called_once()
    assert module.release_status()["pending_instance_ids"] == []
    gpu_mod.create_periodic_thread.assert_not_called()


def test_busy_stream_is_deferred_not_closed(monkeypatch, clock, dev) -> None:
    """The caller returns at once; the mapping stays open until drained."""
    module = _module(monkeypatch)
    stream = _Stream(drained=False)
    ctx = _register(module, monkeypatch, clock, 1, stream, _ForcingBackend(stream))

    module.unregister_kv_cache(1)

    ctx.close.assert_not_called()
    dev.empty_cache.assert_not_called()
    assert module.release_status()["pending_instance_ids"] == [1]
    gpu_mod.create_periodic_thread.return_value.start.assert_called_once()

    stream.drained = True
    module._release_cycle()

    ctx.close.assert_called_once()
    dev.empty_cache.assert_called_once()
    assert module.release_status()["pending_instance_ids"] == []


def test_reaper_does_not_block_on_busy_stream(monkeypatch, clock, dev) -> None:
    module = _module(monkeypatch)
    stream = _Stream(drained=False)
    ctx = _register(module, monkeypatch, clock, 1, stream, _ForcingBackend(stream))
    clock.now += 500.0

    reaped = module.reap_stale_instances(reap_timeout_s=60.0, registration_grace_s=60.0)

    assert reaped == [1]
    ctx.close.assert_not_called()
    assert module.tracked_instance_count() == 0
    assert module.release_status()["pending_instance_ids"] == [1]


# ------------------------------------------------------- forcing + poisoning --


def test_force_after_grace_releases_and_poisons(monkeypatch, clock, dev) -> None:
    module = _module(monkeypatch, grace=30.0)
    stream = _Stream(drained=False)
    backend = _ForcingBackend(stream)
    ctx = _register(module, monkeypatch, clock, 1, stream, backend)
    gen = _generation(module, 1)
    module.unregister_kv_cache(1)

    clock.now += 29.0
    module._release_cycle()
    assert backend.forced == []  # still inside the grace

    clock.now += 2.0
    module._release_cycle()  # forces; the stream drains by the next check
    assert backend.forced == [["event-1"]]
    ctx.close.assert_not_called()

    # A commit for the forced registration is dropped, not written to L1.
    storage = module.context.storage_manager
    module._finish_write_unless_poisoned((gen, [(0, ["k"])]))
    storage.finish_write_by_owner.assert_not_called()
    # Other registrations still commit.
    module._finish_write_unless_poisoned((gen + 1, [(0, ["k2"])]))
    storage.finish_write_by_owner.assert_called_once_with([(0, ["k2"])])

    module._release_cycle()
    ctx.close.assert_called_once()
    status = module.release_status()
    assert status["forced"] == 1
    assert status["dropped_commits"] == 1
    assert status["pending_instance_ids"] == []


def test_force_repeats_every_cycle_while_busy(monkeypatch, clock, dev) -> None:
    """Waits queued after the first force (a transfer that took the entry
    before it was popped) are forced on a later cycle, not left stuck."""
    module = _module(monkeypatch, grace=1.0, stuck=60.0)
    stream = _Stream(drained=False)
    backend = _ForcingBackend(stream, frees_stream=False)
    _register(module, monkeypatch, clock, 1, stream, backend)
    module.unregister_kv_cache(1)

    for _ in range(3):
        clock.now += 2.0
        module._release_cycle()

    assert len(backend.forced) == 3
    assert module.release_status()["forced"] == 1  # counted once


def test_too_old_wait_is_never_forced(monkeypatch, clock, dev) -> None:
    """Forcing a STORE whose L1 write reservation may have expired would let
    the copy write into a buffer that now belongs to another key."""
    module = _module(monkeypatch, grace=30.0, stuck=60.0, max_age=300.0)
    stream = _Stream(drained=False)
    backend = _ForcingBackend(stream)
    ctx = _register(module, monkeypatch, clock, 1, stream, backend)
    # The pending wait was held 400s before the release.
    ctx.held_imports_snapshot.return_value = (["event-1"], clock.now - 400.0)
    module.unregister_kv_cache(1)

    clock.now += 31.0
    module._release_cycle()
    clock.now += 61.0
    summary = module._release_cycle()

    assert backend.forced == []
    assert summary.success is False
    ctx.close.assert_not_called()
    status = module.release_status()
    assert status["force_blocked_reasons"] == {"1": "too_old"}
    assert status["stuck_instance_ids"] == [1]


def test_reregistered_instance_is_not_poisoned(monkeypatch, clock, dev) -> None:
    """A falsely reaped worker re-registers under the same instance ID; its
    new registration's commits must still land."""
    module = _module(monkeypatch, grace=1.0)
    stream = _Stream(drained=False)
    _register(module, monkeypatch, clock, 1, stream, _ForcingBackend(stream))
    old_gen = _generation(module, 1)
    module.unregister_kv_cache(1)
    clock.now += 2.0
    module._release_cycle()  # forced -> old registration poisoned

    stream2 = _Stream(drained=True)
    _register(module, monkeypatch, clock, 1, stream2, _ForcingBackend(stream2))
    new_gen = _generation(module, 1)
    assert new_gen != old_gen

    storage = module.context.storage_manager
    module._finish_write_unless_poisoned((new_gen, [(0, ["fresh"])]))
    storage.finish_write_by_owner.assert_called_once_with([(0, ["fresh"])])


def test_same_instance_released_twice_closes_both(monkeypatch, clock, dev) -> None:
    module = _module(monkeypatch)
    stream1 = _Stream(drained=False)
    ctx1 = _register(module, monkeypatch, clock, 1, stream1, _ForcingBackend(stream1))
    module.unregister_kv_cache(1)
    stream2 = _Stream(drained=False)
    ctx2 = _register(module, monkeypatch, clock, 1, stream2, _ForcingBackend(stream2))
    module.unregister_kv_cache(1)
    assert module.release_status()["pending_instance_ids"] == [1, 1]

    stream1.drained = stream2.drained = True
    module._release_cycle()

    ctx1.close.assert_called_once()
    ctx2.close.assert_called_once()


def test_poison_expires_after_release(monkeypatch, clock, dev) -> None:
    module = _module(monkeypatch, grace=1.0)
    stream = _Stream(drained=False)
    _register(module, monkeypatch, clock, 1, stream, _ForcingBackend(stream))
    gen = _generation(module, 1)
    module.unregister_kv_cache(1)
    clock.now += 2.0
    module._release_cycle()
    module._release_cycle()

    clock.now += gpu_mod._POISON_RETENTION_SECONDS + 1.0
    module._release_cycle()

    module._finish_write_unless_poisoned((gen, [(0, ["k"])]))
    module.context.storage_manager.finish_write_by_owner.assert_called_once()


# ------------------------------------------------------ device isolation --


def test_busy_release_blocks_unmaps_on_its_device_only(monkeypatch, clock, dev) -> None:
    """Freeing or unmapping next to a stream that never drains can block, so a
    busy release holds back other releases on the same device, but not on
    another device; and empty_cache waits until nothing is pending."""
    module = _module(monkeypatch, grace=1000.0)
    stuck = _Stream(drained=False)
    _register(module, monkeypatch, clock, 1, stuck, _LegacyBackend(), "cuda:0")
    module.unregister_kv_cache(1)

    same = _Stream(drained=True)
    ctx_same = _register(
        module, monkeypatch, clock, 2, same, _LegacyBackend(), "cuda:0"
    )
    other = _Stream(drained=True)
    ctx_other = _register(
        module, monkeypatch, clock, 3, other, _LegacyBackend(), "cuda:1"
    )
    module.unregister_kv_cache(2)  # deferred: a release is already pending
    module.unregister_kv_cache(3)
    ctx_same.close.assert_not_called()
    ctx_other.close.assert_not_called()

    module._release_cycle()

    ctx_same.close.assert_not_called()  # held back by the busy cuda:0 stream
    ctx_other.close.assert_called_once()
    dev.empty_cache.assert_not_called()
    dev.ipc_collect.assert_called()
    assert sorted(module.release_status()["pending_instance_ids"]) == [1, 2]


# --------------------------------------------------------------- stuck path --


def test_unforceable_backend_is_quarantined(monkeypatch, clock, dev) -> None:
    """Legacy CUDA IPC events cannot be forced: the entry stays mapped."""
    module = _module(monkeypatch, grace=30.0, stuck=60.0)
    stream = _Stream(drained=False)
    ctx = _register(module, monkeypatch, clock, 1, stream, _LegacyBackend())
    module.unregister_kv_cache(1)

    clock.now += 31.0
    module._release_cycle()
    clock.now += 61.0
    summary = module._release_cycle()

    assert summary.success is False
    ctx.close.assert_not_called()
    dev.empty_cache.assert_not_called()
    status = module.release_status()
    assert status["force_blocked_reasons"] == {"1": "unforceable"}
    assert status["stuck_instance_ids"] == [1]


def test_stuck_release_exits_when_configured(monkeypatch, clock, dev) -> None:
    module = _module(monkeypatch, grace=1.0, stuck=1.0, action="exit")
    stream = _Stream(drained=False)
    backend = _ForcingBackend(stream, frees_stream=False)
    _register(module, monkeypatch, clock, 1, stream, backend)
    module.unregister_kv_cache(1)
    exits: list[int] = []

    def fake_exit(code: int) -> None:
        exits.append(code)
        raise SystemExit(code)

    monkeypatch.setattr(gpu_mod.os, "_exit", fake_exit)
    monkeypatch.setattr(gpu_mod.os, "write", lambda fd, data: len(data))

    clock.now += 2.0
    module._release_cycle()  # forced, still busy
    clock.now += 2.0
    with pytest.raises(SystemExit):
        module._release_cycle()
    assert exits == [STUCK_RELEASE_EXIT_CODE]


def test_overlapping_cycles_do_not_raise(monkeypatch, clock, dev) -> None:
    """A cycle that cannot get the cycle lock in time gives up quietly."""
    module = _module(monkeypatch)
    stream = _Stream(drained=False)
    _register(module, monkeypatch, clock, 1, stream, _ForcingBackend(stream))
    module.unregister_kv_cache(1)
    stream.drained = True

    module._release_cycle_lock.acquire()
    try:
        summary = module._release_cycle(lock_timeout=0.0)
    finally:
        module._release_cycle_lock.release()
    assert summary.message == "cycle busy"
    module._release_cycle()
    assert module.release_status()["pending_instance_ids"] == []


# ------------------------------------------------------------------ close --


def test_close_forces_without_grace_and_never_exits(monkeypatch, clock, dev) -> None:
    module = _module(monkeypatch, grace=30.0, stuck=5.0, action="exit")
    monkeypatch.setattr(gpu_mod.os, "_exit", MagicMock(side_effect=AssertionError))
    stream = _Stream(drained=False)
    backend = _ForcingBackend(stream)
    ctx = _register(module, monkeypatch, clock, 1, stream, backend)

    module.close()

    assert backend.forced == [["event-1"]]
    ctx.close.assert_called_once()


def test_close_gives_up_on_stuck_stream(monkeypatch, clock, dev) -> None:
    module = _module(monkeypatch, stuck=5.0)
    stream = _Stream(drained=False)
    ctx = _register(
        module,
        monkeypatch,
        clock,
        1,
        stream,
        _ForcingBackend(stream, frees_stream=False),
    )
    start = clock.now

    module.close()

    ctx.close.assert_not_called()
    assert clock.now - start <= 5.0 + 1.0


# ------------------------------------------- timeline force_complete_events --


def _imported(base: int, slot: int, seq: int, device: int = 0):
    return ts.TimelineSemaphoreEvent(
        origin=ts.TimelineSemaphoreEventOrigin.IMPORTED,
        device_index=device,
        base_ptr=base,
        slot_offset=slot,
        seq=seq,
    )


def test_force_complete_events_writes_max_seq_per_slot(monkeypatch) -> None:
    writes: list[tuple[int, int]] = []
    fake_driver = SimpleNamespace(
        CUstream=lambda raw: raw,
        CUdeviceptr=lambda ptr: ptr,
        CUstreamWriteValue_flags=SimpleNamespace(CU_STREAM_WRITE_VALUE_DEFAULT=0),
        cuStreamWriteValue64=lambda stream, ptr, value, flags: (
            writes.append((ptr, value)) or (0,)
        ),
    )
    fake_runtime = SimpleNamespace(cudaStreamSynchronize=lambda stream: (0,))
    monkeypatch.setattr(
        ts, "_cuda", SimpleNamespace(driver=fake_driver, runtime=fake_runtime)
    )
    monkeypatch.setattr(ts, "_CHECK_CUDA", lambda result, what: None)
    monkeypatch.setattr(ts, "_raw_stream_handle", lambda stream, device: 7)
    monkeypatch.setattr(ts.torch.cuda, "device", lambda index: nullcontext())
    monkeypatch.setattr(ts, "_LOCAL_BUFFERS_BY_HANDLE", {b"own": 0x9000})
    backend = ts.TimelineSemaphoreEventIPCBackend()
    monkeypatch.setattr(
        backend, "_get_buffer", lambda index: SimpleNamespace(buffer_ops_stream="s")
    )
    # Slot at 0x1008 already reached 9: it must not be moved backwards.
    current = {0x1000: 0, 0x1008: 9}
    monkeypatch.setattr(
        backend, "_read_slot", lambda e: current[e.base_ptr + e.slot_offset]
    )
    local = ts.TimelineSemaphoreEvent(
        origin=ts.TimelineSemaphoreEventOrigin.LOCAL,
        device_index=0,
        base_ptr=0x5000,
        slot_offset=0,
        seq=3,
    )
    events = [
        _imported(0x1000, 0, 4),
        _imported(0x1000, 0, 6),  # same slot, higher seq wins
        _imported(0x1000, 8, 5),  # already satisfied (slot is 9)
        _imported(0x1000, 16, 0),  # never recorded
        _imported(0x9000, 0, 2),  # resolves to this process's own buffer
        local,
        "not-an-event",
    ]

    written = backend.force_complete_events(events)

    assert writes == [(0x1000, 6)]
    assert written == 1


def test_concurrent_release_cycles_are_serialized(monkeypatch, clock, dev) -> None:
    """The release thread and shutdown never run a cycle at the same time."""
    module = _module(monkeypatch)
    inside = threading.Event()
    proceed = threading.Event()
    seen: list[str] = []
    original = module._release_cycle_locked

    def slow(allow_exit: bool):
        seen.append("enter")
        inside.set()
        proceed.wait(5)
        seen.append("exit")
        return original(allow_exit)

    monkeypatch.setattr(module, "_release_cycle_locked", slow)
    worker = threading.Thread(target=module._release_cycle)
    worker.start()
    inside.wait(5)
    summary = module._release_cycle(lock_timeout=0.05)
    proceed.set()
    worker.join(5)
    assert summary.message == "cycle busy"
    assert seen == ["enter", "exit"]

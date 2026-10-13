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


class _Thread:
    """Release thread stand-in: records start, and wake() runs one cycle."""

    def __init__(self, execute_fn) -> None:
        self.execute_fn = execute_fn
        self.started = 0

    def start(self) -> None:
        self.started += 1

    def stop(self, *args, **kwargs) -> None:
        pass

    def wake(self) -> None:
        self.execute_fn()


def _module(monkeypatch, **cfg) -> LMCacheDrivenTransferModule:
    monkeypatch.setattr(gpu_mod, "DeviceHostFuncDispatcher", MagicMock())
    # Releases run on the release thread and unmaps on the unmapper thread;
    # here wake() runs one cycle inline and unmaps run inline, and further
    # cycles are stepped by hand through _release_cycle().
    threads: list[_Thread] = []

    def make_thread(name, interval, execute_fn, level=None, **_):
        threads.append(_Thread(execute_fn))
        return threads[-1]

    monkeypatch.setattr(gpu_mod, "create_periodic_thread", make_thread)
    config = IPCReleaseConfig(
        drain_grace_seconds=cfg.get("grace", 30.0),
        stuck_timeout_seconds=cfg.get("stuck", 60.0),
        stuck_action=cfg.get("action", "quarantine"),
        poll_interval_seconds=1.0,
        max_force_wait_age_seconds=cfg.get("max_age", 300.0),
    )
    module = LMCacheDrivenTransferModule(MagicMock(name="ctx"), config)
    module._submit_unmap = module._finalize_releases
    module._test_threads = threads
    return module


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
    # Like the real context: holds are dropped (by the stream's release
    # callbacks) by the time the stream has drained.
    cache_context._holds = {"value": ([f"event-{instance_id}"], clock.now)}
    cache_context.held_imports_snapshot.side_effect = lambda: (
        cache_context._holds["value"] if not stream.drained else ([], None)
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
    # Even a drained release is unmapped by the release path, never by the
    # caller (the MQ loop): one release thread was started for it.
    assert [t.started for t in module._test_threads] == [1]


def test_busy_stream_is_deferred_not_closed(monkeypatch, clock, dev) -> None:
    """The caller returns at once; the mapping stays open until drained."""
    module = _module(monkeypatch)
    stream = _Stream(drained=False)
    ctx = _register(module, monkeypatch, clock, 1, stream, _ForcingBackend(stream))

    module.unregister_kv_cache(1)

    ctx.close.assert_not_called()
    dev.empty_cache.assert_not_called()
    assert module.release_status()["pending_instance_ids"] == [1]
    assert [t.started for t in module._test_threads] == [1]

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
    ctx._holds["value"] = (["event-1"], clock.now - 400.0)
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
    """Unmapping next to a stream that never drains blocks until it drains,
    so a busy release holds back other releases on the same device, but not
    on another device; and empty_cache waits until nothing is pending."""
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
    module.unregister_kv_cache(2)
    module.unregister_kv_cache(3)
    module._release_cycle()

    ctx_same.close.assert_not_called()  # held back by the busy cuda:0 stream
    ctx_other.close.assert_called_once()
    dev.empty_cache.assert_not_called()
    dev.ipc_collect.assert_called()
    status = module.release_status()
    assert sorted(status["pending_instance_ids"]) == [1, 2]
    assert status["held_back_instance_ids"] == [2]

    stuck.drained = True
    module._release_cycle()
    ctx_same.close.assert_called_once()


def test_stale_wait_on_registered_instance_blocks_its_device(
    monkeypatch, clock, dev
) -> None:
    """A dead worker that is not reaped yet already has a stuck stream; a
    release on the same GPU must wait, one on another GPU must not."""
    module = _module(monkeypatch)
    dead = _Stream(drained=False)
    ctx_dead = _register(
        module, monkeypatch, clock, 1, dead, _LegacyBackend(), "cuda:0"
    )
    ctx_dead._holds["value"] = (["e"], clock.now)
    clock.now += 11.0  # past stale_wait_seconds (10s default)

    same = _Stream(drained=True)
    ctx_same = _register(
        module, monkeypatch, clock, 2, same, _LegacyBackend(), "cuda:0"
    )
    other = _Stream(drained=True)
    ctx_other = _register(
        module, monkeypatch, clock, 3, other, _LegacyBackend(), "cuda:1"
    )
    module.unregister_kv_cache(2)
    module.unregister_kv_cache(3)

    ctx_same.close.assert_not_called()
    ctx_other.close.assert_called_once()
    dev.empty_cache.assert_not_called()  # a stale wait exists somewhere

    dead.drained = True  # e.g. the worker was reaped and forced
    module._release_cycle()
    ctx_same.close.assert_called_once()


def _stop_unmappers(module) -> None:
    for unmap_queue in list(module._unmap_queues.values()):
        unmap_queue.put(None)


def _wait_for(predicate, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        time.sleep(0.01)
    return predicate()


def test_unregister_never_unmaps_on_the_caller_thread(monkeypatch, clock, dev) -> None:
    """With the real unmappers, UNREGISTER returns before the unmap, which
    runs on the device's own lmcache-mp-ipc-unmap thread."""
    module = _module(monkeypatch)
    del module._submit_unmap  # restore the real thread-backed unmappers
    stream = _Stream(drained=True)
    ctx = _register(module, monkeypatch, clock, 1, stream, _LegacyBackend())
    closed_on: list[str] = []
    ctx.close.side_effect = lambda: closed_on.append(threading.current_thread().name)

    module.unregister_kv_cache(1)

    assert _wait_for(lambda: bool(closed_on))
    assert closed_on == ["lmcache-mp-ipc-unmap-cuda:0"]
    assert _wait_for(lambda: module.release_status()["unmapping"] == 0)
    _stop_unmappers(module)


def test_blocked_unmap_on_one_gpu_does_not_delay_another(
    monkeypatch, clock, dev
) -> None:
    """An unmap stuck in the driver on cuda:0 must not hold up cuda:1."""
    module = _module(monkeypatch)
    del module._submit_unmap
    release_gpu0 = threading.Event()
    s0, s1 = _Stream(drained=True), _Stream(drained=True)
    ctx0 = _register(module, monkeypatch, clock, 1, s0, _LegacyBackend(), "cuda:0")
    ctx1 = _register(module, monkeypatch, clock, 2, s1, _LegacyBackend(), "cuda:1")
    ctx0.close.side_effect = lambda: release_gpu0.wait(10)

    module.unregister_kv_cache(1)
    assert _wait_for(lambda: ctx0.close.called)  # cuda:0 unmap now blocked
    module.unregister_kv_cache(2)

    assert _wait_for(lambda: ctx1.close.called)
    assert module.release_status()["unmapping"] == 1  # only cuda:0 in flight
    release_gpu0.set()
    assert _wait_for(lambda: module.release_status()["unmapping"] == 0)
    _stop_unmappers(module)


def test_unmapper_rechecks_gating_before_closing(monkeypatch, clock, dev) -> None:
    """A device can get a stuck stream after a job was queued; the unmapper
    waits for it to clear instead of blocking in the driver."""
    module = _module(monkeypatch)
    del module._submit_unmap
    dead = _Stream(drained=False)
    ctx_dead = _register(
        module, monkeypatch, clock, 1, dead, _LegacyBackend(), "cuda:0"
    )
    ctx_dead._holds["value"] = (["e"], clock.now - 60.0)
    same = _Stream(drained=True)
    ctx_same = _register(
        module, monkeypatch, clock, 2, same, _LegacyBackend(), "cuda:0"
    )

    # Queue the job directly, as if gating had passed a moment earlier.
    with module._lock:
        entry = module._cache_contexts.pop(2)
    module._submit_unmap([entry])
    del entry
    time.sleep(0.3)
    ctx_same.close.assert_not_called()
    assert module.release_status()["stale_wait_instance_ids"] == [1]

    dead.drained = True
    assert _wait_for(lambda: ctx_same.close.called)
    _stop_unmappers(module)


def test_failed_registration_rollback_runs_off_the_caller(monkeypatch, clock) -> None:
    """Wrappers opened by a failed context construction are closed by the
    device unmapper, not by REGISTER on the MQ loop; with no server module the
    platform helper closes them inline."""
    # First Party
    from lmcache.v1.platform.base import cache_context as base_cc

    module = _module(monkeypatch)
    del module._submit_unmap
    closed_on: list[str] = []
    wrapper = MagicMock()
    wrapper.close.side_effect = lambda: closed_on.append(
        threading.current_thread().name
    )

    base_cc.close_wrappers_after_failed_init([wrapper], "cuda:3")
    assert _wait_for(lambda: bool(closed_on))
    assert closed_on == ["lmcache-mp-ipc-unmap-cuda:3"]
    _stop_unmappers(module)

    base_cc.set_failed_init_closer(None)
    inline = MagicMock()
    base_cc.close_wrappers_after_failed_init([inline], "cuda:3")
    inline.close.assert_called_once()


def test_failed_registration_after_construction_unmaps_context(
    monkeypatch, clock, dev
) -> None:
    """A REGISTER failing after the context mapped the worker's KV pool (here:
    event support check) hands the context to the unmapper, not the registry."""
    module = _module(monkeypatch)
    backend = _LegacyBackend()
    backend.check_event_support = MagicMock(side_effect=RuntimeError("no events"))
    with pytest.raises(RuntimeError, match="no events"):
        _register(module, monkeypatch, clock, 1, _Stream(drained=True), backend)

    ctx = gpu_mod.create_cache_context()
    ctx.close.assert_called_once()
    assert module.tracked_instance_count() == 0


# ---------------------------------------------------------- allocation guard --

_MIB = 1 << 20


def test_allocator_segment_bytes_mirrors_caching_allocator() -> None:
    seg = gpu_mod._allocator_segment_bytes
    assert seg(8) == 2 * _MIB
    assert seg(_MIB) == 2 * _MIB
    assert seg(_MIB + 1) == 20 * _MIB
    assert seg(8 * _MIB) == 20 * _MIB
    assert seg(10 * _MIB) == 10 * _MIB
    assert seg(10 * _MIB + 1) == 12 * _MIB
    assert seg(300 * _MIB) == 300 * _MIB


def _cuda(index: int):
    # Third Party
    import torch

    return torch.device("cuda", index)


def test_alloc_guard_allows_when_a_segment_fits(monkeypatch, clock, dev) -> None:
    module = _module(monkeypatch)
    dead = _Stream(drained=False)
    ctx = _register(module, monkeypatch, clock, 1, dead, _LegacyBackend())
    ctx._holds["value"] = (["e"], clock.now - 60.0)
    dev.mem_get_info.return_value = (20 * _MIB + 64 * _MIB, 80 << 30)

    module._check_device_alloc(_cuda(0), 8 * _MIB)  # 20 MiB segment + margin fits

    dev.mem_get_info.return_value = (80 * _MIB, 80 << 30)
    with pytest.raises(RuntimeError, match="Refusing to allocate"):
        module._check_device_alloc(_cuda(0), 8 * _MIB)
    assert module.release_status()["alloc_refused"] == 1


def test_alloc_guard_refuses_next_to_a_stuck_wait_only_on_its_device(
    monkeypatch, clock, dev
) -> None:
    """Low memory + a registered instance with a wait pending past the guard
    age on the same device: refuse. Another device, or a fresh wait: allow."""
    module = _module(monkeypatch)
    dead = _Stream(drained=False)
    ctx = _register(module, monkeypatch, clock, 1, dead, _LegacyBackend(), "cuda:0")
    dev.mem_get_info.return_value = (10 * _MIB, 80 << 30)

    ctx._holds["value"] = (["e"], clock.now - 0.5)
    module._check_device_alloc(_cuda(0), 8 * _MIB)  # live wait: allowed

    ctx._holds["value"] = (["e"], clock.now - 2.0)
    with pytest.raises(RuntimeError, match=r"instance\(s\) \[1\] have a wait"):
        module._check_device_alloc(_cuda(0), 8 * _MIB)
    module._check_device_alloc(_cuda(1), 8 * _MIB)  # other GPU unaffected

    dead.drained = True  # the wait completed
    module._check_device_alloc(_cuda(0), 8 * _MIB)
    assert module.release_status()["alloc_refused"] == 1


def test_alloc_guard_refuses_next_to_an_undrained_release(
    monkeypatch, clock, dev
) -> None:
    module = _module(monkeypatch)
    busy = _Stream(drained=False)
    _register(module, monkeypatch, clock, 1, busy, _LegacyBackend(), "cuda:0")
    module.unregister_kv_cache(1)
    dev.mem_get_info.return_value = (10 * _MIB, 80 << 30)

    with pytest.raises(RuntimeError, match=r"release of instance\(s\) \[1\]"):
        module._check_device_alloc(_cuda(0), 300 * _MIB)
    module._check_device_alloc(_cuda(1), 300 * _MIB)


def test_alloc_guard_ignores_non_cuda_devices(monkeypatch, clock, dev) -> None:
    module = _module(monkeypatch)
    module._check_device_alloc(SimpleNamespace(type="cpu"), 1 << 40)
    dev.mem_get_info.assert_not_called()


def test_closing_one_module_keeps_another_modules_hooks(
    monkeypatch, clock, dev
) -> None:
    # First Party
    from lmcache.v1.platform.base import cache_context as base_cc

    old = _module(monkeypatch)
    new = _module(monkeypatch)  # installs its hooks over old's
    old.close()
    assert base_cc._device_alloc_guard() == new._check_device_alloc
    assert base_cc._failed_init_closer() == new._submit_rollback_close
    new.close()
    assert base_cc._device_alloc_guard is None
    assert base_cc._failed_init_closer is None


def test_alloc_guard_is_the_platform_hook_until_close(monkeypatch, clock, dev) -> None:
    # First Party
    from lmcache.v1.platform.base import cache_context as base_cc

    module = _module(monkeypatch)
    calls: list[tuple] = []
    monkeypatch.setattr(
        module, "_check_device_alloc", lambda d, n: calls.append((str(d), n))
    )
    # The hook holds the bound method installed at construction, not the patch.
    dev.mem_get_info.return_value = (1 << 40, 1 << 40)
    base_cc.check_device_alloc(_cuda(0), 4096)
    dev.mem_get_info.assert_called_once()

    module.close()
    dev.mem_get_info.reset_mock()
    base_cc.check_device_alloc(_cuda(0), 4096)
    dev.mem_get_info.assert_not_called()


class _NeverQueryStream(_Stream):
    """A stream whose query() would block: behind a never-completing wait its
    work queue fills up and cudaStreamQuery blocks (holding the GIL)."""

    def query(self) -> bool:
        raise AssertionError("queried a stream that still has a held wait")


def test_stream_with_held_waits_is_never_queried(monkeypatch, clock, dev) -> None:
    """Release, gating, the alloc guard and /status must decide "not drained"
    host-side while a worker-event wait is held, without touching the stream."""
    module = _module(monkeypatch)
    stream = _NeverQueryStream(drained=False)
    ctx = _register(module, monkeypatch, clock, 1, stream, _LegacyBackend())
    ctx._holds["value"] = (["e"], clock.now - 60.0)
    other = _Stream(drained=True)
    _register(module, monkeypatch, clock, 2, other, _LegacyBackend(), "cuda:0")
    dev.mem_get_info.return_value = (1 << 20, 80 << 30)

    assert module.release_status()["stale_wait_instance_ids"] == [1]
    with pytest.raises(RuntimeError, match="have a wait pending"):
        module._check_device_alloc(_cuda(0), 300 << 20)
    module.unregister_kv_cache(2)  # same device: held back, not queried
    module.unregister_kv_cache(1)
    clock.now += 100.0
    module._release_cycle()
    assert sorted(module.release_status()["pending_instance_ids"]) == [1, 2]


def test_release_callback_drops_the_hold_of_a_reaped_instance(
    monkeypatch, clock, dev
) -> None:
    """A wait that retires after its instance was reaped still releases its
    hold, on the context now waiting in the pending release."""
    # Third Party
    import torch

    # First Party
    from lmcache.v1.platform.base.cache_context import BaseCacheContext

    module = _module(monkeypatch)
    context = MagicMock(name="context")
    context.device = torch.device("cuda", 0)
    context.stream = _Stream(drained=True)
    context._held_imports = {}
    context._held_import_times = {}
    context._held_imports_lock = threading.Lock()
    context.hold_imported_event = lambda e: BaseCacheContext.hold_imported_event(
        context, e
    )
    context.release_imported_event = lambda t: BaseCacheContext.release_imported_event(
        context, t
    )
    context.held_imports_snapshot = lambda: BaseCacheContext.held_imports_snapshot(
        context
    )
    monkeypatch.setattr(gpu_mod, "create_cache_context", lambda *a, **kw: context)
    monkeypatch.setattr(gpu_mod, "get_event_ipc_backend", lambda d: _LegacyBackend())
    monkeypatch.setattr(gpu_mod, "get_layout_desc", lambda *a, **kw: MagicMock())
    module.register_kv_cache(1, MagicMock(), "m", 1, MagicMock(), MagicMock(), [])
    token = context.hold_imported_event("event")
    clock.now += 500.0
    module.reap_stale_instances(reap_timeout_s=60.0, registration_grace_s=60.0)
    assert module.release_status()["pending_instance_ids"] == [1]

    module._release_imported_event((1, token))

    assert context.held_imports_snapshot() == ([], None)
    module._release_cycle()
    context.close.assert_called_once()


def _real_hold_context(monkeypatch):
    """A context mock whose hold bookkeeping is BaseCacheContext's own."""
    # Third Party
    import torch

    # First Party
    from lmcache.v1.platform.base.cache_context import BaseCacheContext

    context = MagicMock(name="context")
    context.device = torch.device("cuda", 0)
    context.stream = _NeverQueryStream(drained=True)
    context._held_imports = {}
    context._held_import_times = {}
    context._held_imports_lock = threading.Lock()
    for name in ("hold_imported_event", "release_imported_event"):
        method = getattr(BaseCacheContext, name)
        setattr(context, name, lambda *a, _m=method: _m(context, *a))
    context.held_imports_snapshot = lambda: BaseCacheContext.held_imports_snapshot(
        context
    )
    monkeypatch.setattr(gpu_mod, "create_cache_context", lambda *a, **kw: context)
    monkeypatch.setattr(gpu_mod, "get_event_ipc_backend", lambda d: _LegacyBackend())
    monkeypatch.setattr(gpu_mod, "get_layout_desc", lambda *a, **kw: MagicMock())
    return context


def test_callback_between_registry_pop_and_pending_insert_is_not_lost(
    monkeypatch, clock, dev
) -> None:
    """A release callback landing while UNREGISTER has popped the entry but not
    queued its pending release yet must still drop the hold; otherwise an idle
    stream would look busy forever (forced, poisoned, then stuck)."""
    module = _module(monkeypatch)
    context = _real_hold_context(monkeypatch)
    module.register_kv_cache(1, MagicMock(), "m", 1, MagicMock(), MagicMock(), [])
    token = context.hold_imported_event("event")
    module._ctx.layout_desc_registry.unregister.side_effect = lambda *a: (
        module._release_imported_event((1, token))
    )

    module.unregister_kv_cache(1)

    assert context.held_imports_snapshot() == ([], None)
    context.close.assert_called_once()
    status = module.release_status()
    assert status["pending_instance_ids"] == [] and status["forced"] == 0


def test_close_keeps_dispatching_callbacks_until_releases_ran(
    monkeypatch, clock, dev
) -> None:
    module = _module(monkeypatch)
    order: list[str] = []
    module._device_host_func_dispatcher.stop.side_effect = lambda: order.append(
        "dispatcher.stop"
    )
    real_cycle = module._release_cycle

    def cycle(*args, **kwargs):
        order.append("release_cycle")
        return real_cycle(*args, **kwargs)

    monkeypatch.setattr(module, "_release_cycle", cycle)
    _register(module, monkeypatch, clock, 1, _Stream(drained=True), _LegacyBackend())
    module.close()
    assert order.index("dispatcher.stop") > order.index("release_cycle")


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


def test_failed_force_is_retried_with_backoff(monkeypatch, clock, dev) -> None:
    """A force that raises (e.g. a transient driver error) is retried with a
    growing delay instead of leaving the release stuck for good."""
    module = _module(monkeypatch, grace=30.0, stuck=60.0)
    stream = _Stream(drained=False)
    backend = _ForcingBackend(stream)
    real_force = backend.force_complete_events
    failures = {"left": 2}

    def flaky_force(events: list[object]) -> int:
        if failures["left"]:
            failures["left"] -= 1
            raise RuntimeError("transient")
        return real_force(events)

    backend.force_complete_events = flaky_force
    ctx = _register(module, monkeypatch, clock, 1, stream, backend)
    module.unregister_kv_cache(1)

    clock.now += 31.0
    module._release_cycle()  # first attempt fails; retry in 2s
    clock.now += 1.0
    module._release_cycle()  # backing off: no attempt
    assert failures["left"] == 1
    clock.now += 1.0
    module._release_cycle()  # second attempt fails; retry in 4s
    assert failures["left"] == 0
    clock.now += 4.0
    module._release_cycle()  # third attempt forces; the stream drains
    module._release_cycle()

    ctx.close.assert_called_once()
    status = module.release_status()
    assert status["force_errors"] == 2
    assert status["forced"] == 1
    assert status["pending_instance_ids"] == []


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
    monkeypatch.setattr(ts, "_nogil_driver", lambda: None)  # cuda-python path
    monkeypatch.setattr(ts, "_raw_stream_handle", lambda stream, device: 7)
    monkeypatch.setattr(ts.torch.cuda, "device", lambda index: nullcontext())
    monkeypatch.setattr(ts, "_LOCAL_BUFFERS_BY_HANDLE", {b"own": 0x9000})
    backend = ts.TimelineSemaphoreEventIPCBackend()
    monkeypatch.setattr(
        backend, "_get_buffer", lambda index: SimpleNamespace(buffer_ops_stream="s")
    )
    # The slot is never read back (a D2H read can block next to a stuck
    # stream): every slot with an awaited sequence is written.
    monkeypatch.setattr(
        backend,
        "_read_slot",
        lambda e: pytest.fail("force must not read the slot back"),
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
        _imported(0x1000, 8, 5),  # another slot
        _imported(0x1000, 16, 0),  # never recorded
        _imported(0x9000, 0, 2),  # resolves to this process's own buffer
        local,
        "not-an-event",
    ]

    written = backend.force_complete_events(events)

    assert writes == [(0x1000, 6), (0x1008, 5)]
    assert written == 2


class _FakeNoGil:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def wait_value_geq(self, stream: int, address: int, value: int) -> None:
        self.calls.append(("wait", stream, address, value))

    def write_value(self, stream: int, address: int, value: int) -> None:
        self.calls.append(("write", stream, address, value))

    def synchronize(self, stream: int) -> None:
        self.calls.append(("sync", stream))


def test_stream_memory_ops_go_through_the_gil_free_driver(monkeypatch) -> None:
    """wait/record/force use the ctypes driver (drops the GIL) when present,
    never cuda-python, which keeps the GIL while a full stream blocks it."""
    fake = _FakeNoGil()
    monkeypatch.setattr(ts, "_nogil_driver", lambda: fake)
    monkeypatch.setattr(
        ts, "_cuda", SimpleNamespace(driver=None, runtime=None)
    )  # any cuda-python use would raise
    monkeypatch.setattr(ts, "_raw_stream_handle", lambda stream, device: 7)
    monkeypatch.setattr(ts.torch.cuda, "device", lambda index: nullcontext())
    monkeypatch.setattr(ts, "_LOCAL_BUFFERS_BY_HANDLE", {})
    backend = ts.TimelineSemaphoreEventIPCBackend()
    buffer = SimpleNamespace(
        buffer_ops_stream="s",
        lock=threading.Lock(),
        base_ptr=0x2000,
        handle_bytes=b"h",
        assign_locked=lambda stream: (8, 3),
    )
    monkeypatch.setattr(backend, "_get_buffer", lambda index: buffer)

    ts._enqueue_wait(_imported(0x1000, 0, 4), "stream")
    local = ts.TimelineSemaphoreEvent(
        origin=ts.TimelineSemaphoreEventOrigin.LOCAL, device_index=0
    )
    backend.record_event(local, "stream")
    backend.force_complete_events([_imported(0x1000, 16, 9)])

    assert fake.calls == [
        ("wait", 7, 0x1000, 4),
        ("write", 7, 0x2008, 3),
        ("write", 7, 0x1010, 9),
        ("sync", 7),
    ]


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

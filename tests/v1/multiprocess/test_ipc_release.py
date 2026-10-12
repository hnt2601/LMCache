# SPDX-License-Identifier: Apache-2.0
"""Non-blocking IPC release of unregistered / reaped instances (LMCache#4676).

A released instance whose transfer stream still has queued work must not be
unmapped (that hangs the driver) and must not block its caller (that wedges
the reaper or the MQ loop). These tests drive the staged release through the
module's public surface with a fake clock, a stubbed stream and a stubbed
release thread, so each stage is stepped explicitly:

    drained -> unmap now
    busy    -> deferred; after the drain grace, force-complete held events
               (instance poisoned: its store commits are dropped)
    still busy after the stuck timeout -> quarantine, or exit the process
"""

# Standard
from types import SimpleNamespace
from unittest.mock import MagicMock
import time

# Third Party
import pytest

# First Party
from lmcache.v1.multiprocess.modules.lmcache_driven_transfer import (
    STUCK_RELEASE_EXIT_CODE,
    IPCReleaseConfig,
    LMCacheDrivenTransferModule,
)
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

    def check_event_support(self, device: object) -> None:
        pass

    def __init__(self, stream: _Stream, frees_stream: bool = True) -> None:
        self.forced: list[list[object]] = []
        self._stream = stream
        self._frees_stream = frees_stream

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
    )
    return LMCacheDrivenTransferModule(MagicMock(name="ctx"), config)


def _register(module, monkeypatch, instance_id: int, stream: _Stream, backend):
    cache_context = MagicMock(name=f"cache_context-{instance_id}")
    cache_context.num_layers = 1
    cache_context.stream = stream
    cache_context.held_imported_events.return_value = [f"event-{instance_id}"]
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


def test_drained_stream_is_unmapped_inline(monkeypatch, clock, dev) -> None:
    module = _module(monkeypatch)
    stream = _Stream(drained=True)
    ctx = _register(module, monkeypatch, 1, stream, _ForcingBackend(stream))

    module.unregister_kv_cache(1)

    ctx.close.assert_called_once()
    dev.empty_cache.assert_called_once()
    assert module.release_status()["pending_instance_ids"] == []
    gpu_mod.create_periodic_thread.assert_not_called()


def test_busy_stream_is_deferred_not_closed(monkeypatch, clock, dev) -> None:
    """The caller returns at once; the mapping stays open until drained."""
    module = _module(monkeypatch)
    stream = _Stream(drained=False)
    ctx = _register(module, monkeypatch, 1, stream, _ForcingBackend(stream))

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
    ctx = _register(module, monkeypatch, 1, stream, _ForcingBackend(stream))
    clock.now += 500.0

    reaped = module.reap_stale_instances(reap_timeout_s=60.0, registration_grace_s=60.0)

    assert reaped == [1]
    ctx.close.assert_not_called()
    assert module.tracked_instance_count() == 0
    assert module.release_status()["pending_instance_ids"] == [1]


def test_force_after_grace_releases_and_poisons(monkeypatch, clock, dev) -> None:
    module = _module(monkeypatch, grace=30.0)
    stream = _Stream(drained=False)
    backend = _ForcingBackend(stream)
    ctx = _register(module, monkeypatch, 1, stream, backend)
    module.unregister_kv_cache(1)

    clock.now += 29.0
    module._release_cycle()
    assert backend.forced == []  # still inside the grace

    clock.now += 2.0
    module._release_cycle()  # forces; the stream drains on the next check
    assert backend.forced == [["event-1"]]
    ctx.close.assert_not_called()

    # A commit for the forced instance is dropped, not written to L1.
    storage = module.context.storage_manager
    module._finish_write_unless_poisoned((1, [(0, ["k"])]))
    storage.finish_write_by_owner.assert_not_called()
    # Other instances still commit.
    module._finish_write_unless_poisoned((2, [(0, ["k2"])]))
    storage.finish_write_by_owner.assert_called_once_with([(0, ["k2"])])

    module._release_cycle()
    ctx.close.assert_called_once()
    status = module.release_status()
    assert status["forced"] == 1
    assert status["dropped_commits"] == 1
    assert status["pending_instance_ids"] == []


def test_poison_expires_after_release(monkeypatch, clock, dev) -> None:
    module = _module(monkeypatch, grace=1.0)
    stream = _Stream(drained=False)
    _register(module, monkeypatch, 1, stream, _ForcingBackend(stream))
    module.unregister_kv_cache(1)
    clock.now += 2.0
    module._release_cycle()
    module._release_cycle()

    clock.now += gpu_mod._POISON_RETENTION_SECONDS + 1.0
    module._release_cycle()

    module._finish_write_unless_poisoned((1, [(0, ["k"])]))
    module.context.storage_manager.finish_write_by_owner.assert_called_once()


def test_unforceable_backend_is_quarantined(monkeypatch, clock, dev) -> None:
    """Legacy CUDA IPC events cannot be forced: the entry stays mapped."""
    module = _module(monkeypatch, grace=30.0, stuck=60.0)
    stream = _Stream(drained=False)
    ctx = _register(module, monkeypatch, 1, stream, _LegacyBackend())
    module.unregister_kv_cache(1)

    clock.now += 31.0
    module._release_cycle()
    clock.now += 61.0
    summary = module._release_cycle()

    assert summary.success is False
    ctx.close.assert_not_called()
    dev.empty_cache.assert_not_called()
    status = module.release_status()
    assert status["unforceable"] == 1
    assert status["stuck_instance_ids"] == [1]


def test_stuck_release_exits_when_configured(monkeypatch, clock, dev) -> None:
    module = _module(monkeypatch, grace=1.0, stuck=1.0, action="exit")
    stream = _Stream(drained=False)
    backend = _ForcingBackend(stream, frees_stream=False)
    _register(module, monkeypatch, 1, stream, backend)
    module.unregister_kv_cache(1)
    exits: list[int] = []

    def fake_exit(code: int) -> None:
        exits.append(code)
        raise SystemExit(code)

    monkeypatch.setattr(gpu_mod.os, "_exit", fake_exit)
    monkeypatch.setattr(gpu_mod.logging, "shutdown", lambda: None)

    clock.now += 2.0
    module._release_cycle()  # forced, still busy
    clock.now += 2.0
    with pytest.raises(SystemExit):
        module._release_cycle()
    assert exits == [STUCK_RELEASE_EXIT_CODE]


def test_close_forces_without_grace_and_never_exits(monkeypatch, clock, dev) -> None:
    module = _module(monkeypatch, grace=30.0, stuck=5.0, action="exit")
    monkeypatch.setattr(gpu_mod.os, "_exit", MagicMock(side_effect=AssertionError))
    stream = _Stream(drained=False)
    backend = _ForcingBackend(stream)
    ctx = _register(module, monkeypatch, 1, stream, backend)

    module.close()

    assert backend.forced == [["event-1"]]
    ctx.close.assert_called_once()


def test_close_gives_up_on_stuck_stream(monkeypatch, clock, dev) -> None:
    module = _module(monkeypatch, stuck=5.0)
    stream = _Stream(drained=False)
    ctx = _register(
        module, monkeypatch, 1, stream, _ForcingBackend(stream, frees_stream=False)
    )
    start = clock.now

    module.close()

    ctx.close.assert_not_called()
    assert clock.now - start <= 5.0 + 1.0

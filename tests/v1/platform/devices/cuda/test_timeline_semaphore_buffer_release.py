# SPDX-License-Identifier: Apache-2.0
"""Closing imported semaphore buffers with the registration that used them.

A worker's semaphore buffer is opened once per (handle, device) on first
import and was kept for the process lifetime: 2 MiB of driver mapping per
dead worker, never returned (measured as +2 MiB of server VRAM per worker
crash). These tests drive the registry with a fake CUDA runtime, so they
need torch importable but no device:

    import with user A and B   -> one open
    release A                  -> still mapped (B uses it)
    release B                  -> closed, registry empty
    import after close         -> reopened
    import without a user      -> pinned, never closed
    close fails                -> stays mapped and pinned, no raise
"""

# Standard
from contextlib import nullcontext
from types import SimpleNamespace
import threading

# Third Party
import pytest

# First Party
from lmcache.v1.platform.devices.cuda import timeline_semaphore_event_ipc as ts


class _Runtime:
    """cudaIpcOpenMemHandle / cudaIpcCloseMemHandle stand-ins."""

    def __init__(self) -> None:
        self.opened: list[bytes] = []
        self.closed: list[int] = []
        self.fail_close = False
        self._next_ptr = 0x1000
        self.cudaIpcMemLazyEnablePeerAccess = 0

    def cudaIpcOpenMemHandle(self, handle: bytes, flags: int) -> tuple[int, int]:
        self.opened.append(handle)
        ptr = self._next_ptr
        self._next_ptr += 0x1000
        return (0, ptr)

    def cudaIpcCloseMemHandle(self, ptr: int) -> int:
        if self.fail_close:
            return 1
        self.closed.append(ptr)
        return 0


class _User:
    """An import owner, like a registration's cache context (takes attributes)."""


class _SlotsUser:
    """An owner that takes no attributes but can be weakly referenced."""

    __slots__ = ("__weakref__",)


@pytest.fixture
def runtime(monkeypatch) -> _Runtime:
    rt = _Runtime()

    def check(result, what: str) -> None:
        err = result[0] if isinstance(result, tuple) else result
        if err:
            raise RuntimeError(f"{what} failed: {err}")

    monkeypatch.setattr(ts, "_cuda", SimpleNamespace(runtime=rt))
    monkeypatch.setattr(ts, "_CHECK_CUDA", check)
    monkeypatch.setattr(ts, "_resolve_device_index", lambda device: 0)
    monkeypatch.setattr(ts, "cuda_ipc_handle_from_bytes", lambda b: b)
    monkeypatch.setattr(
        ts, "torch", SimpleNamespace(cuda=SimpleNamespace(device=lambda i: nullcontext()))
    )
    monkeypatch.setattr(ts, "_LOCAL_BUFFERS_BY_HANDLE", {})
    monkeypatch.setattr(ts, "_IMPORTED_BUFFERS_BY_HANDLE", {})
    monkeypatch.setattr(ts, "_IMPORTED_BUFFER_USERS", {})
    monkeypatch.setattr(ts, "_IMPORTING_BUFFERS", {})
    return rt


def _handle(name: bytes, seq: int = 1) -> bytes:
    return ts._EXPORT_STRUCT.pack(ts._HANDLE_VERSION, name.ljust(64, b"\0"), 8, seq)


def test_buffer_closes_with_its_last_user(runtime) -> None:
    backend = ts.TimelineSemaphoreEventIPCBackend()
    user_a, user_b = _User(), _User()

    backend.import_event(_handle(b"w1"), "cuda:0", user=user_a)
    backend.import_event(_handle(b"w1", seq=2), "cuda:0", user=user_b)
    assert len(runtime.opened) == 1
    assert backend.imported_buffer_count() == 1

    assert backend.release_imported_buffers(user_a) == 0
    assert backend.imported_buffer_count() == 1
    assert runtime.closed == []

    assert backend.release_imported_buffers(user_b) == 1
    assert backend.imported_buffer_count() == 0
    assert runtime.closed == [0x1000]
    assert ts._IMPORTED_BUFFER_USERS == {}
    # Releasing again is a no-op, not a double close.
    assert backend.release_imported_buffers(user_b) == 0


def test_import_after_close_reopens(runtime) -> None:
    backend = ts.TimelineSemaphoreEventIPCBackend()
    user = _User()
    first = backend.import_event(_handle(b"w1"), "cuda:0", user=user)
    backend.release_imported_buffers(user)

    again = backend.import_event(_handle(b"w1"), "cuda:0", user=_User())

    assert len(runtime.opened) == 2
    assert again.base_ptr != first.base_ptr
    assert backend.imported_buffer_count() == 1


def test_import_without_user_pins_the_buffer(runtime) -> None:
    backend = ts.TimelineSemaphoreEventIPCBackend()
    user = _User()
    backend.import_event(_handle(b"w1"), "cuda:0")
    backend.import_event(_handle(b"w1"), "cuda:0", user=user)

    assert backend.release_imported_buffers(user) == 0
    assert backend.imported_buffer_count() == 1
    assert runtime.closed == []


def test_slots_user_is_tracked_through_the_weak_registry(runtime) -> None:
    backend = ts.TimelineSemaphoreEventIPCBackend()
    user = _SlotsUser()
    backend.import_event(_handle(b"w1"), "cuda:0", user=user)

    assert backend.release_imported_buffers(user) == 1
    assert backend.imported_buffer_count() == 0


def test_untrackable_user_pins_the_buffer(runtime) -> None:
    backend = ts.TimelineSemaphoreEventIPCBackend()
    user = object()  # no attributes, no weakref: cannot be told apart later
    backend.import_event(_handle(b"w1"), "cuda:0", user=user)

    assert backend.release_imported_buffers(user) == 0
    assert backend.imported_buffer_count() == 1


def test_unknown_user_releases_nothing(runtime) -> None:
    backend = ts.TimelineSemaphoreEventIPCBackend()
    backend.import_event(_handle(b"w1"), "cuda:0", user=_User())

    assert backend.release_imported_buffers(_User()) == 0
    assert backend.imported_buffer_count() == 1


def test_failed_close_keeps_the_mapping_pinned(runtime) -> None:
    backend = ts.TimelineSemaphoreEventIPCBackend()
    user = _User()
    event = backend.import_event(_handle(b"w1"), "cuda:0", user=user)
    runtime.fail_close = True

    assert backend.release_imported_buffers(user) == 0

    # Still mapped, and a later import reuses it instead of reopening.
    assert backend.imported_buffer_count() == 1
    again = backend.import_event(_handle(b"w1"), "cuda:0", user=_User())
    assert again.base_ptr == event.base_ptr
    assert len(runtime.opened) == 1
    # Pinned: no user can close it any more.
    runtime.fail_close = False
    assert backend.release_imported_buffers(_User()) == 0
    assert backend.imported_buffer_count() == 1


def test_import_during_close_waits_then_reopens(runtime, monkeypatch) -> None:
    """A transfer importing the handle while its buffer is being closed must
    not get the pointer being closed: it waits for the close and reopens."""
    backend = ts.TimelineSemaphoreEventIPCBackend()
    user = _User()
    backend.import_event(_handle(b"w1"), "cuda:0", user=user)

    closing = threading.Event()
    proceed = threading.Event()
    real_close = runtime.cudaIpcCloseMemHandle

    def slow_close(ptr: int) -> int:
        closing.set()
        proceed.wait(timeout=10)
        return real_close(ptr)

    runtime.cudaIpcCloseMemHandle = slow_close
    releaser = threading.Thread(target=backend.release_imported_buffers, args=(user,))
    releaser.start()
    assert closing.wait(timeout=10)

    result: dict[str, object] = {}

    def importer() -> None:
        result["event"] = backend.import_event(_handle(b"w1"), "cuda:0", user=_User())

    importing = threading.Thread(target=importer)
    importing.start()
    importing.join(timeout=0.5)
    assert importing.is_alive(), "import must wait for the close in progress"
    proceed.set()
    releaser.join(timeout=10)
    importing.join(timeout=10)

    assert runtime.closed == [0x1000]
    assert len(runtime.opened) == 2
    assert result["event"].base_ptr == 0x2000
    assert backend.imported_buffer_count() == 1

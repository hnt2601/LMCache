# SPDX-License-Identifier: Apache-2.0

# Future
from __future__ import annotations

# Standard
from abc import abstractmethod
from collections import Counter
from collections.abc import Callable
from typing import TYPE_CHECKING
import threading
import time
import weakref

# Third Party
from opentelemetry import metrics

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.config import EvictionConfig
from lmcache.v1.distributed.eviction import L1EvictionPolicy, L2EvictionPolicy
from lmcache.v1.distributed.eviction_policy import CreateEvictionPolicy
from lmcache.v1.distributed.eviction_policy.idle_ttl import (
    IdleTTLTracker,
    L1IdleTTLListener,
    L2IdleTTLListener,
)
from lmcache.v1.distributed.internal_api import (
    EvictionAction,
    EvictionDestination,
    L1ManagerInterface,
)
from lmcache.v1.distributed.l2_adapters.base import L2AdapterInterface
from lmcache.v1.distributed.storage_controller import StorageControllerInterface
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import get_event_bus

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.distributed.quota_manager import QuotaManager

logger = init_logger(__name__)

# Idle-TTL observability (FCI patch 0027). Counters are no-ops until a meter
# provider is installed; the gauge reads every live tracker at scrape time.
_meter = metrics.get_meter("lmcache_mp.health")
_IDLE_TTL_EVICTED = _meter.create_counter(
    "lmcache_mp.idle_ttl_evicted_chunk",
    description="Chunks deleted because they were idle longer than the "
    "tier's idle TTL. Tagged by tier = l1 | l2.",
    unit="chunks",
)
_IDLE_TTL_SKIPPED = _meter.create_counter(
    "lmcache_mp.idle_ttl_skipped_chunk",
    description="Expired chunks kept this tick because an object was locked "
    "(L1) or the delete was refused. Tagged by tier.",
    unit="chunks",
)
_LIVE_TRACKERS: "weakref.WeakValueDictionary[str, IdleTTLTracker]" = (
    weakref.WeakValueDictionary()
)


def _observe_oldest_age(_options):  # pragma: no cover - exercised by OTel
    for name, tracker in list(_LIVE_TRACKERS.items()):
        tier = name.split(":", 1)[0]
        yield metrics.Observation(tracker.oldest_age_seconds(), {"tier": tier})


_meter.create_observable_gauge(
    "lmcache_mp.idle_ttl_oldest_chunk_age",
    callbacks=[_observe_oldest_age],
    description="Idle time of the least recently used tracked chunk.",
    unit="s",
)

# Scan this many expired candidates per allowed delete, so a few pinned or
# locked chunks at the head of the order cannot stall the TTL pass.
_IDLE_TTL_SCAN_FACTOR = 4


class EvictionController(StorageControllerInterface):
    """
    Abstract base class for eviction controllers.

    Provides the shared eviction loop structure: background thread and stop
    flag. Subclasses implement eviction_loop and execute_eviction_action
    for their specific tier (L1 or L2).
    """

    def __init__(self):
        self._stop_flag = threading.Event()
        self._thread = threading.Thread(
            target=self.eviction_loop,
            daemon=True,
        )

    def start(self):
        logger.info("Starting %s...", self.__class__.__name__)
        self._thread.start()

    def stop(self):
        self._stop_flag.set()
        self._thread.join()

    @abstractmethod
    def report_status(self) -> dict:
        """Return a status dict for this controller.

        The child class needs to override this function to report
        controller-specific health and configuration information.
        """
        pass

    @abstractmethod
    def eviction_loop(self):
        """Run the eviction loop.

        The child class needs to override this function to implement
        internal eviction controlling logic.
        """
        pass

    @abstractmethod
    def execute_eviction_action(self, action: EvictionAction):
        """Execute a single eviction action.

        The child class needs to override this function to implement
        internal eviction controlling logic.
        """
        pass


class L1EvictionController(EvictionController):
    """
    Eviction controller for L1 cache.

    Uses an L1EvictionPolicy bridge to keep the eviction policy up-to-date
    with L1 manager events, and periodically triggers eviction based on
    L1 memory usage.
    """

    def __init__(
        self,
        l1_manager: L1ManagerInterface,
        eviction_config: EvictionConfig,
        clock: Callable[[], float] = time.monotonic,
    ):
        super().__init__()
        self._eviction_config = eviction_config
        self._eviction_policy = CreateEvictionPolicy(eviction_config)
        self._l1_manager = l1_manager
        self._listener = L1EvictionPolicy(self._eviction_policy)
        self._l1_manager.register_listener(self._listener)
        self._event_bus = get_event_bus()
        self._last_extra_log = time.monotonic()
        # Idle TTL (patch 0027): only wired up when enabled, so the default
        # config registers no extra listener and changes nothing.
        self._idle_tracker: IdleTTLTracker | None = None
        self._idle_evicted_chunks = 0
        self._idle_skipped_chunks = 0
        if eviction_config.idle_ttl_seconds > 0:
            self._idle_tracker = IdleTTLTracker(clock=clock)
            l1_manager.register_listener(L1IdleTTLListener(self._idle_tracker))
            _LIVE_TRACKERS[f"l1:{id(self)}"] = self._idle_tracker

    def report_status(self) -> dict:
        status = {
            "is_healthy": self._thread.is_alive(),
            "thread_alive": self._thread.is_alive(),
            "eviction_policy": self._eviction_config.eviction_policy,
            "trigger_watermark": self._eviction_config.trigger_watermark,
            "eviction_ratio": self._eviction_config.eviction_ratio,
        }
        if self._idle_tracker is not None:
            status["idle_ttl"] = {
                "idle_ttl_seconds": self._eviction_config.idle_ttl_seconds,
                "idle_ttl_batch": self._eviction_config.idle_ttl_batch,
                "tracked_chunks": len(self._idle_tracker),
                "oldest_chunk_age_seconds": self._idle_tracker.oldest_age_seconds(),
                "evicted_chunks": self._idle_evicted_chunks,
                "skipped_chunks": self._idle_skipped_chunks,
            }
        return status

    def run_idle_ttl_pass(self, now: float | None = None) -> int:
        """Delete up to ``idle_ttl_batch`` whole L1 chunks idle past the TTL.

        A chunk with any locked object (being read, written or stored to L2)
        is kept this tick. Runs on the eviction thread every second; exposed
        for tests.

        Returns:
            Number of chunks deleted.
        """
        tracker = self._idle_tracker
        if tracker is None:
            return 0
        cfg = self._eviction_config
        candidates = tracker.expired_chunks(
            cfg.idle_ttl_seconds,
            cfg.idle_ttl_batch * _IDLE_TTL_SCAN_FACTOR,
            now=now,
        )
        to_delete = []
        chunks = 0
        skipped = 0
        for _cid, members in candidates:
            if chunks >= cfg.idle_ttl_batch:
                break
            if members and all(self._l1_manager.is_key_evictable(k) for k in members):
                to_delete.extend(members)
                chunks += 1
            else:
                skipped += 1
        if to_delete:
            self._l1_manager.delete(to_delete)
        self._idle_evicted_chunks += chunks
        self._idle_skipped_chunks += skipped
        if chunks:
            _IDLE_TTL_EVICTED.add(chunks, {"tier": "l1"})
            logger.info(
                "L1 idle TTL %.0fs: deleted %d chunk(s), kept %d locked",
                cfg.idle_ttl_seconds,
                chunks,
                skipped,
            )
        if skipped:
            _IDLE_TTL_SKIPPED.add(skipped, {"tier": "l1"})
        return chunks

    def _publish_skipped(self, usage: float, watermark: float) -> None:
        """Publish a below-watermark loop tick (no eviction this cycle)."""
        self._event_bus.publish(
            Event(
                event_type=EventType.L1_EVICTION_LOOP_TICK,
                metadata={
                    "usage": usage,
                    "watermark": watermark,
                    "triggered": False,
                },
            )
        )

    def _publish_triggered(self, usage: float, watermark: float) -> None:
        """Publish an above-watermark loop tick (eviction policy ran)."""
        self._event_bus.publish(
            Event(
                event_type=EventType.L1_EVICTION_LOOP_TICK,
                metadata={
                    "usage": usage,
                    "watermark": watermark,
                    "triggered": True,
                },
            )
        )

    def _maybe_log_memory_usage(self, used_bytes: int, total_bytes: int) -> None:
        """Emit the opt-in L1 memory usage INFO line, throttled to the interval."""
        now = time.monotonic()
        if now - self._last_extra_log < self._eviction_config.extra_logging_interval:
            return
        self._last_extra_log = now
        pct = 0.0 if total_bytes == 0 else used_bytes / total_bytes * 100.0
        logger.info(
            "L1 memory usage: %.2f/%.2f GiB (%.1f%%)",
            used_bytes / (1 << 30),
            total_bytes / (1 << 30),
            pct,
        )

    def eviction_loop(self):
        watermark = self._eviction_config.trigger_watermark
        eviction_ratio = self._eviction_config.eviction_ratio

        while not self._stop_flag.is_set():
            time.sleep(1)
            if self._idle_tracker is not None:
                try:
                    self.run_idle_ttl_pass()
                except Exception:
                    logger.exception("L1 idle TTL pass failed")
            used_bytes, total_bytes = self._l1_manager.get_memory_usage()
            if self._eviction_config.extra_logging_enabled:
                self._maybe_log_memory_usage(used_bytes, total_bytes)
            usage = 0 if total_bytes == 0 else used_bytes / total_bytes
            if usage < watermark:
                logger.debug(
                    "L1 memory usage %.2f below watermark %.2f; skipping eviction.",
                    usage,
                    watermark,
                )
                self._publish_skipped(usage, watermark)
                continue

            logger.info(
                "L1 memory usage %.2f above watermark %.2f; triggering eviction.",
                usage,
                watermark,
            )
            actions = self._eviction_policy.get_eviction_actions(
                eviction_ratio,
                key_eligible_filter=self._l1_manager.is_key_evictable,
            )
            for action in actions:
                self.execute_eviction_action(action)
            self._publish_triggered(usage, watermark)

    def execute_eviction_action(self, action: EvictionAction):
        if action.destination == EvictionDestination.DISCARD:
            self._l1_manager.delete(action.keys)
        else:
            logger.error("Unsupported eviction destination: %s", action.destination)
            logger.error("Treating it as DISCARD.")
            self._l1_manager.delete(action.keys)


class L2AdapterEvictionState:
    """Per-adapter eviction state: its own policy, listener, and config."""

    def __init__(
        self,
        adapter_id: int,
        adapter: L2AdapterInterface,
        eviction_config: EvictionConfig,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.adapter_id = adapter_id
        self.adapter = adapter
        self.eviction_config = eviction_config
        self.eviction_policy = CreateEvictionPolicy(eviction_config)
        self.listener = L2EvictionPolicy(self.eviction_policy)
        adapter.register_listener(self.listener)
        # Idle TTL (patch 0027), only when enabled.
        self.idle_tracker: IdleTTLTracker | None = None
        self.idle_evicted_chunks = 0
        self.idle_skipped_chunks = 0
        if eviction_config.idle_ttl_seconds > 0:
            self.idle_tracker = IdleTTLTracker(clock=clock)
            adapter.register_listener(L2IdleTTLListener(self.idle_tracker))
            _LIVE_TRACKERS[f"l2:{adapter_id}:{id(self)}"] = self.idle_tracker


class L2EvictionController(StorageControllerInterface):
    """
    Unified eviction controller for all L2 adapters.

    Each adapter gets its own eviction policy and listener bridge, but a
    single background thread loops over all of them.

    When the adapter's policy sets ``support_isolation == True``
    (e.g. :class:`IsolatedLRUEvictionPolicy`), the controller consults
    the injected :class:`QuotaManager` to decide which ``cache_salt``
    buckets are over budget and evicts from each one in isolation.
    Otherwise it uses the adapter's aggregate ``usage_fraction``
    against the configured watermark — unchanged from the pre-PR5
    behavior.
    """

    def __init__(
        self,
        l2_adapter_states: list[L2AdapterEvictionState],
        quota_manager: QuotaManager | None = None,
    ):
        self._adapter_states = l2_adapter_states
        self._quota_manager = quota_manager
        # Guards _adapter_states against concurrent runtime add/remove.
        self._states_lock = threading.Lock()
        self._stop_flag = threading.Event()
        self._thread = threading.Thread(
            target=self._eviction_loop,
            daemon=True,
        )

    def start(self):
        logger.info("Starting %s...", self.__class__.__name__)
        self._thread.start()

    def stop(self):
        self._stop_flag.set()
        self._thread.join()

    def add_adapter_state(self, state: L2AdapterEvictionState) -> None:
        """Register a new adapter's eviction state at runtime."""
        with self._states_lock:
            self._adapter_states.append(state)

    def remove_adapter_state(self, adapter_id: int) -> None:
        """Drop the eviction state for ``adapter_id``.

        Blocks until any in-progress eviction pass finishes (it holds the
        same lock), so the adapter is guaranteed idle here before the
        caller closes it. A no-op if the adapter has no eviction state.
        """
        with self._states_lock:
            self._adapter_states = [
                s for s in self._adapter_states if s.adapter_id != adapter_id
            ]

    def report_status(self) -> dict:
        # NOTE: ``usage.bytes_by_cache_salt`` is intentionally NOT
        # surfaced here. A deployment can have 10k+ salts, so embedding
        # the full bucket map in the status response would blow up the
        # payload. Per-salt inspection goes through the dedicated HTTP
        # quota endpoints (which pull from ``QuotaManager`` +
        # ``StorageManager.get_usage_bytes_by_cache_salt``).
        adapter_statuses = []
        with self._states_lock:
            states = list(self._adapter_states)
        for state in states:
            usage = state.adapter.get_usage()
            adapter_statuses.append(
                {
                    "eviction_policy": state.eviction_config.eviction_policy,
                    "trigger_watermark": state.eviction_config.trigger_watermark,
                    "eviction_ratio": state.eviction_config.eviction_ratio,
                    "current_usage": usage.usage_fraction,
                    "total_bytes_used": usage.total_bytes_used,
                    "total_capacity_bytes": usage.total_capacity_bytes,
                    "num_cache_salt_buckets": len(usage.bytes_by_cache_salt),
                }
            )
            if state.idle_tracker is not None:
                adapter_statuses[-1]["idle_ttl"] = {
                    "idle_ttl_seconds": state.eviction_config.idle_ttl_seconds,
                    "idle_ttl_batch": state.eviction_config.idle_ttl_batch,
                    "tracked_chunks": len(state.idle_tracker),
                    "oldest_chunk_age_seconds": (
                        state.idle_tracker.oldest_age_seconds()
                    ),
                    "evicted_chunks": state.idle_evicted_chunks,
                    "skipped_chunks": state.idle_skipped_chunks,
                }
            if state.eviction_config.touch_on_l1_hit:
                adapter_statuses[-1]["touch_on_l1_hit"] = True
        return {
            "is_healthy": self._thread.is_alive(),
            "thread_alive": self._thread.is_alive(),
            "adapters": adapter_statuses,
        }

    def _eviction_loop(self):
        while not self._stop_flag.is_set():
            time.sleep(1)
            # Hold the lock across the whole pass so remove_adapter_state
            # cannot detach (and the caller close) an adapter while we are
            # calling into it.
            with self._states_lock:
                for state in self._adapter_states:
                    if state.idle_tracker is not None:
                        try:
                            self.run_idle_ttl_pass(state)
                        except Exception:
                            logger.exception(
                                "L2 idle TTL pass failed for adapter %d",
                                state.adapter_id,
                            )
                    self._check_and_evict(state)

    def on_l1_keys_touched(self, keys: list[ObjectKey]) -> None:
        """Cross-tier touch: an L1 hit refreshes the same keys in L2.

        In memory only, no I/O. Refreshes the L2 idle-TTL clock of adapters
        with idle TTL on, so a chunk that keeps hitting L1 does not expire
        from L2 underneath it. With ``touch_on_l1_hit`` it also refreshes the
        adapter's eviction-policy order. Keys the adapter does not hold are
        ignored by both.
        """
        if not keys:
            return
        with self._states_lock:
            states = list(self._adapter_states)
        for state in states:
            if state.idle_tracker is not None:
                state.idle_tracker.on_keys_touched(keys)
            if state.eviction_config.touch_on_l1_hit:
                state.eviction_policy.on_keys_touched(keys)

    def run_idle_ttl_pass(
        self, state: L2AdapterEvictionState, now: float | None = None
    ) -> int:
        """Delete up to ``idle_ttl_batch`` whole L2 chunks idle past the TTL.

        Goes through ``adapter.delete``, so adapters that pin-check or
        tombstone (nixl_store_dynamic, patch 0026) keep their guarantees.
        Exposed for tests.

        Returns:
            Number of chunks handed to the adapter for deletion.
        """
        tracker = state.idle_tracker
        if tracker is None:
            return 0
        cfg = state.eviction_config
        candidates = tracker.expired_chunks(
            cfg.idle_ttl_seconds,
            cfg.idle_ttl_batch * _IDLE_TTL_SCAN_FACTOR,
            now=now,
        )
        picked = candidates[: cfg.idle_ttl_batch]
        keys = [k for _cid, members in picked for k in members]
        if not keys:
            return 0
        self._execute_eviction_action(
            state.adapter,
            EvictionAction(keys=keys, destination=EvictionDestination.DISCARD),
        )
        # A chunk the adapter refused (pinned) is still tracked afterwards.
        deleted = sum(1 for cid, _m in picked if tracker.last_access(cid) is None)
        skipped = len(picked) - deleted
        state.idle_evicted_chunks += deleted
        state.idle_skipped_chunks += skipped
        if deleted:
            _IDLE_TTL_EVICTED.add(deleted, {"tier": "l2"})
            logger.info(
                "L2 idle TTL %.0fs (adapter %d): deleted %d chunk(s), kept %d",
                cfg.idle_ttl_seconds,
                state.adapter_id,
                deleted,
                skipped,
            )
        if skipped:
            _IDLE_TTL_SKIPPED.add(skipped, {"tier": "l2"})
        return deleted

    def _check_and_evict(self, state: L2AdapterEvictionState):
        if state.eviction_policy.support_isolation and self._quota_manager is not None:
            self._check_and_evict_by_cache_salt(state)
        else:
            self._check_and_evict_global(state)

    def _check_and_evict_global(self, state: L2AdapterEvictionState):
        """Aggregate-usage eviction (``LRU`` / ``noop``)."""
        watermark = state.eviction_config.trigger_watermark
        eviction_ratio = state.eviction_config.eviction_ratio

        # ``usage_fraction == -1`` means the adapter doesn't support
        # usage-based eviction (no max_capacity_bytes declared), so we
        # do not trigger eviction. Adapters with ``supports_global_eviction ==
        # False`` should already have been filtered out at construction
        # time in ``StorageManager``; this check is a defensive belt.
        current_usage = state.adapter.get_usage().usage_fraction
        if current_usage < 0 or current_usage < watermark:
            logger.debug(
                "L2 usage %.2f below watermark %.2f; skipping eviction.",
                current_usage,
                watermark,
            )
            return

        logger.info(
            "L2 usage %.2f above watermark %.2f; triggering eviction.",
            current_usage,
            watermark,
        )
        actions = state.eviction_policy.get_eviction_actions(eviction_ratio)
        for action in actions:
            self._execute_eviction_action(state.adapter, action)

    def _check_and_evict_by_cache_salt(self, state: L2AdapterEvictionState):
        """Per-``cache_salt`` eviction driven by :class:`QuotaManager`.

        For every salt with non-zero bytes, compare its usage against
        ``watermark * quota``. Salts over threshold get eviction scoped
        to their own LRU list. Salts with no quota registered have an
        effective limit of ``0`` and are therefore always over budget,
        so they get a full eviction (``effective_ratio=1.0``) — this
        enforces the allowlist rule: only registered salts retain data.

        Per-destination keys are batched across all over-budget salts
        before invoking the adapter — one ``adapter.delete(...)`` call
        per destination instead of one per (salt, destination) pair.
        Adapters with non-trivial per-call overhead (NIXL handle setup,
        FS sync, etc.) see this as a real win when many salts go over
        budget in the same cycle.
        """
        assert self._quota_manager is not None
        watermark = state.eviction_config.trigger_watermark
        eviction_ratio = state.eviction_config.eviction_ratio
        usage = state.adapter.get_usage()

        # destination -> accumulated keys across all over-budget salts.
        pending: dict[EvictionDestination, list[ObjectKey]] = {}

        for cache_salt, user_bytes in usage.bytes_by_cache_salt.items():
            if user_bytes <= 0:
                continue
            limit = self._quota_manager.get_limit_bytes(cache_salt)
            # Trigger on ``>=`` to match the global branch's ``usage <
            # watermark`` short-circuit. Salts with no quota (limit=0)
            # always land here because ``user_bytes > 0 >= 0``.
            if user_bytes < watermark * limit:
                continue

            # Unregistered / zero-quota salts: wipe everything.
            # Registered salts: evict the configured ratio of their list.
            effective_ratio = 1.0 if limit == 0 else eviction_ratio
            logger.info(
                "cache_salt=%r over quota (bytes=%d, limit=%d, "
                "watermark=%.2f); evicting ratio=%.2f.",
                cache_salt,
                user_bytes,
                limit,
                watermark,
                effective_ratio,
            )
            actions = state.eviction_policy.get_eviction_actions(
                effective_ratio, cache_salt=cache_salt
            )
            for action in actions:
                pending.setdefault(action.destination, []).extend(action.keys)

        for destination, keys in pending.items():
            self._execute_eviction_action(
                state.adapter,
                EvictionAction(keys=keys, destination=destination),
            )

    def _execute_eviction_action(
        self, adapter: L2AdapterInterface, action: EvictionAction
    ):
        if action.destination == EvictionDestination.DISCARD:
            adapter.delete(action.keys)
        else:
            logger.error("Unsupported eviction destination: %s", action.destination)
            logger.error("Treating it as DISCARD.")
            adapter.delete(action.keys)

        if action.keys:
            get_event_bus().publish(
                Event(
                    event_type=EventType.L2_KEYS_EVICTED,
                    metadata={
                        "key_count": len(action.keys),
                        "key_count_per_salt": Counter(
                            k.cache_salt for k in action.keys
                        ),
                    },
                )
            )

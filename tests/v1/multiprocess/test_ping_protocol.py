# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the PING wire protocol (#4687 boot-token restart
detection).

Covers the two things the integration relies on: (1) the declared wire
type is ``int``, and (2) a legacy ``bool``-protocol peer and a new
``int``-protocol peer decode each other's PING response without a crash,
because ``msgspec_decode``'s bool<->int coercion (mq.py) degrades a boot
token to a plain truthy/falsy signal rather than raising -- TC-208.
No GPU is required.
"""

# Standard
from unittest.mock import MagicMock

# Third Party
import zmq

# First Party
from lmcache.v1.multiprocess.modules.management import ManagementModule
from lmcache.v1.multiprocess.mq import (
    MessageQueueClient,
    MessageQueueServer,
    msgspec_decode,
    msgspec_encode,
)
from lmcache.v1.multiprocess.protocol import RequestType, get_response_class
from lmcache.v1.multiprocess.server import add_handler_helper


def test_ping_response_class_is_int() -> None:
    """The PING protocol declares an int response, not bool -- the
    wire-visible half of the #4687 change."""
    assert get_response_class(RequestType.PING) is int


def test_management_ping_returns_a_stable_positive_boot_token() -> None:
    """ManagementModule mints one boot token per process and returns it on
    every ping() call; it must never collide with the reserved legacy
    value 1 (see the interop tests below)."""
    mgmt = ManagementModule(ctx=MagicMock())
    first = mgmt.ping(None)
    second = mgmt.ping(None)
    assert isinstance(first, int)
    assert first >= 2
    assert first == second


def test_two_management_modules_mint_different_boot_tokens() -> None:
    """Boot tokens are per-process random, not a fixed or sequential
    value -- otherwise two servers restarting in quick succession could
    coincidentally mint the same token and mask a restart."""
    token_a = ManagementModule(ctx=MagicMock()).ping(None)
    token_b = ManagementModule(ctx=MagicMock()).ping(None)
    assert token_a != token_b


def test_legacy_bool_true_decodes_as_reserved_token_one() -> None:
    """TC-208: an old server still speaking the bool protocol encodes its
    PING response as ``True``. A new client, whose protocol declares PING
    as ``int``, must decode that as the reserved token ``1`` instead of
    raising -- so restart detection is simply unavailable (token never
    changes) rather than crashing the client."""
    legacy_wire_bytes = msgspec_encode(True, bool)
    decoded = msgspec_decode(legacy_wire_bytes, int)
    assert decoded == 1


def test_new_int_token_decodes_as_truthy_for_a_legacy_client() -> None:
    """TC-208, reverse direction: a new server's positive boot token must
    decode as truthy for an old client whose protocol still declares PING
    as ``bool``, so the legacy client's `if response:` health check keeps
    working during a rolling upgrade."""
    new_wire_bytes = msgspec_encode(123456789, int)
    decoded = msgspec_decode(new_wire_bytes, bool)
    assert decoded is True


def test_ping_round_trip_over_real_server_returns_int_boot_token() -> None:
    """End-to-end: a real MessageQueueServer serving ManagementModule.ping
    via the actual PING protocol definition returns an int over the wire,
    not the legacy bool."""
    server_url = "tcp://127.0.0.1:16040"
    context = zmq.Context.instance()

    management = ManagementModule(ctx=MagicMock())
    server = MessageQueueServer(server_url, context)
    add_handler_helper(server, RequestType.PING, management.ping)
    server.add_normal_thread_pool([RequestType.PING], max_workers=2)
    server.start()

    try:
        client = MessageQueueClient(server_url, context)
        try:
            future = client.submit_request(RequestType.PING, [None])
            response = future.result(timeout=5)
            assert isinstance(response, int)
            assert response == management.ping(None)
        finally:
            client.close()
    finally:
        server.close()

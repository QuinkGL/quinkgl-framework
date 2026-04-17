"""
B7 regression tests — Deduplicate chunk rate-limiting.

Validates that:
 - CHUNK_SEND_INTERVAL is a positive constant exposed at module level.
 - send_model_update uses a single sleep per inter-chunk gap (no double sleep).
"""

import ast
import inspect
import textwrap

import pytest


# ---------------------------------------------------------------------------
# B7-1: CHUNK_SEND_INTERVAL constant exists and is positive
# ---------------------------------------------------------------------------

def test_chunk_send_interval_constant():
    from quinkgl.network.gossip_community import CHUNK_SEND_INTERVAL
    assert isinstance(CHUNK_SEND_INTERVAL, (int, float))
    assert CHUNK_SEND_INTERVAL > 0, "CHUNK_SEND_INTERVAL must be positive"


# ---------------------------------------------------------------------------
# B7-2: send_model_update body has exactly ONE asyncio.sleep per chunk loop
# ---------------------------------------------------------------------------

def test_single_sleep_in_send_loop():
    """Parse the source of send_model_update and count asyncio.sleep calls
    inside the for-loop that iterates over chunks.
    """
    from quinkgl.network.gossip_community import GossipLearningCommunity

    src = inspect.getsource(GossipLearningCommunity.send_model_update)
    # Dedent so ast.parse works even if indented
    src = textwrap.dedent(src)
    tree = ast.parse(src)

    # Walk AST: find For nodes, then count Call nodes with 'asyncio.sleep'
    sleep_count_in_for = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.For):
            for child in ast.walk(node):
                if isinstance(child, ast.Call):
                    func = child.func
                    # asyncio.sleep(...)
                    if (isinstance(func, ast.Attribute)
                            and func.attr == "sleep"
                            and isinstance(func.value, ast.Name)
                            and func.value.id == "asyncio"):
                        sleep_count_in_for += 1

    assert sleep_count_in_for == 1, (
        f"Expected exactly 1 asyncio.sleep in the chunk-send for-loop, "
        f"found {sleep_count_in_for}"
    )


# ---------------------------------------------------------------------------
# B7-3: CHUNK_SEND_INTERVAL used in send_model_update source
# ---------------------------------------------------------------------------

def test_chunk_send_interval_referenced_in_source():
    from quinkgl.network.gossip_community import GossipLearningCommunity

    src = inspect.getsource(GossipLearningCommunity.send_model_update)
    assert "CHUNK_SEND_INTERVAL" in src, (
        "send_model_update should reference CHUNK_SEND_INTERVAL, not a magic number"
    )

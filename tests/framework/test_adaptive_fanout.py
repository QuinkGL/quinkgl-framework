"""Adaptive gossip fanout policy tests."""

from quinkgl.gossip.aggregator import ModelAggregator


def test_adaptive_fanout_uses_default_for_small_swarms():
    assert ModelAggregator._select_fanout(0) == 0
    assert ModelAggregator._select_fanout(1) == 1
    assert ModelAggregator._select_fanout(2) == 2
    assert ModelAggregator._select_fanout(3) == 3
    assert ModelAggregator._select_fanout(100) == 3


def test_adaptive_fanout_increases_for_large_swarms():
    assert ModelAggregator._select_fanout(101) == 5
    assert ModelAggregator._select_fanout(250) == 5
    assert ModelAggregator._select_fanout(251) == 7
    assert ModelAggregator._select_fanout(500) == 7
    assert ModelAggregator._select_fanout(501) == 10
    assert ModelAggregator._select_fanout(1000) == 10

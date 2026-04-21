import pytest

from quinkgl.telemetry.stream import STREAM_CLOSE_CODE_QUEUE_FULL, TelemetryStreamHub


@pytest.mark.asyncio
async def test_stream_hub_broadcasts_messages_to_subscribers():
    hub = TelemetryStreamHub()
    queue = await hub.subscribe()

    await hub.publish({"type": "node_snapshot_updated", "payload": {"node_id": "node-a"}})
    message = await queue.get()

    assert message["type"] == "node_snapshot_updated"
    assert message["payload"]["node_id"] == "node-a"


@pytest.mark.asyncio
async def test_stream_hub_supports_multiple_subscribers_and_unsubscribe():
    hub = TelemetryStreamHub()
    queue_a = await hub.subscribe()
    queue_b = await hub.subscribe()

    await hub.publish({"type": "node_event_received", "payload": {"event_type": "model_sent"}})
    message_a = await queue_a.get()
    message_b = await queue_b.get()
    await hub.unsubscribe(queue_b)
    await hub.publish({"type": "session_stats_updated", "payload": {"active_node_count": 1}})
    final_message = await queue_a.get()

    assert message_a["type"] == "node_event_received"
    assert message_b["payload"]["event_type"] == "model_sent"
    assert final_message["type"] == "session_stats_updated"


@pytest.mark.asyncio
async def test_stream_hub_marks_overflowed_subscriber_for_close():
    hub = TelemetryStreamHub()
    queue = await hub.subscribe()

    for index in range(queue.maxsize):
        queue.put_nowait({"type": "buffered", "payload": {"i": index}})

    await hub.publish({"type": "node_snapshot_updated", "payload": {"node_id": "node-a"}})
    message = await queue.get()

    assert message["type"] == "stream_closed"
    assert message["code"] == STREAM_CLOSE_CODE_QUEUE_FULL

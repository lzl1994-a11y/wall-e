from unittest.mock import Mock
import pytest
from services.remote.remote_audio_publisher import RemoteAudioPublisher


def publisher():
    node, sink = Mock(), Mock()
    value = RemoteAudioPublisher(node, sink)
    return value, sink, node.create_guard_condition.return_value


def test_media_caller_does_not_run_blocking_ros_publication():
    value, sink, guard = publisher()
    frames = [object() for _ in range(6)]
    for frame in frames:
        value.publish(frame)
    sink.publish.assert_not_called()
    guard.trigger.assert_called_once()
    value._flush()
    assert [call.args[0] for call in sink.publish.call_args_list] == frames
    assert value.peak_pending_frames == 6


def test_next_frame_rearms_after_empty_flush():
    value, sink, guard = publisher()
    value.publish("first"); value._flush()
    value.publish("second"); value._flush()
    assert guard.trigger.call_count == 2
    assert [call.args[0] for call in sink.publish.call_args_list] == ["first", "second"]


def test_full_queue_fails_explicitly_without_dropping_received_frames():
    value, sink, guard = publisher()
    for index in range(value.MAX_PENDING_FRAMES):
        value.publish(index)
    with pytest.raises(ValueError, match="bounded queue"):
        value.publish("overflow")
    value._flush()
    assert [call.args[0] for call in sink.publish.call_args_list] == list(range(16))


def test_arrival_during_publish_keeps_fifo_and_wakeup():
    value, sink, guard = publisher()
    sink.publish.side_effect = lambda frame: value.publish("second") if frame == "first" else None
    value.publish("first"); value._flush()
    assert [call.args[0] for call in sink.publish.call_args_list] == ["first", "second"]
    value.publish("third")
    assert guard.trigger.call_count == 2


def test_continuous_producer_cannot_monopolize_the_ros_executor():
    value, sink, guard = publisher()
    sink.publish.side_effect = lambda frame: value.publish(frame + 1)
    value.publish(0); value._flush()
    assert sink.publish.call_count == value.MAX_PENDING_FRAMES
    assert list(value._pending) == [16]
    assert guard.trigger.call_count == 2


def test_ros_publication_failure_is_not_swallowed():
    value, sink, guard = publisher()
    sink.publish.side_effect = RuntimeError("DDS unavailable")
    value.publish("first")
    with pytest.raises(RuntimeError, match="DDS unavailable"):
        value._flush()

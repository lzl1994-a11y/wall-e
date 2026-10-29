"""Exercise the real planning graph and validator without model or hardware."""

import copy
import json
from unittest.mock import Mock

from services.llm.llm_ordered_planning import OrderedTaskPlanningWorkflow


PROMPT = "不要低头，然后挥手，再看看前面有什么"
VALID = {"steps": [
    {"name": "play_sequence", "arguments": {"sequence_name": "wave_hello"}, "grounding": "挥手"},
    {"name": "inspect_camera", "arguments": {"question": "看看前面有什么"}, "grounding": "看看前面有什么"},
]}
INVALID = {"steps": [
    {"name": "play_sequence", "arguments": {"sequence_name": "head_down"}, "grounding": "不要低头"},
    *copy.deepcopy(VALID["steps"]),
]}


def test_forbidden_head_step_is_repaired_using_the_real_validation_error():
    propose = Mock(side_effect=[copy.deepcopy(INVALID), copy.deepcopy(VALID)])
    result = OrderedTaskPlanningWorkflow(propose=propose).invoke(PROMPT)

    assert result == {"plan": VALID, "error": None, "attempts": 2}
    assert propose.call_count == 2
    assert propose.call_args_list[0].args == ("",)
    feedback = json.loads(propose.call_args_list[1].args[0])
    assert feedback["candidate"] == INVALID
    assert "negated_action" in feedback["validation_error"]
    assert INVALID["steps"][0]["grounding"] == "不要低头"


def test_two_invalid_candidates_end_without_returning_an_executable_plan():
    propose = Mock(return_value=copy.deepcopy(INVALID))
    result = OrderedTaskPlanningWorkflow(propose=propose).invoke(PROMPT)

    assert result["plan"] is None
    assert "negated_action" in result["error"]
    assert result["attempts"] == propose.call_count == 2


def test_cancelled_turn_never_calls_the_proposer():
    propose = Mock()
    result = OrderedTaskPlanningWorkflow(propose=propose, cancelled=lambda: True).invoke(PROMPT)

    assert result == {"plan": None, "error": "turn_cancelled", "attempts": 0}
    propose.assert_not_called()


def test_cancel_after_proposal_prevents_validation_repair():
    cancelled = False

    def propose(_feedback):
        nonlocal cancelled
        cancelled = True
        return copy.deepcopy(INVALID)

    callback = Mock(side_effect=propose)
    result = OrderedTaskPlanningWorkflow(propose=callback, cancelled=lambda: cancelled).invoke(PROMPT)

    assert result == {"plan": None, "error": "turn_cancelled", "attempts": 1}
    assert callback.call_count == 1


def test_transport_exception_is_not_retried():
    propose = Mock(side_effect=ConnectionError("unavailable"))
    result = OrderedTaskPlanningWorkflow(propose=propose).invoke(PROMPT)

    assert result == {"plan": None, "error": "ordered_planning_failed:ConnectionError", "attempts": 1}
    assert propose.call_count == 1


def test_invalid_supplied_candidate_gets_only_one_repair_call():
    propose = Mock(return_value=copy.deepcopy(VALID))
    result = OrderedTaskPlanningWorkflow(propose=propose).invoke(PROMPT, copy.deepcopy(INVALID))

    assert result == {"plan": VALID, "error": None, "attempts": 2}
    propose.assert_called_once()
    assert json.loads(propose.call_args.args[0])["candidate"] == INVALID


def test_valid_supplied_candidate_does_not_call_the_proposer():
    propose = Mock()
    result = OrderedTaskPlanningWorkflow(propose=propose).invoke(PROMPT, copy.deepcopy(VALID))

    assert result == {"plan": VALID, "error": None, "attempts": 1}
    propose.assert_not_called()


def test_failed_repair_of_supplied_candidate_stops_at_two_candidates():
    propose = Mock(return_value=copy.deepcopy(INVALID))
    result = OrderedTaskPlanningWorkflow(propose=propose).invoke(PROMPT, copy.deepcopy(INVALID))

    assert result["plan"] is None
    assert result["attempts"] == 2
    propose.assert_called_once()


def test_valid_json_string_candidate_is_parsed_before_validation():
    propose = Mock()
    result = OrderedTaskPlanningWorkflow(propose=propose).invoke(
        user_prompt=PROMPT, initial_candidate=json.dumps(VALID, ensure_ascii=False),
    )

    assert result == {"plan": VALID, "error": None, "attempts": 1}
    propose.assert_not_called()


def test_malformed_json_is_a_repairable_candidate_not_a_transport_exception():
    malformed = '{"steps":'
    propose = Mock(side_effect=[malformed, json.dumps(VALID, ensure_ascii=False)])
    result = OrderedTaskPlanningWorkflow(propose=propose).invoke(user_prompt=PROMPT)

    assert result == {"plan": VALID, "error": None, "attempts": 2}
    feedback = json.loads(propose.call_args_list[1].args[0])
    assert feedback["candidate"] == malformed
    assert feedback["validation_error"].startswith("ordered_plan_json_invalid:")

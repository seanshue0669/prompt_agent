# tests/system/test_orchestrator_system.py
"""System test for Orchestrator full flow across all stages."""

import json
import re
from pathlib import Path
from unittest.mock import Mock, patch

from agentcore import test_wrapper
from agents.orchestrator import Orchestrator
from config.runtime_config import RuntimeConfig

CONFIG_PATH = Path("config/json_config/system_test_config.json")


def load_system_test_config():
    """Load system test config from config/json_config/system_test_config.json."""
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def extract_prompt(user_prompt: str) -> str:
    """Extract prompt text from the first fenced block in user_prompt."""
    match = re.search(r"```\s*(.*?)\s*```", user_prompt, re.DOTALL)
    if not match:
        raise AssertionError("prompt block not found in user_prompt")
    return match.group(1).strip()


def extract_stage_tag(system_prompt: str) -> str:
    """Extract stage tag like stage_1 from system_prompt."""
    match = re.search(r"STAGE_(\d+)", system_prompt)
    if not match:
        return "stage_unknown"
    return f"stage_{match.group(1)}"


@test_wrapper
def test_orchestrator_full_flow_system():
    """Run full 6-stage flow and verify cross-stage propagation."""
    config = load_system_test_config()
    RuntimeConfig.config_data = config

    mock_cli = Mock()
    mock_cli.get_user_input.return_value = "system test answer"
    mock_cli.update_stage.return_value = None
    mock_cli.show_waiting_message.return_value = None
    mock_cli.clear_conversation.return_value = None
    RuntimeConfig.cli_interface = mock_cli

    diagnostic_prompts = []
    integration_prompts = []

    def mock_invoke(*args, **kwargs):
        user_prompt = kwargs.get("user_prompt", "")
        system_prompt = kwargs.get("system_prompt", "")

        if "SYSTEM_TEST_DIAGNOSTIC" in system_prompt:
            current_prompt = extract_prompt(user_prompt)
            diagnostic_prompts.append((system_prompt, current_prompt))
            if "STAGE_2" in system_prompt:
                return {"content": "{\"questions\": []}"}
            return {"content": "{\"questions\": [\"Test question?\"]}"}

        if "SYSTEM_TEST_FOLLOWUP" in system_prompt:
            return {"content": "{\"need_followup\": false}"}

        if "SYSTEM_TEST_COMPRESS" in system_prompt:
            return {"content": "{\"compressed\": \"Q: test? A: answer\"}"}

        if "SYSTEM_TEST_INTEGRATION" in system_prompt:
            current_prompt = extract_prompt(user_prompt)
            stage_tag = extract_stage_tag(system_prompt)
            integration_prompts.append((system_prompt, current_prompt))
            return {
                "content": json.dumps({
                    "current_prompt": f"{current_prompt} | integrated:{stage_tag}"
                })
            }

        raise AssertionError(f"Unexpected system_prompt: {system_prompt}")

    mock_client = Mock()
    mock_client.invoke.side_effect = mock_invoke

    with patch("agents.orchestrator.tool.load_config", return_value=config):
        with patch("agents.orchestrator.tool.validate_config"):
            orchestrator = Orchestrator(mock_client, "initial prompt")
            compiled = orchestrator.compile()
            result = compiled.invoke({}, config={"recursion_limit": 200})

    assert result["stage_idx"] == 7

    expected_diag_prompts = [
        "initial prompt",
        "initial prompt | integrated:stage_1",
        "initial prompt | integrated:stage_1",
        "initial prompt | integrated:stage_1 | integrated:stage_3",
        "initial prompt | integrated:stage_1 | integrated:stage_3 | integrated:stage_4",
        "initial prompt | integrated:stage_1 | integrated:stage_3 | integrated:stage_4 | integrated:stage_5",
    ]
    actual_diag_prompts = [prompt for _, prompt in diagnostic_prompts]
    assert actual_diag_prompts == expected_diag_prompts

    expected_final_prompt = (
        "initial prompt | integrated:stage_1 | integrated:stage_3 | "
        "integrated:stage_4 | integrated:stage_5 | integrated:stage_6"
    )
    assert result["current_prompt"] == expected_final_prompt

    assert mock_cli.get_user_input.call_count == 5

    integration_stage_tags = [extract_stage_tag(sp) for sp, _ in integration_prompts]
    assert integration_stage_tags == [
        "stage_1",
        "stage_3",
        "stage_4",
        "stage_5",
        "stage_6",
    ]


if __name__ == "__main__":
    print("Running Orchestrator system tests...\n")
    test_orchestrator_full_flow_system()
    print("\nAll tests completed!")

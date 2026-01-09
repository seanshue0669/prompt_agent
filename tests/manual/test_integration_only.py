# tests/manual/test_integration_only.py
"""
Manual integration-only test driven by a JSON config file.

Usage:
    python tests/manual/test_integration_only.py --config config/json_config/integration_test_config.json
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from agentcore import LLMClient
from agents.integration_agent import IntegrationAgent
from config.config_loader import load_config
from config.runtime_config import RuntimeConfig


def load_input_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_input_config(data: dict) -> None:
    required_fields = ["stage_idx", "current_prompt", "answer_list"]
    for field in required_fields:
        if field not in data:
            raise ValueError(f"Missing required field: {field}")

    if not isinstance(data["stage_idx"], int):
        raise ValueError("stage_idx must be an integer")
    if not isinstance(data["current_prompt"], str):
        raise ValueError("current_prompt must be a string")
    if not isinstance(data["answer_list"], list):
        raise ValueError("answer_list must be a list of strings")
    if not all(isinstance(item, str) for item in data["answer_list"]):
        raise ValueError("answer_list must be a list of strings")


def resolve_integration_prompt(stage_idx: int, config: dict) -> str:
    stage_names = config["stage_names"]
    if stage_idx < 1 or stage_idx > len(stage_names):
        raise ValueError(f"stage_idx {stage_idx} out of range (1-{len(stage_names)})")

    stage_name = stage_names[stage_idx - 1]
    return config["stage_prompts"][stage_name]["integration"]


def get_config_root(config_path: str) -> str:
    config_dir = os.path.dirname(os.path.abspath(config_path))
    if os.path.basename(config_dir) == "json_config":
        return os.path.abspath(os.path.join(config_dir, os.pardir))
    return config_dir


def resolve_current_prompt(prompt_value: str, config_root: str) -> str:
    if os.path.isabs(prompt_value):
        raise ValueError("current_prompt must be a relative path or inline prompt text")

    candidate = os.path.abspath(os.path.join(config_root, prompt_value))
    base_dir = os.path.abspath(config_root)
    if os.path.isfile(candidate):
        if os.path.commonpath([candidate, base_dir]) != base_dir:
            raise ValueError("current_prompt path must be within the config directory")
        with open(candidate, "r", encoding="utf-8") as handle:
            return handle.read()

    return prompt_value


def write_output(path: str, content: str) -> None:
    if not path:
        return
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(content)
    print(f"Saved output to: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run integration-only test")
    parser.add_argument(
        "--config",
        default="config/json_config/integration_test_config.json",
        help="Path to integration test config JSON",
    )
    args = parser.parse_args()

    input_config = load_input_config(args.config)
    validate_input_config(input_config)
    config_root = get_config_root(args.config)

    config = load_config("config/json_config/config.json")
    RuntimeConfig.config_data = config

    system_prompt = resolve_integration_prompt(input_config["stage_idx"], config)

    print("\n=== System Prompt ===")
    print(system_prompt)

    llm_config = {
        "model": "hf.co/unsloth/gemma-3n-E4B-it-GGUF:Q4_K_M",
        "temperature": 0.7,
        "max_completion_tokens": 1000,
    }
    llm_client = LLMClient(
        api_key="",
        base_url="http://localhost:11434/v1",
        default_config=llm_config,
    )

    agent = IntegrationAgent(llm_client)
    compiled = agent.compile()
    current_prompt = resolve_current_prompt(input_config["current_prompt"], config_root)

    state = {
        "system_prompt": system_prompt,
        "current_prompt": current_prompt,
        "answer_list": input_config["answer_list"],
    }

    result = compiled.invoke(state)
    current_prompt = result.get("current_prompt", "")

    print("\n=== Integration Result ===")
    print(current_prompt)

    write_output(input_config.get("output_path", ""), current_prompt)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(1)
    except Exception as exc:
        print(f"\nError: {exc}")
        raise

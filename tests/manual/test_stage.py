# tests/manual/test_stage.py
"""
Interactive script for testing individual stages and agents.

Usage:
    python tests/manual/test_stage.py --stage 1
    python tests/manual/test_stage.py --stage 3
"""

import sys
import os
import json
import argparse

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from agentcore import LLMClient
from agents.diagnostic_agent import DiagnosticAgent
from agents.questioning_agent import QuestioningAgent
from agents.integration_agent import IntegrationAgent
from agents.orchestrator.tool import OrchestratorTool
from config.runtime_config import RuntimeConfig
from config.config_loader import load_config
from cli.cli_interface import CLIInterface


def load_test_config():
    """Load test configuration."""
    config_path = "config/test_config.json"
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def print_state(state, title="State"):
    """Pretty print state dictionary."""
    print()
    print("=" * 80)
    print(f" {title}")
    print("=" * 80)
    for key, value in state.items():
        if isinstance(value, list):
            print(f"{key}:")
            for i, item in enumerate(value, 1):
                print(f"  {i}. {item}")
        elif isinstance(value, str) and len(value) > 100:
            print(f"{key}: {value[:100]}...")
        else:
            print(f"{key}: {value}")
    print("=" * 80)
    print()




def write_prompt_output(prompt: str, output_dir: str = "outputs", filename: str = "result.txt"):
    """Write the prompt to a text file for later review."""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(prompt)
    print(f"Saved prompt to: {output_path}")


def run_diagnostic(llm_client, stage_idx, test_prompt):
    """Run DiagnosticAgent and return state."""
    print("\n[自動執行] DiagnosticAgent - 生成問題...")
    
    # Use OrchestratorTool to get prompt
    orch_tool = OrchestratorTool()
    system_prompt = orch_tool.get_system_prompt(stage_idx, "diagnostic")
    
    agent = DiagnosticAgent(llm_client)
    compiled = agent.compile()
    
    state = {
        "system_prompt": system_prompt,
        "current_prompt": test_prompt,
        "question_list": []
    }
    
    print_state(state, "DiagnosticAgent 傳入 State")
    
    result = compiled.invoke(state)
    
    print_state(result, "DiagnosticAgent 傳出 State")
    
    return result


def run_questioning(llm_client, stage_idx, current_prompt, question_list):
    """Run QuestioningAgent and return state."""
    print("\n[執行] QuestioningAgent - CLI 互動...")
    
    # Use OrchestratorTool to get prompts
    orch_tool = OrchestratorTool()
    system_prompt_followup = orch_tool.get_system_prompt(stage_idx, "questioning_followup")
    system_prompt_compress = orch_tool.get_system_prompt(stage_idx, "questioning_compress")
    
    agent = QuestioningAgent(llm_client)
    compiled = agent.compile()
    
    state = {
        "system_prompt_followup": system_prompt_followup,
        "system_prompt_compress": system_prompt_compress,
        "current_prompt": current_prompt,
        "question_list": question_list,
        "dialogue_idx": 0,
        "answer_list": [],
        "followup_count": 0,
        "stage_idx": stage_idx
    }
    
    print_state(state, "QuestioningAgent 傳入 State")
    
    # Ask each question
    for i in range(len(question_list)):
        state["dialogue_idx"] = i
        state["stage_idx"] = stage_idx
        cli = RuntimeConfig.cli_interface
        if cli is not None and i > 0:
            cli.clear_conversation()
        result = compiled.invoke(state)
        state = result
    
    print_state(result, "QuestioningAgent 傳出 State")
    
    return result


def run_integration(llm_client, stage_idx, current_prompt, answer_list):
    """Run IntegrationAgent and return state."""
    print("\n[執行] IntegrationAgent - 整合答案...")
    
    # Use OrchestratorTool to get prompt
    orch_tool = OrchestratorTool()
    system_prompt = orch_tool.get_system_prompt(stage_idx, "integration")
    
    agent = IntegrationAgent(llm_client)
    compiled = agent.compile()
    
    state = {
        "system_prompt": system_prompt,
        "current_prompt": current_prompt,
        "answer_list": answer_list
    }
    
    print_state(state, "IntegrationAgent 傳入 State")
    
    result = compiled.invoke(state)
    
    print_state(result, "IntegrationAgent 傳出 State")
    
    return result


def main():
    parser = argparse.ArgumentParser(description='Interactive stage testing script')
    parser.add_argument('--stage', type=int, required=True, 
                       help='Stage number (1-6)')
    args = parser.parse_args()
    
    stage_idx = args.stage
    
    print("=" * 80)
    print(f" 階段測試工具 - Stage {stage_idx}")
    print("=" * 80)
    
    # Load configurations
    print("\n載入配置...")
    config = load_config("config/config.json")
    test_config = load_test_config()
    test_prompt = test_config["test_prompt"]
    
    print(f"測試 Prompt: {test_prompt}")
    
    # Setup CLI
    cli = CLIInterface(terminal_width=80)
    RuntimeConfig.cli_interface = cli
    RuntimeConfig.config_data = config
    
    # Setup LLM client
    print("\n連接 LLM...")
    
    llm_config = {
        "model": "hf.co/unsloth/gemma-3n-E4B-it-GGUF:Q4_K_M",
        "temperature": 0.7,
        "max_completion_tokens": 1000
    }
    
    llm_client = LLMClient(
        api_key="",
        base_url="http://localhost:11434/v1",
        default_config=llm_config
    )
    
    # Interactive menu
    print("\n請選擇要測試的 Agent:")
    print("1. Diagnostic (診斷並生成問題)")
    print("2. Questioning (詢問問題並收集答案)")
    print("3. Integration (整合答案改進 prompt)")
    
    choice = input("\n選擇 (1-3): ").strip()
    
    try:
        final_prompt = None
        if choice == "1":
            # Test Diagnostic only
            diagnostic_result = run_diagnostic(llm_client, stage_idx, test_prompt)
            final_prompt = diagnostic_result.get("current_prompt", test_prompt)
            
        elif choice == "2":
            # Test Questioning (auto-run Diagnostic first)
            diagnostic_result = run_diagnostic(llm_client, stage_idx, test_prompt)
            question_list = diagnostic_result["question_list"]
            current_prompt = diagnostic_result["current_prompt"]
            
            questioning_result = run_questioning(llm_client, stage_idx, current_prompt, question_list)
            final_prompt = current_prompt
            
        elif choice == "3":
            # Test Integration (auto-run Diagnostic + Questioning first)
            diagnostic_result = run_diagnostic(llm_client, stage_idx, test_prompt)
            question_list = diagnostic_result["question_list"]
            current_prompt = diagnostic_result["current_prompt"]
            
            questioning_result = run_questioning(llm_client, stage_idx, 
                                                current_prompt, question_list)
            answer_list = questioning_result["answer_list"]
            
            integration_result = run_integration(llm_client, stage_idx, current_prompt, answer_list)
            final_prompt = integration_result.get("current_prompt", current_prompt)
            
        else:
            print("無效的選擇!")
            sys.exit(1)
            
        if final_prompt:
            write_prompt_output(final_prompt)
        print("\n測試完成!")
        
    except KeyboardInterrupt:
        print("\n\n測試中斷。")
        sys.exit(0)
    except Exception as e:
        print(f"\n錯誤: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

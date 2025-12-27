# tests/manual/test_stage.py
"""
Interactive script for testing individual stages and agents with CoT support.

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
from config.runtime_config import RuntimeConfig
from config.config_loader import load_config
from cli.cli_interface import CLIInterface


# CoT prompt templates for different levels
COT_PROMPTS = {
    "low": """
請簡要分析並回答。
""",
    "medium": """
請按照以下步驟思考：
1. 首先，分析問題的關鍵要素
2. 然後，逐步推導
3. 最後，給出結論

讓我們一步一步地思考。
""",
    "high": """
請進行深度分析，按照以下步驟詳細思考：
1. 首先，完整列出所有相關資訊和假設
2. 然後，逐步分析每個要素，考慮所有可能性
3. 接著，評估每個步驟的邏輯是否嚴謹
4. 最後，綜合所有分析給出結論

讓我們非常仔細地、一步一步地分析。
"""
}


def load_test_config():
    """Load test configuration."""
    config_path = "config/test_config.json"
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_system_prompt(stage_idx, agent_type, config):
    """Load system prompt for given stage and agent type."""
    stage_names = config["stage_names"]
    stage_prompts = config["stage_prompts"]
    
    if stage_idx < 1 or stage_idx > len(stage_names):
        raise ValueError(f"Invalid stage_idx: {stage_idx}")
    
    stage_name = stage_names[stage_idx - 1]
    return stage_prompts[stage_name][agent_type]


def enhance_prompt_with_cot(system_prompt, cot_level):
    """Add CoT instructions to system prompt."""
    cot_instruction = COT_PROMPTS.get(cot_level, COT_PROMPTS["medium"])
    return system_prompt + "\n\n" + cot_instruction


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


def run_diagnostic(llm_client, stage_idx, config, test_prompt, use_cot, cot_level):
    """Run DiagnosticAgent and return state."""
    print("\n[自動執行] DiagnosticAgent - 生成問題...")
    
    system_prompt = load_system_prompt(stage_idx, "diagnostic", config)
    
    # Add CoT if enabled
    if use_cot:
        system_prompt = enhance_prompt_with_cot(system_prompt, cot_level)
        print(f"[CoT 啟用] 強度: {cot_level}")
    
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


def run_questioning(llm_client, stage_idx, config, current_prompt, question_list, 
                   use_cot, cot_level):
    """Run QuestioningAgent and return state."""
    print("\n[執行] QuestioningAgent - CLI 互動...")
    
    system_prompt = load_system_prompt(stage_idx, "questioning", config)
    
    # Add CoT if enabled
    if use_cot:
        system_prompt = enhance_prompt_with_cot(system_prompt, cot_level)
        print(f"[CoT 啟用] 強度: {cot_level}")
    
    agent = QuestioningAgent(llm_client)
    compiled = agent.compile()
    
    state = {
        "system_prompt": system_prompt,
        "current_prompt": current_prompt,
        "question_list": question_list,
        "dialogue_idx": 0,
        "answer_list": [],
        "followup_count": 0
    }
    
    print_state(state, "QuestioningAgent 傳入 State")
    
    # Ask each question
    for i in range(len(question_list)):
        result = compiled.invoke(state)
        state = result
    
    print_state(result, "QuestioningAgent 傳出 State")
    
    return result


def run_integration(llm_client, stage_idx, config, current_prompt, answer_list,
                   use_cot, cot_level):
    """Run IntegrationAgent and return state."""
    print("\n[執行] IntegrationAgent - 整合答案...")
    
    system_prompt = load_system_prompt(stage_idx, "integration", config)
    
    # Add CoT if enabled
    if use_cot:
        system_prompt = enhance_prompt_with_cot(system_prompt, cot_level)
        print(f"[CoT 啟用] 強度: {cot_level}")
    
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
    
    # CoT configuration
    use_cot = test_config.get("cot_enabled", False)
    cot_level = test_config.get("cot_level", "medium")
    
    # Ask user if they want to override CoT settings
    print(f"\n當前 CoT 設定: {'啟用' if use_cot else '停用'}, 強度: {cot_level}")
    override = input("是否要修改 CoT 設定? (y/n): ").strip().lower()
    
    if override == 'y':
        cot_choice = input("啟用 CoT? (y/n): ").strip().lower()
        use_cot = (cot_choice == 'y')
        
        if use_cot:
            print("\n選擇 CoT 強度:")
            print("1. Low (簡單推理)")
            print("2. Medium (標準推理)")
            print("3. High (深度推理)")
            
            level_choice = input("選擇 (1-3): ").strip()
            level_map = {"1": "low", "2": "medium", "3": "high"}
            cot_level = level_map.get(level_choice, "medium")
    
    print(f"\n最終 CoT 設定: {'啟用' if use_cot else '停用'}, 強度: {cot_level}")
    
    # Setup CLI
    cli = CLIInterface(terminal_width=80)
    RuntimeConfig.cli_interface = cli
    RuntimeConfig.config_data = config
    
    # Setup LLM client
    print("\n連接 LLM...")
    
    # Prepare LLM config with CoT settings
    llm_config = {
        "model": "hf.co/unsloth/gemma-3n-E4B-it-GGUF:Q4_K_M",
        "temperature": 0.7,
        "max_completion_tokens": 1000 if use_cot else 500
    }
    
    # Add reasoning_effort if CoT enabled
    if use_cot:
        llm_config["reasoning_effort"] = cot_level
    
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
        if choice == "1":
            # Test Diagnostic only
            run_diagnostic(llm_client, stage_idx, config, test_prompt, 
                         use_cot, cot_level)
            
        elif choice == "2":
            # Test Questioning (auto-run Diagnostic first)
            diagnostic_result = run_diagnostic(llm_client, stage_idx, config, 
                                              test_prompt, use_cot, cot_level)
            question_list = diagnostic_result["question_list"]
            current_prompt = diagnostic_result["current_prompt"]
            
            run_questioning(llm_client, stage_idx, config, current_prompt, 
                          question_list, use_cot, cot_level)
            
        elif choice == "3":
            # Test Integration (auto-run Diagnostic + Questioning first)
            diagnostic_result = run_diagnostic(llm_client, stage_idx, config, 
                                              test_prompt, use_cot, cot_level)
            question_list = diagnostic_result["question_list"]
            current_prompt = diagnostic_result["current_prompt"]
            
            questioning_result = run_questioning(llm_client, stage_idx, config, 
                                                current_prompt, question_list,
                                                use_cot, cot_level)
            answer_list = questioning_result["answer_list"]
            
            run_integration(llm_client, stage_idx, config, current_prompt, 
                          answer_list, use_cot, cot_level)
            
        else:
            print("無效的選擇！")
            sys.exit(1)
            
        print("\n測試完成！")
        
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
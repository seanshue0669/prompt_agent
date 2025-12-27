# tests/manual/test_cot_support.py
"""
Test if the LLM model supports Chain of Thought (CoT) reasoning.

This script tests:
1. Basic reasoning without CoT prompt
2. Reasoning with explicit CoT prompt
3. OpenAI-style reasoning_effort parameter (if supported)
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from agentcore import LLMClient


def test_basic_reasoning(client):
    """Test basic reasoning without CoT."""
    print("\n" + "=" * 80)
    print("測試 1: 基本推理（無 CoT 提示）")
    print("=" * 80)
    
    user_prompt = """
問題：小明有 15 顆蘋果，他給了小華 3 顆，然後又買了 8 顆。小明現在有多少顆蘋果？
請直接回答數字。
"""
    
    system_prompt = "你是一個數學助手。"
    
    try:
        response = client.invoke(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            config_override={}
        )
        
        print(f"\n回答: {response['content']}")
        print(f"Tokens - In: {response['tokens_in']}, Out: {response['tokens_out']}")
        
    except Exception as e:
        print(f"錯誤: {e}")


def test_explicit_cot(client):
    """Test reasoning with explicit CoT prompt."""
    print("\n" + "=" * 80)
    print("測試 2: 明確的 CoT 提示（Let's think step by step）")
    print("=" * 80)
    
    user_prompt = """
問題：小明有 15 顆蘋果，他給了小華 3 顆，然後又買了 8 顆。小明現在有多少顆蘋果？

請按照以下步驟思考：
1. 首先，列出初始狀態
2. 然後，計算每一步的變化
3. 最後，給出最終答案

讓我們一步一步地思考。
"""
    
    system_prompt = "你是一個數學助手，擅長逐步推理。"
    
    try:
        response = client.invoke(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            config_override={}
        )
        
        print(f"\n回答:\n{response['content']}")
        print(f"\nTokens - In: {response['tokens_in']}, Out: {response['tokens_out']}")
        
    except Exception as e:
        print(f"錯誤: {e}")


def test_reasoning_effort_parameter(client):
    """Test OpenAI-style reasoning_effort parameter."""
    print("\n" + "=" * 80)
    print("測試 3: reasoning_effort 參數（OpenAI o1 風格）")
    print("=" * 80)
    
    user_prompt = """
問題：小明有 15 顆蘋果，他給了小華 3 顆，然後又買了 8 顆。小明現在有多少顆蘋果？
"""
    
    system_prompt = "你是一個數學助手。"
    
    for effort in ["low", "medium", "high"]:
        print(f"\n--- Reasoning Effort: {effort} ---")
        
        try:
            response = client.invoke(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                config_override={
                    "reasoning_effort": effort
                }
            )
            
            print(f"回答: {response['content'][:200]}...")
            print(f"Tokens - In: {response['tokens_in']}, Out: {response['tokens_out']}")
            
        except Exception as e:
            print(f"不支援 reasoning_effort='{effort}': {e}")


def test_complex_reasoning(client):
    """Test complex reasoning problem."""
    print("\n" + "=" * 80)
    print("測試 4: 複雜推理問題（含 CoT）")
    print("=" * 80)
    
    user_prompt = """
問題：
有三個人 A、B、C 參加比賽。
- A 說：「我不是第一名」
- B 說：「我不是最後一名」
- C 說：「我是第一名」

已知其中只有一個人說謊，請推理出正確的名次。

請按照以下步驟思考：
1. 列出所有可能的名次組合
2. 對每種組合，檢查是否符合「只有一人說謊」的條件
3. 找出唯一符合的組合

讓我們一步一步地分析。
"""
    
    system_prompt = "你是一個邏輯推理專家，擅長逐步分析問題。"
    
    try:
        response = client.invoke(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            config_override={}
        )
        
        print(f"\n回答:\n{response['content']}")
        print(f"\nTokens - In: {response['tokens_in']}, Out: {response['tokens_out']}")
        
    except Exception as e:
        print(f"錯誤: {e}")


def main():
    print("=" * 80)
    print(" Chain of Thought (CoT) 支援測試")
    print("=" * 80)
    
    # Setup LLM client
    print("\n連接到 Ollama...")
    client = LLMClient(
        api_key="",
        base_url="http://localhost:11434/v1",
        default_config={
            "model": "hf.co/unsloth/gemma-3n-E4B-it-GGUF:Q4_K_M",
            "temperature": 0.7,
            "max_completion_tokens": 1000
        }
    )
    
    print(f"模型: hf.co/unsloth/gemma-3n-E4B-it-GGUF:Q4_K_M")
    
    # Run tests
    try:
        test_basic_reasoning(client)
        test_explicit_cot(client)
        test_reasoning_effort_parameter(client)
        test_complex_reasoning(client)
        
        print("\n" + "=" * 80)
        print(" 測試完成")
        print("=" * 80)
        print("\n分析結果:")
        print("- 如果測試 2 和 4 的回答比測試 1 更詳細，表示模型理解 CoT 提示")
        print("- 如果測試 3 沒有錯誤，表示模型支援 reasoning_effort 參數")
        print("- 如果測試 3 出現錯誤，表示需要用明確的 CoT 提示（測試 2 的方式）")
        
    except KeyboardInterrupt:
        print("\n\n測試中斷。")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n測試失敗: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
#!/usr/bin/env python
# tests/manual/test_llm_connection.py
"""
Simple test to verify Ollama connection.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from agentcore import LLMClient


def main():
    print("Testing Ollama connection...\n")
    
    # Setup LLM client
    client = LLMClient(
        api_key="",
        base_url="http://localhost:11434/v1",
        default_config={
            "model": "hf.co/unsloth/gemma-3n-E4B-it-GGUF:Q4_K_M",
            "temperature": 0.7,
            "max_completion_tokens": 100
        }
    )
    
    # Simple test
    print("Sending: 'Hello, how are you?'\n")
    
    try:
        response = client.invoke(
            user_prompt="Hello, how are you?",
            system_prompt="You are a helpful assistant.",
            config_override={}
        )
        
        print("Response:")
        print(response["content"])
        print(f"\nTokens - In: {response['tokens_in']}, Out: {response['tokens_out']}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
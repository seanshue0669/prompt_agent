# tests/integration/test_integration_agent.py
"""
Integration tests for IntegrationAgent.

Tests the complete agent including graph compilation and execution.
"""

from unittest.mock import Mock
from agentcore import test_wrapper
from agents.integration_agent import IntegrationAgent


@test_wrapper
def test_integration_agent_full_flow():
    """Test complete IntegrationAgent flow from input to output"""
    # Setup mock LLM client
    mock_client = Mock()
    mock_client.invoke.return_value = {
        "content": '{"improved_prompt": "You are a programming tutor for beginners. Explain concepts in simple terms. Provide code examples in Python. Use step-by-step breakdowns."}',
        "tokens_in": 120,
        "tokens_out": 60
    }
    
    # Create and compile agent
    agent = IntegrationAgent(mock_client)
    compiled_graph = agent.compile()
    
    # Prepare input state
    input_state = {
        "system_prompt": "Integrate answers to clarify prompt purpose.",
        "current_prompt": "Help with coding.",
        "answer_list": [
            "Beginners learning to code",
            "Simple explanations please",
            "Python examples would help",
            "Step-by-step breakdowns"
        ]
    }
    
    # Execute graph
    result = compiled_graph.invoke(input_state)
    
    # Verify output
    assert "current_prompt" in result
    assert isinstance(result["current_prompt"], str)
    
    # Verify prompt was actually improved (changed from original)
    assert result["current_prompt"] != input_state["current_prompt"]
    assert len(result["current_prompt"]) > len(input_state["current_prompt"])
    
    # Verify key concepts from answers were integrated
    improved = result["current_prompt"].lower()
    assert "beginner" in improved or "simple" in improved
    assert "python" in improved
    
    # Verify LLM was called
    assert mock_client.invoke.call_count == 1


@test_wrapper
def test_integration_agent_preserves_input_fields():
    """Test that agent preserves non-modified input fields"""
    # Setup mock LLM client
    mock_client = Mock()
    mock_client.invoke.return_value = {
        "content": '{"improved_prompt": "Improved version of the prompt."}',
        "tokens_in": 50,
        "tokens_out": 20
    }
    
    # Create and compile agent
    agent = IntegrationAgent(mock_client)
    compiled_graph = agent.compile()
    
    # Prepare input state
    input_state = {
        "system_prompt": "Test system prompt",
        "current_prompt": "Original prompt",
        "answer_list": ["Answer 1", "Answer 2"]
    }
    
    # Execute graph
    result = compiled_graph.invoke(input_state)
    
    # Verify input fields are preserved
    assert result["system_prompt"] == "Test system prompt"
    assert result["answer_list"] == ["Answer 1", "Answer 2"]
    
    # Verify current_prompt was updated
    assert result["current_prompt"] == "Improved version of the prompt."


@test_wrapper
def test_integration_agent_with_empty_answers():
    """Test agent behavior with empty answer list"""
    # Setup mock LLM client
    mock_client = Mock()
    mock_client.invoke.return_value = {
        "content": '{"improved_prompt": "Prompt with minimal changes."}',
        "tokens_in": 30,
        "tokens_out": 15
    }
    
    # Create and compile agent
    agent = IntegrationAgent(mock_client)
    compiled_graph = agent.compile()
    
    # Prepare input state with empty answers
    input_state = {
        "system_prompt": "Handle empty answers gracefully.",
        "current_prompt": "Base prompt",
        "answer_list": []
    }
    
    # Execute graph
    result = compiled_graph.invoke(input_state)
    
    # Verify it still produces output
    assert "current_prompt" in result
    assert isinstance(result["current_prompt"], str)
    assert len(result["current_prompt"]) > 0


@test_wrapper
def test_integration_agent_error_propagation():
    """Test that errors from tool are properly propagated"""
    # Setup mock LLM client that returns invalid response
    mock_client = Mock()
    mock_client.invoke.return_value = {
        "content": "not json",
        "tokens_in": 10,
        "tokens_out": 5
    }
    
    # Create and compile agent
    agent = IntegrationAgent(mock_client)
    compiled_graph = agent.compile()
    
    # Prepare input state
    input_state = {
        "system_prompt": "Test",
        "current_prompt": "Test",
        "answer_list": ["Test"]
    }
    
    # Execute graph - should raise exception
    try:
        compiled_graph.invoke(input_state)
        assert False, "Should have raised exception"
    except Exception as e:
        # Verify error contains context from error wrapping
        error_str = str(e)
        assert "IntegrationAgent" in error_str or "integrate" in error_str.lower()


# Run all tests
if __name__ == "__main__":
    print("Running IntegrationAgent integration tests...\n")
    
    test_integration_agent_full_flow()
    test_integration_agent_preserves_input_fields()
    test_integration_agent_with_empty_answers()
    test_integration_agent_error_propagation()
    
    print("\nAll tests completed!")
# tests/integration/test_diagnostic_agent.py
"""
Integration tests for DiagnosticAgent.

Tests the complete agent including graph compilation and execution.
"""

from unittest.mock import Mock
from agentcore import test_wrapper
from agents.diagnostic_agent import DiagnosticAgent


@test_wrapper
def test_diagnostic_agent_full_flow():
    """Test complete DiagnosticAgent flow from input to output"""
    # Setup mock LLM client
    mock_client = Mock()
    mock_client.invoke.return_value = {
        "content": '{"questions": ["What is the target audience?", "What format is preferred?"]}',
        "tokens_in": 100,
        "tokens_out": 50
    }
    
    # Create and compile agent
    agent = DiagnosticAgent(mock_client)
    compiled_graph = agent.compile()
    
    # Prepare input state
    input_state = {
        "system_prompt": "Analyze for input/output clarity.",
        "current_prompt": "Create a teaching assistant for students.",
        "question_list": []
    }
    
    # Execute graph
    result = compiled_graph.invoke(input_state)
    
    # Verify output
    assert "question_list" in result
    assert isinstance(result["question_list"], list)
    assert len(result["question_list"]) == 2
    assert "target audience" in result["question_list"][0].lower()
    assert "format" in result["question_list"][1].lower()
    
    # Verify LLM was called
    assert mock_client.invoke.call_count == 1


@test_wrapper
def test_diagnostic_agent_preserves_input_state():
    """Test that agent preserves input fields in state"""
    # Setup mock LLM client
    mock_client = Mock()
    mock_client.invoke.return_value = {
        "content": '{"questions": ["Q1?", "Q2?"]}',
        "tokens_in": 50,
        "tokens_out": 20
    }
    
    # Create and compile agent
    agent = DiagnosticAgent(mock_client)
    compiled_graph = agent.compile()
    
    # Prepare input state
    input_state = {
        "system_prompt": "Test system prompt",
        "current_prompt": "Test current prompt",
        "question_list": []
    }
    
    # Execute graph
    result = compiled_graph.invoke(input_state)
    
    # Verify input fields are preserved
    assert result["system_prompt"] == "Test system prompt"
    assert result["current_prompt"] == "Test current prompt"
    
    # Verify output field is populated
    assert len(result["question_list"]) == 2


@test_wrapper
def test_diagnostic_agent_error_propagation():
    """Test that errors from tool are properly propagated"""
    # Setup mock LLM client that returns invalid response
    mock_client = Mock()
    mock_client.invoke.return_value = {
        "content": "invalid json",
        "tokens_in": 10,
        "tokens_out": 5
    }
    
    # Create and compile agent
    agent = DiagnosticAgent(mock_client)
    compiled_graph = agent.compile()
    
    # Prepare input state
    input_state = {
        "system_prompt": "Test",
        "current_prompt": "Test",
        "question_list": []
    }
    
    # Execute graph - should raise exception
    try:
        compiled_graph.invoke(input_state)
        assert False, "Should have raised exception"
    except Exception as e:
        # Verify error contains context from error wrapping
        error_str = str(e)
        assert "DiagnosticAgent" in error_str or "diagnose" in error_str.lower()


# Run all tests
if __name__ == "__main__":
    print("Running DiagnosticAgent integration tests...\n")
    
    test_diagnostic_agent_full_flow()
    test_diagnostic_agent_preserves_input_state()
    test_diagnostic_agent_error_propagation()
    
    print("\nAll tests completed!")
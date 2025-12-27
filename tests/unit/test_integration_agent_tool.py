# tests/unit/test_integration_agent_tool.py
"""
Unit tests for IntegrationAgentTool.

Tests the tool's ability to integrate user answers into prompts.
"""

from unittest.mock import Mock
from agentcore import test_wrapper
from agents.integration_agent.tool import IntegrationAgentTool


@test_wrapper
def test_integrate_answers_success():
    """Test successful answer integration with valid JSON response"""
    # Setup mock LLM client
    mock_client = Mock()
    mock_client.invoke.return_value = {
        "content": '{"improved_prompt": "You are a friendly teaching assistant for university students. Use clear explanations with examples. Format output as structured lessons."}',
        "tokens_in": 150,
        "tokens_out": 80
    }
    
    # Create tool
    tool = IntegrationAgentTool(mock_client)
    
    # Call integrate_answers
    system_prompt = "Integrate user answers to improve prompt clarity."
    current_prompt = "Help students with homework."
    answer_list = [
        "University level students",
        "Clear explanations with examples",
        "Structured lesson format"
    ]
    
    result = tool.integrate_answers(system_prompt, current_prompt, answer_list)
    
    # Verify
    assert isinstance(result, str)
    assert len(result) > len(current_prompt)  # Should be more detailed
    assert "university students" in result.lower()
    assert "examples" in result.lower()
    
    # Verify LLM was called correctly
    assert mock_client.invoke.call_count == 1
    call_args = mock_client.invoke.call_args
    assert "Help students with homework" in call_args.kwargs["user_prompt"]
    assert "1. University level students" in call_args.kwargs["user_prompt"]
    assert call_args.kwargs["system_prompt"] == system_prompt
    assert call_args.kwargs["config_override"]["response_format"]["type"] == "json_object"


@test_wrapper
def test_integrate_answers_empty_answer_list():
    """Test integration with empty answer list"""
    # Setup mock LLM client
    mock_client = Mock()
    mock_client.invoke.return_value = {
        "content": '{"improved_prompt": "Original prompt remains mostly unchanged."}',
        "tokens_in": 50,
        "tokens_out": 20
    }
    
    # Create tool
    tool = IntegrationAgentTool(mock_client)
    
    # Call with empty answers
    result = tool.integrate_answers("system", "original prompt", [])
    
    # Verify it still works (LLM handles empty case)
    assert isinstance(result, str)
    assert len(result) > 0


@test_wrapper
def test_integrate_answers_empty_content():
    """Test error handling when LLM returns empty content"""
    # Setup mock LLM client that returns empty content
    mock_client = Mock()
    mock_client.invoke.return_value = {
        "content": "",
        "tokens_in": 10,
        "tokens_out": 0
    }
    
    # Create tool
    tool = IntegrationAgentTool(mock_client)
    
    # Call integrate_answers - should raise exception
    try:
        tool.integrate_answers("system", "prompt", ["answer"])
        assert False, "Should have raised exception"
    except Exception as e:
        assert "empty content" in str(e).lower()


@test_wrapper
def test_integrate_answers_invalid_json():
    """Test error handling when LLM returns invalid JSON"""
    # Setup mock LLM client that returns invalid JSON
    mock_client = Mock()
    mock_client.invoke.return_value = {
        "content": "This is not valid JSON",
        "tokens_in": 10,
        "tokens_out": 5
    }
    
    # Create tool
    tool = IntegrationAgentTool(mock_client)
    
    # Call integrate_answers - should raise exception
    try:
        tool.integrate_answers("system", "prompt", ["answer"])
        assert False, "Should have raised exception"
    except Exception as e:
        assert "json" in str(e).lower()


@test_wrapper
def test_integrate_answers_missing_field():
    """Test error handling when JSON missing 'improved_prompt' field"""
    # Setup mock LLM client
    mock_client = Mock()
    mock_client.invoke.return_value = {
        "content": '{"wrong_field": "some text"}',
        "tokens_in": 10,
        "tokens_out": 5
    }
    
    # Create tool
    tool = IntegrationAgentTool(mock_client)
    
    # Call integrate_answers - should raise exception
    try:
        tool.integrate_answers("system", "prompt", ["answer"])
        assert False, "Should have raised exception"
    except Exception as e:
        assert "improved_prompt" in str(e).lower()


@test_wrapper
def test_integrate_answers_empty_improved_prompt():
    """Test error handling when improved_prompt is empty"""
    # Setup mock LLM client
    mock_client = Mock()
    mock_client.invoke.return_value = {
        "content": '{"improved_prompt": ""}',
        "tokens_in": 10,
        "tokens_out": 5
    }
    
    # Create tool
    tool = IntegrationAgentTool(mock_client)
    
    # Call integrate_answers - should raise exception
    try:
        tool.integrate_answers("system", "prompt", ["answer"])
        assert False, "Should have raised exception"
    except Exception as e:
        assert "empty" in str(e).lower()


@test_wrapper
def test_integrate_answers_wrong_type():
    """Test error handling when improved_prompt is not a string"""
    # Setup mock LLM client
    mock_client = Mock()
    mock_client.invoke.return_value = {
        "content": '{"improved_prompt": ["not", "a", "string"]}',
        "tokens_in": 10,
        "tokens_out": 5
    }
    
    # Create tool
    tool = IntegrationAgentTool(mock_client)
    
    # Call integrate_answers - should raise exception
    try:
        tool.integrate_answers("system", "prompt", ["answer"])
        assert False, "Should have raised exception"
    except Exception as e:
        assert "string" in str(e).lower()


# Run all tests
if __name__ == "__main__":
    print("Running IntegrationAgentTool unit tests...\n")
    
    test_integrate_answers_success()
    test_integrate_answers_empty_answer_list()
    test_integrate_answers_empty_content()
    test_integrate_answers_invalid_json()
    test_integrate_answers_missing_field()
    test_integrate_answers_empty_improved_prompt()
    test_integrate_answers_wrong_type()
    
    print("\nAll tests completed!")
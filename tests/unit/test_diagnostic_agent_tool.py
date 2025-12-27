# tests/unit/test_diagnostic_agent_tool.py
"""
Unit tests for DiagnosticAgentTool.

Tests the tool's ability to call LLM and parse JSON responses.
"""

from unittest.mock import Mock
from agentcore import test_wrapper
from agents.diagnostic_agent.tool import DiagnosticAgentTool


@test_wrapper
def test_diagnose_prompt_success():
    """Test successful prompt diagnosis with valid JSON response"""
    # Setup mock LLM client
    mock_client = Mock()
    mock_client.invoke.return_value = {
        "content": '{"questions": ["Question 1?", "Question 2?", "Question 3?"]}',
        "tokens_in": 100,
        "tokens_out": 50
    }
    
    # Create tool
    tool = DiagnosticAgentTool(mock_client)
    
    # Call diagnose_prompt
    system_prompt = "Analyze the prompt for goal clarity."
    current_prompt = "Help me with my homework."
    
    result = tool.diagnose_prompt(system_prompt, current_prompt)
    
    # Verify
    assert isinstance(result, list)
    assert len(result) == 3
    assert result[0] == "Question 1?"
    assert result[1] == "Question 2?"
    assert result[2] == "Question 3?"
    
    # Verify LLM was called correctly
    assert mock_client.invoke.call_count == 1
    call_args = mock_client.invoke.call_args
    assert "Help me with my homework" in call_args.kwargs["user_prompt"]
    assert call_args.kwargs["system_prompt"] == system_prompt
    assert call_args.kwargs["config_override"]["response_format"]["type"] == "json_object"


@test_wrapper
def test_diagnose_prompt_empty_content():
    """Test error handling when LLM returns empty content"""
    # Setup mock LLM client that returns empty content
    mock_client = Mock()
    mock_client.invoke.return_value = {
        "content": "",
        "tokens_in": 10,
        "tokens_out": 0
    }
    
    # Create tool
    tool = DiagnosticAgentTool(mock_client)
    
    # Call diagnose_prompt - should raise exception
    try:
        tool.diagnose_prompt("system", "prompt")
        assert False, "Should have raised exception"
    except Exception as e:
        assert "empty content" in str(e).lower()


@test_wrapper
def test_diagnose_prompt_invalid_json():
    """Test error handling when LLM returns invalid JSON"""
    # Setup mock LLM client that returns invalid JSON
    mock_client = Mock()
    mock_client.invoke.return_value = {
        "content": "This is not JSON",
        "tokens_in": 10,
        "tokens_out": 5
    }
    
    # Create tool
    tool = DiagnosticAgentTool(mock_client)
    
    # Call diagnose_prompt - should raise exception
    try:
        tool.diagnose_prompt("system", "prompt")
        assert False, "Should have raised exception"
    except Exception as e:
        assert "json" in str(e).lower()


@test_wrapper
def test_diagnose_prompt_missing_questions_field():
    """Test error handling when JSON missing 'questions' field"""
    # Setup mock LLM client
    mock_client = Mock()
    mock_client.invoke.return_value = {
        "content": '{"answers": ["wrong", "field"]}',
        "tokens_in": 10,
        "tokens_out": 5
    }
    
    # Create tool
    tool = DiagnosticAgentTool(mock_client)
    
    # Call diagnose_prompt - should raise exception
    try:
        tool.diagnose_prompt("system", "prompt")
        assert False, "Should have raised exception"
    except Exception as e:
        assert "questions" in str(e).lower()


@test_wrapper
def test_diagnose_prompt_empty_question_list():
    """Test error handling when questions list is empty"""
    # Setup mock LLM client
    mock_client = Mock()
    mock_client.invoke.return_value = {
        "content": '{"questions": []}',
        "tokens_in": 10,
        "tokens_out": 5
    }
    
    # Create tool
    tool = DiagnosticAgentTool(mock_client)
    
    # Call diagnose_prompt - should raise exception
    try:
        tool.diagnose_prompt("system", "prompt")
        assert False, "Should have raised exception"
    except Exception as e:
        assert "empty" in str(e).lower()


@test_wrapper
def test_diagnose_prompt_questions_not_list():
    """Test error handling when questions field is not a list"""
    # Setup mock LLM client
    mock_client = Mock()
    mock_client.invoke.return_value = {
        "content": '{"questions": "not a list"}',
        "tokens_in": 10,
        "tokens_out": 5
    }
    
    # Create tool
    tool = DiagnosticAgentTool(mock_client)
    
    # Call diagnose_prompt - should raise exception
    try:
        tool.diagnose_prompt("system", "prompt")
        assert False, "Should have raised exception"
    except Exception as e:
        assert "list" in str(e).lower()


# Run all tests
if __name__ == "__main__":
    print("Running DiagnosticAgentTool unit tests...\n")
    
    test_diagnose_prompt_success()
    test_diagnose_prompt_empty_content()
    test_diagnose_prompt_invalid_json()
    test_diagnose_prompt_missing_questions_field()
    test_diagnose_prompt_empty_question_list()
    test_diagnose_prompt_questions_not_list()
    
    print("\nAll tests completed!")
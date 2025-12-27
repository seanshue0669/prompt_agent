# tests/unit/test_questioning_agent_tool.py
"""
Unit tests for QuestioningAgentTool.

Tests the tool's ability to ask questions via CLI and determine followup needs.
"""

from unittest.mock import Mock
from agentcore import test_wrapper
from agents.questioning_agent.tool import QuestioningAgentTool
from config.runtime_config import RuntimeConfig


@test_wrapper
def test_ask_question_and_collect_success():
    """Test successful question asking and answer collection via CLI"""
    # Setup mock CLI
    mock_cli = Mock()
    mock_cli.get_user_input.return_value = "I want a friendly teaching assistant"
    RuntimeConfig.cli_interface = mock_cli
    
    # Setup mock LLM client (not used in this method, but required for init)
    mock_client = Mock()
    
    # Create tool
    tool = QuestioningAgentTool(mock_client)
    
    # Ask question and collect answer
    answer = tool.ask_question_and_collect(
        system_prompt="Test system prompt",
        question="What kind of assistant do you want?",
        stage_idx=1,
        question_idx=1,
        total_questions=3
    )
    
    # Verify
    assert answer == "I want a friendly teaching assistant"
    
    # Verify CLI was called correctly
    assert mock_cli.update_stage.call_count == 1
    assert mock_cli.get_user_input.call_count == 1
    
    stage_call = mock_cli.update_stage.call_args
    assert stage_call.kwargs["stage_idx"] == 1
    assert stage_call.kwargs["substage"] == "對話"
    assert stage_call.kwargs["question_idx"] == 1
    assert stage_call.kwargs["total_questions"] == 3


@test_wrapper
def test_ask_question_no_cli():
    """Test error when CLI not initialized"""
    # Clear CLI from RuntimeConfig
    RuntimeConfig.cli_interface = None
    
    # Setup mock LLM client
    mock_client = Mock()
    
    # Create tool
    tool = QuestioningAgentTool(mock_client)
    
    # Try to ask question - should raise exception
    try:
        tool.ask_question_and_collect("sys", "question", 1, 1, 1)
        assert False, "Should have raised exception"
    except Exception as e:
        assert "CLI interface not initialized" in str(e)


@test_wrapper
def test_should_followup_yes():
    """Test followup determination when followup is needed"""
    # Setup mock LLM client
    mock_client = Mock()
    mock_client.invoke.return_value = {
        "content": '{"need_followup": true, "followup_question": "Can you be more specific about what you mean by friendly?"}',
        "tokens_in": 80,
        "tokens_out": 40
    }
    
    # Create tool
    tool = QuestioningAgentTool(mock_client)
    
    # Check if followup needed
    result = tool.should_followup(
        system_prompt="Ask followup if answer is vague",
        question="What kind of assistant?",
        answer="A friendly one",
        followup_count=0,
        max_followup=2
    )
    
    # Verify
    assert result["need_followup"] == True
    assert "specific" in result["followup_question"].lower()
    
    # Verify LLM was called
    assert mock_client.invoke.call_count == 1


@test_wrapper
def test_should_followup_no():
    """Test followup determination when answer is clear enough"""
    # Setup mock LLM client
    mock_client = Mock()
    mock_client.invoke.return_value = {
        "content": '{"need_followup": false}',
        "tokens_in": 70,
        "tokens_out": 20
    }
    
    # Create tool
    tool = QuestioningAgentTool(mock_client)
    
    # Check if followup needed
    result = tool.should_followup(
        system_prompt="Ask followup if answer is vague",
        question="What kind of assistant?",
        answer="A friendly teaching assistant that explains concepts clearly with examples",
        followup_count=0,
        max_followup=2
    )
    
    # Verify
    assert result["need_followup"] == False
    assert result["followup_question"] is None


@test_wrapper
def test_should_followup_at_max_limit():
    """Test that no followup is generated when at max limit"""
    # Setup mock LLM client (should not be called)
    mock_client = Mock()
    
    # Create tool
    tool = QuestioningAgentTool(mock_client)
    
    # Check followup at max limit
    result = tool.should_followup(
        system_prompt="Test",
        question="Test question",
        answer="Test answer",
        followup_count=2,  # Already at max
        max_followup=2
    )
    
    # Verify no followup (without calling LLM)
    assert result["need_followup"] == False
    assert result["followup_question"] is None
    assert mock_client.invoke.call_count == 0


@test_wrapper
def test_should_followup_missing_need_followup_field():
    """Test error when LLM response missing need_followup field"""
    # Setup mock LLM client
    mock_client = Mock()
    mock_client.invoke.return_value = {
        "content": '{"wrong_field": true}',
        "tokens_in": 10,
        "tokens_out": 5
    }
    
    # Create tool
    tool = QuestioningAgentTool(mock_client)
    
    # Try to check followup - should raise exception
    try:
        tool.should_followup("sys", "q", "a", 0, 2)
        assert False, "Should have raised exception"
    except Exception as e:
        assert "need_followup" in str(e).lower()


@test_wrapper
def test_should_followup_missing_followup_question():
    """Test error when followup needed but question missing"""
    # Setup mock LLM client
    mock_client = Mock()
    mock_client.invoke.return_value = {
        "content": '{"need_followup": true}',  # Missing followup_question
        "tokens_in": 10,
        "tokens_out": 5
    }
    
    # Create tool
    tool = QuestioningAgentTool(mock_client)
    
    # Try to check followup - should raise exception
    try:
        tool.should_followup("sys", "q", "a", 0, 2)
        assert False, "Should have raised exception"
    except Exception as e:
        assert "followup_question" in str(e).lower()


@test_wrapper
def test_should_followup_empty_followup_question():
    """Test error when followup question is empty"""
    # Setup mock LLM client
    mock_client = Mock()
    mock_client.invoke.return_value = {
        "content": '{"need_followup": true, "followup_question": ""}',
        "tokens_in": 10,
        "tokens_out": 5
    }
    
    # Create tool
    tool = QuestioningAgentTool(mock_client)
    
    # Try to check followup - should raise exception
    try:
        tool.should_followup("sys", "q", "a", 0, 2)
        assert False, "Should have raised exception"
    except Exception as e:
        assert "empty" in str(e).lower()


# Run all tests
if __name__ == "__main__":
    print("Running QuestioningAgentTool unit tests...\n")
    
    test_ask_question_and_collect_success()
    test_ask_question_no_cli()
    test_should_followup_yes()
    test_should_followup_no()
    test_should_followup_at_max_limit()
    test_should_followup_missing_need_followup_field()
    test_should_followup_missing_followup_question()
    test_should_followup_empty_followup_question()
    
    print("\nAll tests completed!")
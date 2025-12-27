# tests/integration/test_questioning_agent.py
"""
Integration tests for QuestioningAgent.

Tests the complete agent including graph compilation and CLI interaction.
"""

from unittest.mock import Mock
from agentcore import test_wrapper
from agents.questioning_agent import QuestioningAgent
from config.runtime_config import RuntimeConfig


@test_wrapper
def test_questioning_agent_single_question_no_followup():
    """Test asking a single question without followup"""
    # Setup mock CLI
    mock_cli = Mock()
    mock_cli.get_user_input.return_value = "University students learning programming"
    RuntimeConfig.cli_interface = mock_cli
    
    # Setup mock config
    RuntimeConfig.config_data = {"max_followup_count": 2}
    
    # Setup mock LLM client (for followup check)
    mock_client = Mock()
    mock_client.invoke.return_value = {
        "content": '{"need_followup": false}',
        "tokens_in": 50,
        "tokens_out": 20
    }
    
    # Create and compile agent
    agent = QuestioningAgent(mock_client)
    compiled_graph = agent.compile()
    
    # Prepare input state
    input_state = {
        "system_prompt": "Ask questions clearly",
        "question_list": ["Who is your target audience?"],
        "dialogue_idx": 0,
        "answer_list": [],
        "followup_count": 0
    }
    
    # Execute graph
    result = compiled_graph.invoke(input_state)
    
    # Verify answer was collected
    assert len(result["answer_list"]) == 1
    assert result["answer_list"][0] == "University students learning programming"
    
    # Verify followup_count was reset
    assert result["followup_count"] == 0
    
    # Verify CLI was used
    assert mock_cli.get_user_input.call_count == 1


@test_wrapper
def test_questioning_agent_with_followup():
    """Test asking a question with one followup"""
    # Setup mock CLI (will be called twice: original + followup)
    mock_cli = Mock()
    mock_cli.get_user_input.side_effect = [
        "Students",  # Vague answer
        "University level students studying computer science"  # Better answer after followup
    ]
    RuntimeConfig.cli_interface = mock_cli
    
    # Setup mock config
    RuntimeConfig.config_data = {"max_followup_count": 2}
    
    # Setup mock LLM client
    mock_client = Mock()
    # First call: need followup
    # Second call: no more followup needed (but won't be called due to implementation)
    mock_client.invoke.return_value = {
        "content": '{"need_followup": true, "followup_question": "Can you be more specific about what level and subject?"}',
        "tokens_in": 60,
        "tokens_out": 30
    }
    
    # Create and compile agent
    agent = QuestioningAgent(mock_client)
    compiled_graph = agent.compile()
    
    # Prepare input state
    input_state = {
        "system_prompt": "Ask followup if answer is vague",
        "question_list": ["Who is your target audience?"],
        "dialogue_idx": 0,
        "answer_list": [],
        "followup_count": 0
    }
    
    # Execute graph
    result = compiled_graph.invoke(input_state)
    
    # Verify both answers were collected
    assert len(result["answer_list"]) == 2
    assert result["answer_list"][0] == "Students"
    assert "computer science" in result["answer_list"][1].lower()
    
    # Verify followup_count was incremented
    assert result["followup_count"] == 1
    
    # Verify CLI was called twice
    assert mock_cli.get_user_input.call_count == 2


@test_wrapper
def test_questioning_agent_max_followup_reached():
    """Test that followup stops at max limit"""
    # Setup mock CLI
    mock_cli = Mock()
    mock_cli.get_user_input.return_value = "Vague answer"
    RuntimeConfig.cli_interface = mock_cli
    
    # Setup mock config with max_followup = 1
    RuntimeConfig.config_data = {"max_followup_count": 1}
    
    # Setup mock LLM client (should not be called when at limit)
    mock_client = Mock()
    
    # Create and compile agent
    agent = QuestioningAgent(mock_client)
    compiled_graph = agent.compile()
    
    # Prepare input state already at max followup
    input_state = {
        "system_prompt": "Test",
        "question_list": ["Test question?"],
        "dialogue_idx": 0,
        "answer_list": [],
        "followup_count": 1  # Already at max
    }
    
    # Execute graph
    result = compiled_graph.invoke(input_state)
    
    # Verify only one answer collected (no followup generated)
    assert len(result["answer_list"]) == 1
    
    # Verify LLM was called for followup check and it correctly stopped
    # (The should_followup method returns early without calling LLM when at max)


@test_wrapper
def test_questioning_agent_preserves_existing_answers():
    """Test that agent preserves answers from previous questions"""
    # Setup mock CLI
    mock_cli = Mock()
    mock_cli.get_user_input.return_value = "New answer"
    RuntimeConfig.cli_interface = mock_cli
    
    # Setup mock config
    RuntimeConfig.config_data = {"max_followup_count": 2}
    
    # Setup mock LLM client
    mock_client = Mock()
    mock_client.invoke.return_value = {
        "content": '{"need_followup": false}',
        "tokens_in": 40,
        "tokens_out": 15
    }
    
    # Create and compile agent
    agent = QuestioningAgent(mock_client)
    compiled_graph = agent.compile()
    
    # Prepare input state with existing answers
    input_state = {
        "system_prompt": "Test",
        "question_list": ["Question 1?", "Question 2?"],
        "dialogue_idx": 1,  # Asking second question
        "answer_list": ["Previous answer"],  # Already has one answer
        "followup_count": 0
    }
    
    # Execute graph
    result = compiled_graph.invoke(input_state)
    
    # Verify previous answer preserved and new answer added
    assert len(result["answer_list"]) == 2
    assert result["answer_list"][0] == "Previous answer"
    assert result["answer_list"][1] == "New answer"


@test_wrapper
def test_questioning_agent_error_on_invalid_dialogue_idx():
    """Test error handling when dialogue_idx is out of range"""
    # Setup mock CLI
    mock_cli = Mock()
    RuntimeConfig.cli_interface = mock_cli
    
    # Setup mock config
    RuntimeConfig.config_data = {"max_followup_count": 2}
    
    # Setup mock LLM client
    mock_client = Mock()
    
    # Create and compile agent
    agent = QuestioningAgent(mock_client)
    compiled_graph = agent.compile()
    
    # Prepare input state with invalid dialogue_idx
    input_state = {
        "system_prompt": "Test",
        "question_list": ["Question 1?"],
        "dialogue_idx": 5,  # Out of range
        "answer_list": [],
        "followup_count": 0
    }
    
    # Execute graph - should raise exception
    try:
        compiled_graph.invoke(input_state)
        assert False, "Should have raised exception"
    except Exception as e:
        assert "out of range" in str(e).lower()


# Run all tests
if __name__ == "__main__":
    print("Running QuestioningAgent integration tests...\n")
    
    test_questioning_agent_single_question_no_followup()
    test_questioning_agent_with_followup()
    test_questioning_agent_max_followup_reached()
    test_questioning_agent_preserves_existing_answers()
    test_questioning_agent_error_on_invalid_dialogue_idx()
    
    print("\nAll tests completed!")
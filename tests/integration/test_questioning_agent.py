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
    mock_cli.get_user_input.return_value = "大學生學習程式設計"
    RuntimeConfig.cli_interface = mock_cli
    
    # Setup mock config
    RuntimeConfig.config_data = {"max_followup_count": 2}
    
    # Setup mock LLM client (for followup check and compress)
    mock_client = Mock()
    mock_client.invoke.side_effect = [
        # Followup check: not needed
        {"content": '{"need_followup": false}', "tokens_in": 50, "tokens_out": 20},
        # Compress
        {
            "content": '{"思考過程": {"步驟1_對話要素": "test", "步驟2_關鍵資訊": "test", "步驟3_整合資訊": "test", "步驟4_生成答案": "test", "步驟5_驗證": "test"}, "compressed": "Q: 目標受眾? A: 大學程式初學者"}',
            "tokens_in": 80,
            "tokens_out": 60
        }
    ]
    
    # Create and compile agent
    agent = QuestioningAgent(mock_client)
    compiled_graph = agent.compile()
    
    # Prepare input state
    input_state = {
        "system_prompt_followup": "Test followup",
        "system_prompt_compress": "Test compress",
        "question_list": ["誰是你的目標受眾?"],
        "dialogue_idx": 0,
        "answer_list": [],
        "followup_count": 0
    }
    
    # Execute graph
    result = compiled_graph.invoke(input_state)
    
    # Verify answer was collected
    assert len(result["answer_list"]) == 1
    assert "大學程式初學者" in result["answer_list"][0]
    
    # Verify followup_count was reset
    assert result["followup_count"] == 0
    
    # Verify CLI was used once (no options)
    assert mock_cli.get_user_input.call_count == 1
    call_args = mock_cli.get_user_input.call_args
    assert call_args.kwargs.get('options') is None


@test_wrapper
def test_questioning_agent_with_followup():
    """Test asking a question with one followup"""
    # Setup mock CLI (will be called twice: original + followup)
    mock_cli = Mock()
    mock_cli.get_user_input.side_effect = [
        "不知道",  # Vague answer
        "B) 實作導向"  # Better answer after followup
    ]
    RuntimeConfig.cli_interface = mock_cli
    
    # Setup mock config
    RuntimeConfig.config_data = {"max_followup_count": 2}
    
    # Setup mock LLM client
    mock_client = Mock()
    mock_client.invoke.side_effect = [
        # First call: need followup with options
        {
            "content": '{"need_followup": true, "followup_question": "請選擇教學方式:", "options": ["A) 逐步講解", "B) 實作導向", "C) 其他", "D) 沒想法"]}',
            "tokens_in": 60,
            "tokens_out": 40
        },
        # Second call: no more followup needed
        {"content": '{"need_followup": false}', "tokens_in": 50, "tokens_out": 20},
        # Third call: compress
        {
            "content": '{"思考過程": {"步驟1_對話要素": "test", "步驟2_關鍵資訊": "test", "步驟3_整合資訊": "test", "步驟4_生成答案": "test", "步驟5_驗證": "test"}, "compressed": "Q: 教學方式? A: 實作導向"}',
            "tokens_in": 100,
            "tokens_out": 80
        }
    ]
    
    # Create and compile agent
    agent = QuestioningAgent(mock_client)
    compiled_graph = agent.compile()
    
    # Prepare input state
    input_state = {
        "system_prompt_followup": "Test followup",
        "system_prompt_compress": "Test compress",
        "question_list": ["你的教學方式偏好?"],
        "dialogue_idx": 0,
        "answer_list": [],
        "followup_count": 0
    }
    
    # Execute graph
    result = compiled_graph.invoke(input_state)
    
    # Verify answer was collected and compressed
    assert len(result["answer_list"]) == 1
    assert "實作導向" in result["answer_list"][0]
    
    # Verify followup_count was reset
    assert result["followup_count"] == 0
    
    # Verify CLI was called twice
    assert mock_cli.get_user_input.call_count == 2
    
    # Verify first call had no options
    first_call = mock_cli.get_user_input.call_args_list[0]
    assert first_call.kwargs.get('options') is None
    
    # Verify second call had options
    second_call = mock_cli.get_user_input.call_args_list[1]
    assert second_call.kwargs.get('options') is not None
    assert len(second_call.kwargs['options']) == 4


@test_wrapper
def test_questioning_agent_max_followup_reached():
    """Test that followup stops at max limit"""
    # Setup mock CLI
    mock_cli = Mock()
    mock_cli.get_user_input.side_effect = ["模糊", "A) 選項1"]
    RuntimeConfig.cli_interface = mock_cli
    
    # Setup mock config with max_followup = 1
    RuntimeConfig.config_data = {"max_followup_count": 1}
    
    # Setup mock LLM client
    mock_client = Mock()
    mock_client.invoke.side_effect = [
        # First followup check: needed
        {
            "content": '{"need_followup": true, "followup_question": "請選擇:", "options": ["A) 選項1", "B) 選項2", "C) 其他", "D) 沒想法"]}',
            "tokens_in": 50,
            "tokens_out": 30
        },
        # Second followup check: at max, loop stops before this is called
        # Compress
        {
            "content": '{"思考過程": {"步驟1_對話要素": "test", "步驟2_關鍵資訊": "test", "步驟3_整合資訊": "test", "步驟4_生成答案": "test", "步驟5_驗證": "test"}, "compressed": "Q: 問題? A: 選項1"}',
            "tokens_in": 80,
            "tokens_out": 60
        }
    ]
    
    # Create and compile agent
    agent = QuestioningAgent(mock_client)
    compiled_graph = agent.compile()
    
    # Prepare input state
    input_state = {
        "system_prompt_followup": "Test",
        "system_prompt_compress": "Test",
        "question_list": ["測試問題?"],
        "dialogue_idx": 0,
        "answer_list": [],
        "followup_count": 0
    }
    
    # Execute graph
    result = compiled_graph.invoke(input_state)
    
    # Verify answer collected
    assert len(result["answer_list"]) == 1
    
    # Verify CLI was called twice (original + 1 followup at max=1)
    assert mock_cli.get_user_input.call_count == 2


@test_wrapper
def test_questioning_agent_preserves_existing_answers():
    """Test that agent preserves answers from previous questions"""
    # Setup mock CLI
    mock_cli = Mock()
    mock_cli.get_user_input.return_value = "新答案"
    RuntimeConfig.cli_interface = mock_cli
    
    # Setup mock config
    RuntimeConfig.config_data = {"max_followup_count": 2}
    
    # Setup mock LLM client
    mock_client = Mock()
    mock_client.invoke.side_effect = [
        # Followup check: not needed
        {"content": '{"need_followup": false}', "tokens_in": 40, "tokens_out": 15},
        # Compress
        {
            "content": '{"思考過程": {"步驟1_對話要素": "test", "步驟2_關鍵資訊": "test", "步驟3_整合資訊": "test", "步驟4_生成答案": "test", "步驟5_驗證": "test"}, "compressed": "Q: 問題2? A: 新答案"}',
            "tokens_in": 60,
            "tokens_out": 40
        }
    ]
    
    # Create and compile agent
    agent = QuestioningAgent(mock_client)
    compiled_graph = agent.compile()
    
    # Prepare input state with existing answers
    input_state = {
        "system_prompt_followup": "Test",
        "system_prompt_compress": "Test",
        "question_list": ["問題1?", "問題2?"],
        "dialogue_idx": 1,  # Asking second question
        "answer_list": ["Q: 問題1? A: 之前的答案"],  # Already has one answer
        "followup_count": 0
    }
    
    # Execute graph
    result = compiled_graph.invoke(input_state)
    
    # Verify previous answer preserved and new answer added
    assert len(result["answer_list"]) == 2
    assert result["answer_list"][0] == "Q: 問題1? A: 之前的答案"
    assert "新答案" in result["answer_list"][1]


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
        "system_prompt_followup": "Test",
        "system_prompt_compress": "Test",
        "question_list": ["問題1?"],
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
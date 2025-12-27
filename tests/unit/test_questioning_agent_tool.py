# tests/unit/test_questioning_agent_tool.py
"""
Unit tests for QuestioningAgentTool.

Tests the tool's ability to manage question-answer conversations,
including followup logic and compression.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from unittest.mock import Mock, patch
from agentcore import test_wrapper
from agents.questioning_agent.tool import QuestioningAgentTool
from config.runtime_config import RuntimeConfig


@test_wrapper
def test_handle_question_conversation_no_followup():
    """Test complete conversation flow with no followup needed"""
    # Setup mock CLI
    mock_cli = Mock()
    mock_cli.get_user_input.return_value = "I want a clear, step-by-step teaching approach"
    RuntimeConfig.cli_interface = mock_cli
    
    # Setup mock LLM client
    mock_client = Mock()
    mock_client.invoke.side_effect = [
        # First call: check followup (not needed)
        {"content": '{"need_followup": false}', "tokens_in": 50, "tokens_out": 20},
        # Second call: compress conversation
        {
            "content": '{"思考過程": {"步驟1_對話要素": "test", "步驟2_關鍵資訊": "test", "步驟3_整合資訊": "test", "步驟4_生成答案": "test", "步驟5_驗證": "test"}, "compressed": "Q: 教學方式? A: 清晰的逐步教學"}',
            "tokens_in": 100,
            "tokens_out": 80
        }
    ]
    
    # Create tool
    tool = QuestioningAgentTool(mock_client)
    
    # Call handle_question_conversation
    result = tool.handle_question_conversation(
        system_prompt_followup="Test followup prompt",
        system_prompt_compress="Test compress prompt",
        question="你希望的教學方式是什麼?",
        stage_idx=1,
        question_idx=1,
        total_questions=3,
        max_followup=2
    )
    
    # Verify result
    assert isinstance(result, str)
    assert result == "Q: 教學方式? A: 清晰的逐步教學"
    
    # Verify CLI was called once (original question only)
    assert mock_cli.get_user_input.call_count == 1
    
    # Verify LLM was called twice (followup check + compress)
    assert mock_client.invoke.call_count == 2


@test_wrapper
def test_handle_question_conversation_with_one_followup():
    """Test conversation flow with one followup question"""
    # Setup mock CLI (will be called twice: original + followup)
    mock_cli = Mock()
    mock_cli.get_user_input.side_effect = [
        "學生",  # Vague answer
        "大學生,剛學程式的"  # Better answer after followup
    ]
    RuntimeConfig.cli_interface = mock_cli
    
    # Setup mock LLM client
    mock_client = Mock()
    mock_client.invoke.side_effect = [
        # First call: check followup (needed)
        {
            "content": '{"need_followup": true, "followup_question": "什麼程度的學生?"}',
            "tokens_in": 60,
            "tokens_out": 30
        },
        # Second call: check followup again (not needed)
        {"content": '{"need_followup": false}', "tokens_in": 50, "tokens_out": 20},
        # Third call: compress conversation
        {
            "content": '{"思考過程": {"步驟1_對話要素": "test", "步驟2_關鍵資訊": "test", "步驟3_整合資訊": "test", "步驟4_生成答案": "test", "步驟5_驗證": "test"}, "compressed": "Q: 目標受眾? A: 大學程式初學者"}',
            "tokens_in": 120,
            "tokens_out": 100
        }
    ]
    
    # Create tool
    tool = QuestioningAgentTool(mock_client)
    
    # Call handle_question_conversation
    result = tool.handle_question_conversation(
        system_prompt_followup="Test followup prompt",
        system_prompt_compress="Test compress prompt",
        question="你的目標受眾是誰?",
        stage_idx=1,
        question_idx=1,
        total_questions=3,
        max_followup=2
    )
    
    # Verify result
    assert isinstance(result, str)
    assert "大學程式初學者" in result
    
    # Verify CLI was called twice (original + 1 followup)
    assert mock_cli.get_user_input.call_count == 2
    
    # Verify LLM was called 3 times (2 followup checks + 1 compress)
    assert mock_client.invoke.call_count == 3


@test_wrapper
def test_handle_question_conversation_max_followup_reached():
    """Test that followup stops at max limit"""
    # Setup mock CLI
    mock_cli = Mock()
    mock_cli.get_user_input.side_effect = ["模糊", "還是模糊", "依然模糊"]
    RuntimeConfig.cli_interface = mock_cli
    
    # Setup mock LLM client
    mock_client = Mock()
    mock_client.invoke.side_effect = [
        # First followup check: needed
        {
            "content": '{"need_followup": true, "followup_question": "能具體一點嗎?"}',
            "tokens_in": 50,
            "tokens_out": 20
        },
        # Second followup check: needed again
        {
            "content": '{"need_followup": true, "followup_question": "還是不太清楚,能再說明嗎?"}',
            "tokens_in": 50,
            "tokens_out": 20
        },
        # Third followup check would happen but max_followup=2, so loop stops
        # Compression
        {
            "content": '{"思考過程": {"步驟1_對話要素": "test", "步驟2_關鍵資訊": "test", "步驟3_整合資訊": "test", "步驟4_生成答案": "test", "步驟5_驗證": "test"}, "compressed": "Q: 測試問題? A: 無明確答案"}',
            "tokens_in": 100,
            "tokens_out": 80
        }
    ]
    
    # Create tool
    tool = QuestioningAgentTool(mock_client)
    
    # Call with max_followup=2
    result = tool.handle_question_conversation(
        system_prompt_followup="Test",
        system_prompt_compress="Test",
        question="測試問題?",
        stage_idx=1,
        question_idx=1,
        total_questions=1,
        max_followup=2
    )
    
    # Verify result exists
    assert isinstance(result, str)
    
    # Verify CLI was called 3 times (original + 2 followups, stops at max)
    assert mock_cli.get_user_input.call_count == 3
    
    # Verify LLM was called 3 times (2 followup checks + 1 compress)
    assert mock_client.invoke.call_count == 3


@test_wrapper
def test_check_followup_needed_at_max_limit():
    """Test that _check_followup_needed returns false when at max limit"""
    # Setup mock LLM client (should not be called)
    mock_client = Mock()
    
    # Create tool
    tool = QuestioningAgentTool(mock_client)
    
    # Call _check_followup_needed at max limit
    result = tool._check_followup_needed(
        system_prompt="Test",
        original_question="Test question",
        current_answer="Test answer",
        followup_count=2,  # At max
        max_followup=2
    )
    
    # Verify no followup without calling LLM
    assert result["need_followup"] == False
    assert result["followup_question"] is None
    assert mock_client.invoke.call_count == 0


@test_wrapper
def test_check_followup_needed_returns_true():
    """Test _check_followup_needed when LLM indicates followup is needed"""
    # Setup mock LLM client
    mock_client = Mock()
    mock_client.invoke.return_value = {
        "content": '{"need_followup": true, "followup_question": "能再詳細說明嗎?"}',
        "tokens_in": 50,
        "tokens_out": 20
    }
    
    # Create tool
    tool = QuestioningAgentTool(mock_client)
    
    # Call _check_followup_needed
    result = tool._check_followup_needed(
        system_prompt="Test prompt",
        original_question="你的目標?",
        current_answer="學習",
        followup_count=0,
        max_followup=2
    )
    
    # Verify result
    assert result["need_followup"] == True
    assert result["followup_question"] == "能再詳細說明嗎?"
    assert mock_client.invoke.call_count == 1


@test_wrapper
def test_check_followup_needed_returns_false():
    """Test _check_followup_needed when LLM indicates no followup needed"""
    # Setup mock LLM client
    mock_client = Mock()
    mock_client.invoke.return_value = {
        "content": '{"need_followup": false}',
        "tokens_in": 50,
        "tokens_out": 20
    }
    
    # Create tool
    tool = QuestioningAgentTool(mock_client)
    
    # Call _check_followup_needed
    result = tool._check_followup_needed(
        system_prompt="Test prompt",
        original_question="你的目標?",
        current_answer="我希望成為專業的程式設計師,專精於後端開發",
        followup_count=0,
        max_followup=2
    )
    
    # Verify result
    assert result["need_followup"] == False
    assert result["followup_question"] is None


@test_wrapper
def test_check_followup_needed_missing_field():
    """Test error when LLM response missing required field"""
    # Setup mock LLM client with invalid response
    mock_client = Mock()
    mock_client.invoke.return_value = {
        "content": '{"wrong_field": true}',
        "tokens_in": 10,
        "tokens_out": 5
    }
    
    # Create tool
    tool = QuestioningAgentTool(mock_client)
    
    # Call should raise exception
    try:
        tool._check_followup_needed("sys", "q", "a", 0, 2)
        assert False, "Should have raised exception"
    except Exception as e:
        assert "need_followup" in str(e).lower()


@test_wrapper
def test_check_followup_needed_missing_followup_question():
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
    
    # Call should raise exception
    try:
        tool._check_followup_needed("sys", "q", "a", 0, 2)
        assert False, "Should have raised exception"
    except Exception as e:
        assert "followup_question" in str(e).lower()


@test_wrapper
def test_compress_conversation_success():
    """Test successful conversation compression"""
    # Setup mock LLM client
    mock_client = Mock()
    mock_client.invoke.return_value = {
        "content": '{"思考過程": {"步驟1_對話要素": "分析", "步驟2_關鍵資訊": "提取", "步驟3_整合資訊": "整合", "步驟4_生成答案": "生成", "步驟5_驗證": "驗證"}, "compressed": "Q: 教學方式? A: 互動式,結合實例"}',
        "tokens_in": 100,
        "tokens_out": 80
    }
    
    # Create tool
    tool = QuestioningAgentTool(mock_client)
    
    # Call _compress_conversation
    conversation_history = [
        {"question": "你希望的教學方式?", "answer": "互動"},
        {"question": "能具體說明嗎?", "answer": "結合實例"}
    ]
    
    result = tool._compress_conversation(
        system_prompt="Test compress prompt",
        original_question="你希望的教學方式?",
        conversation_history=conversation_history
    )
    
    # Verify result
    assert result == "Q: 教學方式? A: 互動式,結合實例"
    assert mock_client.invoke.call_count == 1


@test_wrapper
def test_compress_conversation_missing_compressed_field():
    """Test error when LLM response missing compressed field"""
    # Setup mock LLM client with invalid response
    mock_client = Mock()
    mock_client.invoke.return_value = {
        "content": '{"思考過程": {"步驟1_對話要素": "test", "步驟2_關鍵資訊": "test", "步驟3_整合資訊": "test", "步驟4_生成答案": "test", "步驟5_驗證": "test"}}',
        "tokens_in": 50,
        "tokens_out": 30
    }
    
    # Create tool
    tool = QuestioningAgentTool(mock_client)
    
    # Call should raise exception
    try:
        tool._compress_conversation(
            "sys",
            "q",
            [{"question": "q", "answer": "a"}]
        )
        assert False, "Should have raised exception"
    except Exception as e:
        assert "compressed" in str(e).lower()


@test_wrapper
def test_compress_conversation_empty_compressed():
    """Test error when compressed result is empty"""
    # Setup mock LLM client
    mock_client = Mock()
    mock_client.invoke.return_value = {
        "content": '{"思考過程": {"步驟1_對話要素": "test", "步驟2_關鍵資訊": "test", "步驟3_整合資訊": "test", "步驟4_生成答案": "test", "步驟5_驗證": "test"}, "compressed": ""}',
        "tokens_in": 50,
        "tokens_out": 30
    }
    
    # Create tool
    tool = QuestioningAgentTool(mock_client)
    
    # Call should raise exception
    try:
        tool._compress_conversation(
            "sys",
            "q",
            [{"question": "q", "answer": "a"}]
        )
        assert False, "Should have raised exception"
    except Exception as e:
        assert "empty" in str(e).lower()


@test_wrapper
def test_ask_question_via_cli_no_cli():
    """Test error when CLI not initialized"""
    # Clear CLI from RuntimeConfig
    RuntimeConfig.cli_interface = None
    
    # Setup mock LLM client
    mock_client = Mock()
    
    # Create tool
    tool = QuestioningAgentTool(mock_client)
    
    # Call should raise exception
    try:
        tool._ask_question_via_cli("q", 1, 1, 1)
        assert False, "Should have raised exception"
    except Exception as e:
        assert "CLI interface not initialized" in str(e)


# Run all tests
if __name__ == "__main__":
    print("Running QuestioningAgentTool unit tests...\n")
    
    test_handle_question_conversation_no_followup()
    test_handle_question_conversation_with_one_followup()
    test_handle_question_conversation_max_followup_reached()
    test_check_followup_needed_at_max_limit()
    test_check_followup_needed_returns_true()
    test_check_followup_needed_returns_false()
    test_check_followup_needed_missing_field()
    test_check_followup_needed_missing_followup_question()
    test_compress_conversation_success()
    test_compress_conversation_missing_compressed_field()
    test_compress_conversation_empty_compressed()
    test_ask_question_via_cli_no_cli()
    
    print("\nAll tests completed!")
# tests/unit/test_orchestrator_tool.py
"""
Unit tests for OrchestratorTool.

Tests configuration loading and system prompt retrieval.
"""

from unittest.mock import patch, Mock
from agentcore import test_wrapper
from agents.orchestrator.tool import OrchestratorTool
from config.runtime_config import RuntimeConfig


@test_wrapper
def test_orchestrator_tool_init_loads_config():
    """Test that OrchestratorTool initializes and loads config"""
    # Mock the config loader
    mock_config = {
        "max_followup_count": 2,
        "stage_names": [
            "input_output_skeleton",
            "execution_strategy_skeleton"
        ],
        "stage_prompts": {
            "input_output_skeleton": {
                "diagnostic": "Diagnostic prompt for stage 1",
                "questioning": "Questioning prompt for stage 1",
                "integration": "Integration prompt for stage 1"
            },
            "execution_strategy_skeleton": {
                "diagnostic": "Diagnostic prompt for stage 2",
                "questioning": "Questioning prompt for stage 2",
                "integration": "Integration prompt for stage 2"
            }
        }
    }
    
    with patch('agents.orchestrator.tool.load_config', return_value=mock_config):
        with patch('agents.orchestrator.tool.validate_config'):
            # Create tool
            tool = OrchestratorTool()
            
            # Verify config loaded
            assert tool.max_followup_count == 2
            assert len(tool.stage_names) == 2
            assert tool.stage_names[0] == "input_output_skeleton"
            
            # Verify RuntimeConfig updated
            assert RuntimeConfig.config_data == mock_config


@test_wrapper
def test_get_system_prompt_valid():
    """Test getting system prompt for valid stage and agent type"""
    mock_config = {
        "max_followup_count": 2,
        "stage_names": ["stage_1", "stage_2"],
        "stage_prompts": {
            "stage_1": {
                "diagnostic": "Stage 1 diagnostic prompt",
                "questioning": "Stage 1 questioning prompt",
                "integration": "Stage 1 integration prompt"
            },
            "stage_2": {
                "diagnostic": "Stage 2 diagnostic prompt",
                "questioning": "Stage 2 questioning prompt",
                "integration": "Stage 2 integration prompt"
            }
        }
    }
    
    with patch('agents.orchestrator.tool.load_config', return_value=mock_config):
        with patch('agents.orchestrator.tool.validate_config'):
            tool = OrchestratorTool()
            
            # Test getting different prompts
            prompt1 = tool.get_system_prompt(1, "diagnostic")
            assert prompt1 == "Stage 1 diagnostic prompt"
            
            prompt2 = tool.get_system_prompt(2, "questioning")
            assert prompt2 == "Stage 2 questioning prompt"
            
            prompt3 = tool.get_system_prompt(1, "integration")
            assert prompt3 == "Stage 1 integration prompt"


@test_wrapper
def test_get_system_prompt_invalid_stage_idx():
    """Test error when stage_idx is out of range"""
    mock_config = {
        "max_followup_count": 2,
        "stage_names": ["stage_1"],
        "stage_prompts": {
            "stage_1": {
                "diagnostic": "Prompt",
                "questioning": "Prompt",
                "integration": "Prompt"
            }
        }
    }
    
    with patch('agents.orchestrator.tool.load_config', return_value=mock_config):
        with patch('agents.orchestrator.tool.validate_config'):
            tool = OrchestratorTool()
            
            # Try invalid stage_idx
            try:
                tool.get_system_prompt(5, "diagnostic")
                assert False, "Should have raised exception"
            except Exception as e:
                assert "out of range" in str(e).lower()


@test_wrapper
def test_get_system_prompt_invalid_agent_type():
    """Test error when agent_type is invalid"""
    mock_config = {
        "max_followup_count": 2,
        "stage_names": ["stage_1"],
        "stage_prompts": {
            "stage_1": {
                "diagnostic": "Prompt",
                "questioning": "Prompt",
                "integration": "Prompt"
            }
        }
    }
    
    with patch('agents.orchestrator.tool.load_config', return_value=mock_config):
        with patch('agents.orchestrator.tool.validate_config'):
            tool = OrchestratorTool()
            
            # Try invalid agent_type
            try:
                tool.get_system_prompt(1, "invalid_type")
                assert False, "Should have raised exception"
            except Exception as e:
                assert "invalid" in str(e).lower()


@test_wrapper
def test_get_stage_name_valid():
    """Test getting stage name for valid index"""
    mock_config = {
        "max_followup_count": 2,
        "stage_names": ["input_output_skeleton", "execution_strategy_skeleton"],
        "stage_prompts": {}
    }
    
    with patch('agents.orchestrator.tool.load_config', return_value=mock_config):
        with patch('agents.orchestrator.tool.validate_config'):
            tool = OrchestratorTool()
            
            # Test getting stage names
            assert tool.get_stage_name(1) == "input_output_skeleton"
            assert tool.get_stage_name(2) == "execution_strategy_skeleton"


@test_wrapper
def test_get_stage_name_invalid():
    """Test error when stage_idx is invalid"""
    mock_config = {
        "max_followup_count": 2,
        "stage_names": ["stage_1"],
        "stage_prompts": {}
    }
    
    with patch('agents.orchestrator.tool.load_config', return_value=mock_config):
        with patch('agents.orchestrator.tool.validate_config'):
            tool = OrchestratorTool()
            
            # Try invalid index
            try:
                tool.get_stage_name(10)
                assert False, "Should have raised exception"
            except Exception as e:
                assert "out of range" in str(e).lower()


# Run all tests
if __name__ == "__main__":
    print("Running OrchestratorTool unit tests...\n")
    
    test_orchestrator_tool_init_loads_config()
    test_get_system_prompt_valid()
    test_get_system_prompt_invalid_stage_idx()
    test_get_system_prompt_invalid_agent_type()
    test_get_stage_name_valid()
    test_get_stage_name_invalid()
    
    print("\nAll tests completed!")
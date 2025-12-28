# tests/integration/test_orchestrator.py
"""
Integration tests for Orchestrator.

Tests the complete orchestration flow across stages.
"""

from unittest.mock import Mock, patch
from agentcore import test_wrapper
from agents.orchestrator import Orchestrator
from config.runtime_config import RuntimeConfig


@test_wrapper
def test_orchestrator_single_stage_flow():
    """Test orchestrator flow for a single stage with one question"""
    # Setup mock config
    mock_config = {
        "max_followup_count": 2,
        "stage_names": ["test_stage"],
        "stage_prompts": {
            "test_stage": {
                "diagnostic": "Diagnostic prompt",
                "questioning": "Questioning prompt",
                "integration": "Integration prompt"
            }
        }
    }
    
    # Setup mock CLI
    mock_cli = Mock()
    mock_cli.get_user_input.return_value = "Test answer"
    RuntimeConfig.cli_interface = mock_cli
    RuntimeConfig.config_data = mock_config
    
    # Setup mock LLM client
    mock_client = Mock()
    
    # Mock for DiagnosticAgent
    mock_client.invoke.side_effect = [
        # Diagnostic call - generate questions
        {"content": '{"questions": ["Test question?"]}', "tokens_in": 50, "tokens_out": 20},
        # Questioning followup check - no followup
        {"content": '{"need_followup": false}', "tokens_in": 40, "tokens_out": 15},
        # Integration call - improve prompt
        {"content": '{"improved_prompt": "Improved test prompt"}', "tokens_in": 60, "tokens_out": 30}
    ]
    
    # Patch config loading in OrchestratorTool
    with patch('agents.orchestrator.tool.load_config', return_value=mock_config):
        with patch('agents.orchestrator.tool.validate_config'):
            # Create orchestrator
            orchestrator = Orchestrator(mock_client, "Initial test prompt")
            compiled = orchestrator.compile()
            
            # Execute (will run one complete stage)
            result = compiled.invoke({})
            
            # Verify flow completed
            assert "current_prompt" in result
            assert result["current_prompt"] == "Improved test prompt"
            assert result["stage_idx"] == 2  # After update_stage, ready for next
            
            # Verify LLM was called for diagnostic, questioning check, and integration
            assert mock_client.invoke.call_count >= 2


@test_wrapper
def test_orchestrator_init_stage():
    """Test init_stage node behavior"""
    mock_config = {
        "max_followup_count": 2,
        "stage_names": ["stage_1"],
        "stage_prompts": {"stage_1": {"diagnostic": "p", "questioning": "p", "integration": "p"}}
    }
    
    RuntimeConfig.config_data = mock_config
    
    mock_client = Mock()
    
    with patch('agents.orchestrator.tool.load_config', return_value=mock_config):
        with patch('agents.orchestrator.tool.validate_config'):
            orchestrator = Orchestrator(mock_client, "Test initial prompt")
            
            # Test init_stage behavior
            state = {}
            result = orchestrator.init_stage(state)
            
            # Verify initialization
            assert result["current_prompt"] == "Test initial prompt"
            assert result["stage_idx"] == 1
            assert result["question_list"] == []
            assert result["answer_list"] == []
            assert result["dialogue_idx"] == 0
            assert result["followup_count"] == 0


@test_wrapper
def test_orchestrator_route_after_diagnostic():
    """Test routing logic after diagnostic"""
    mock_config = {
        "max_followup_count": 2,
        "stage_names": ["stage_1"],
        "stage_prompts": {"stage_1": {"diagnostic": "p", "questioning": "p", "integration": "p"}}
    }
    
    RuntimeConfig.config_data = mock_config
    mock_client = Mock()
    
    with patch('agents.orchestrator.tool.load_config', return_value=mock_config):
        with patch('agents.orchestrator.tool.validate_config'):
            orchestrator = Orchestrator(mock_client, "Test")
            
            state1 = {"question_list": []}
            route1 = orchestrator.route_after_diagnostic(state1)
            assert route1 == "update_stage"
            
            state2 = {"question_list": ["Q1"]}
            route2 = orchestrator.route_after_diagnostic(state2)
            assert route2 == "call_questioning"


@test_wrapper
def test_orchestrator_route_after_questioning():
    """Test routing logic after questioning"""
    mock_config = {
        "max_followup_count": 2,
        "stage_names": ["stage_1"],
        "stage_prompts": {"stage_1": {"diagnostic": "p", "questioning": "p", "integration": "p"}}
    }
    
    RuntimeConfig.config_data = mock_config
    mock_client = Mock()
    
    with patch('agents.orchestrator.tool.load_config', return_value=mock_config):
        with patch('agents.orchestrator.tool.validate_config'):
            orchestrator = Orchestrator(mock_client, "Test")
            
            # Test routing with more questions remaining
            state1 = {
                "dialogue_idx": 1,
                "question_list": ["Q1", "Q2", "Q3"]
            }
            route1 = orchestrator.route_after_questioning(state1)
            assert route1 == "call_questioning"
            
            # Test routing with all questions answered
            state2 = {
                "dialogue_idx": 3,
                "question_list": ["Q1", "Q2", "Q3"]
            }
            route2 = orchestrator.route_after_questioning(state2)
            assert route2 == "call_integration"


@test_wrapper
def test_orchestrator_route_after_integration():
    """Test routing logic after integration"""
    mock_config = {
        "max_followup_count": 2,
        "stage_names": ["stage_1", "stage_2", "stage_3"],
        "stage_prompts": {
            "stage_1": {"diagnostic": "p", "questioning": "p", "integration": "p"},
            "stage_2": {"diagnostic": "p", "questioning": "p", "integration": "p"},
            "stage_3": {"diagnostic": "p", "questioning": "p", "integration": "p"}
        }
    }
    
    RuntimeConfig.config_data = mock_config
    mock_client = Mock()
    
    with patch('agents.orchestrator.tool.load_config', return_value=mock_config):
        with patch('agents.orchestrator.tool.validate_config'):
            orchestrator = Orchestrator(mock_client, "Test")
            
            # Test routing with more stages remaining
            state1 = {"stage_idx": 2}
            route1 = orchestrator.route_after_integration(state1)
            assert route1 == "init_stage"
            
            # Test routing after final stage (stage_idx already incremented)
            state2 = {"stage_idx": 4}
            route2 = orchestrator.route_after_integration(state2)
            from langgraph.graph import END
            assert route2 == END
            
@test_wrapper
def test_orchestrator_update_stage():
    """Test update_stage node behavior"""
    mock_config = {
        "max_followup_count": 2,
        "stage_names": ["stage_1"],
        "stage_prompts": {"stage_1": {"diagnostic": "p", "questioning": "p", "integration": "p"}}
    }
    
    RuntimeConfig.config_data = mock_config
    mock_client = Mock()
    
    with patch('agents.orchestrator.tool.load_config', return_value=mock_config):
        with patch('agents.orchestrator.tool.validate_config'):
            orchestrator = Orchestrator(mock_client, "Test")
            
            # Test stage update
            state = {"stage_idx": 1}
            result = orchestrator.update_stage(state)
            
            assert result["stage_idx"] == 2


@test_wrapper
def test_orchestrator_multiple_questions_per_stage():
    """Test handling multiple questions in a single stage"""
    mock_config = {
        "max_followup_count": 2,
        "stage_names": ["test_stage"],
        "stage_prompts": {
            "test_stage": {
                "diagnostic": "Diagnostic prompt",
                "questioning": "Questioning prompt",
                "integration": "Integration prompt"
            }
        }
    }
    
    # Setup mock CLI to answer multiple questions
    mock_cli = Mock()
    mock_cli.get_user_input.side_effect = ["Answer 1", "Answer 2", "Answer 3"]
    RuntimeConfig.cli_interface = mock_cli
    RuntimeConfig.config_data = mock_config
    
    # Setup mock LLM client
    mock_client = Mock()

    def mock_invoke(*args, **kwargs):
        user_prompt = kwargs.get('user_prompt', '')
        
        # Diagnostic call
        if "analyze" in user_prompt.lower():
            return {"content": '{"questions": ["Q1?", "Q2?", "Q3?"]}', "tokens_in": 50, "tokens_out": 30}
        
        # Followup check
        elif "need_followup" in user_prompt.lower() or "Original question" in user_prompt:
            return {"content": '{"need_followup": false}', "tokens_in": 40, "tokens_out": 15}
        
        # Integration call
        else:
            return {"content": '{"improved_prompt": "Final improved prompt"}', "tokens_in": 70, "tokens_out": 35}

    mock_client.invoke = mock_invoke
    with patch('agents.orchestrator.tool.load_config', return_value=mock_config):
        with patch('agents.orchestrator.tool.validate_config'):
            orchestrator = Orchestrator(mock_client, "Initial prompt")
            compiled = orchestrator.compile()
            
            result = compiled.invoke({})
            
            # Verify all questions were asked
            assert mock_cli.get_user_input.call_count == 3
            
            # Verify final prompt updated
            assert result["current_prompt"] == "Final improved prompt"


# Run all tests
if __name__ == "__main__":
    print("Running Orchestrator integration tests...\n")
    
    test_orchestrator_single_stage_flow()
    test_orchestrator_init_stage()
    test_orchestrator_route_after_diagnostic()
    test_orchestrator_route_after_questioning()
    test_orchestrator_route_after_integration()
    test_orchestrator_update_stage()
    test_orchestrator_multiple_questions_per_stage()
    
    print("\nAll tests completed!")

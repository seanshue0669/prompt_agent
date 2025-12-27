# agents/diagnostic_agent/schema.py
from typing import TypedDict, List
from agentcore import BaseSchema 

# ========================================================
# State definition
# ========================================================
class DiagnosticAgentState(TypedDict):
    """
    State for DiagnosticAgent.
    
    Input fields (from Orchestrator):
        system_prompt: System prompt for this diagnostic stage
        current_prompt: The prompt to be diagnosed
        
    Output fields (to Orchestrator):
        question_list: List of diagnostic questions to ask user
    """
    system_prompt: str
    current_prompt: str
    question_list: List[str]

# ========================================================
# Node definition
# ========================================================
def diagnose(state: DiagnosticAgentState) -> dict:
    """
    Analyze the current prompt and generate diagnostic questions.
    Implementation in controller.
    """
    return state

# ========================================================
# Schema Definition
# ========================================================
class DiagnosticAgentSchema(BaseSchema):
    state_type = DiagnosticAgentState

    # State mapping for subgraph invocation from Orchestrator
    state_mapping = {
        "diagnostic": {
            "input": {
                "system_prompt": "system_prompt",
                "current_prompt": "current_prompt"
            },
            "output": {
                "question_list": "question_list"
            }
        }
    }

    nodes = [
        ("diagnose", diagnose)
    ]
    
    conditional_edges = []
    
    direct_edges = []
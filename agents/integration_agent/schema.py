# agents/integration_agent/schema.py
from typing import TypedDict, List
from agentcore import BaseSchema 

# ========================================================
# State definition
# ========================================================
class IntegrationAgentState(TypedDict):
    """
    State for IntegrationAgent.
    
    Input fields (from Orchestrator):
        system_prompt: System prompt for integration
        current_prompt: Current version of the prompt
        answer_list: All user answers from this stage
        
    Output fields (to Orchestrator):
        current_prompt: Updated/improved prompt (overwrites input)
    """
    system_prompt: str
    current_prompt: str
    answer_list: List[str]

# ========================================================
# Node definition
# ========================================================
def integrate(state: IntegrationAgentState) -> dict:
    """
    Integrate user answers into the prompt to improve it.
    Implementation in controller.
    """
    return state

# ========================================================
# Schema Definition
# ========================================================
class IntegrationAgentSchema(BaseSchema):
    state_type = IntegrationAgentState

    # State mapping for subgraph invocation from Orchestrator
    state_mapping = {
        "integration": {
            "input": {
                "system_prompt": "system_prompt",
                "current_prompt": "current_prompt",
                "answer_list": "answer_list"
            },
            "output": {
                "current_prompt": "current_prompt"
            }
        }
    }

    nodes = [
        ("integrate", integrate)
    ]
    
    conditional_edges = []
    
    direct_edges = []
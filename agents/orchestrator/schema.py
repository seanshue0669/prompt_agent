# agents/orchestrator/schema.py
from typing import TypedDict, List
from agentcore import BaseSchema 

# ========================================================
# State definition
# ========================================================
class OrchestratorState(TypedDict):
    info : str

# ========================================================
# Node definition
# ========================================================
def passthrough(state: OrchestratorState) -> dict:
    """Placeholder node"""
    return state

# ========================================================
# Edge definition
# ========================================================

def conditional(state: OrchestratorState) -> str:
    return state

# ========================================================
# Schema Definition
# ========================================================
class OrchestratorSchema(BaseSchema):
    state_type = OrchestratorState

    state_mapping = {

    }

    nodes = [

    ]
    
    conditional_edges = []
    
    direct_edges = [

    ]
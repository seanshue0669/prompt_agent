# agents/integration_agent/schema.py
from typing import TypedDict, List
from agentcore import BaseSchema 

# ========================================================
# State definition
# ========================================================
class IntegrationAgentState(TypedDict):
    info : str

# ========================================================
# Node definition
# ========================================================
def passthrough(state: IntegrationAgentState) -> dict:
    """Placeholder node"""
    return state

# ========================================================
# Edge definition
# ========================================================

def conditional(state: IntegrationAgentState) -> str:
    return state

# ========================================================
# Schema Definition
# ========================================================
class IntegrationAgentSchema(BaseSchema):
    state_type = IntegrationAgentState

    state_mapping = {

    }

    nodes = [

    ]
    
    conditional_edges = []
    
    direct_edges = [

    ]
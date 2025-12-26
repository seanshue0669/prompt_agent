# agents/diagnostic_agent/schema.py
from typing import TypedDict, List
from agentcore import BaseSchema 

# ========================================================
# State definition
# ========================================================
class DiagnosticAgentState(TypedDict):
    info : str

# ========================================================
# Node definition
# ========================================================
def passthrough(state: DiagnosticAgentState) -> dict:
    """Placeholder node"""
    return state

# ========================================================
# Edge definition
# ========================================================

def conditional(state: DiagnosticAgentState) -> str:
    return state

# ========================================================
# Schema Definition
# ========================================================
class DiagnosticAgentSchema(BaseSchema):
    state_type = DiagnosticAgentState

    state_mapping = {

    }

    nodes = [

    ]
    
    conditional_edges = []
    
    direct_edges = [

    ]
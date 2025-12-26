# agents/questioning_agent/schema.py
from typing import TypedDict, List
from agentcore import BaseSchema 

# ========================================================
# State definition
# ========================================================
class QuestioningAgentState(TypedDict):
    info : str

# ========================================================
# Node definition
# ========================================================
def passthrough(state: QuestioningAgentState) -> dict:
    """Placeholder node"""
    return state

# ========================================================
# Edge definition
# ========================================================

def conditional(state: QuestioningAgentState) -> str:
    return state

# ========================================================
# Schema Definition
# ========================================================
class QuestioningAgentSchema(BaseSchema):
    state_type = QuestioningAgentState

    state_mapping = {

    }

    nodes = [

    ]
    
    conditional_edges = []
    
    direct_edges = [

    ]
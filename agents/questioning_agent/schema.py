# agents/questioning_agent/schema.py
from typing import TypedDict, List
from agentcore import BaseSchema 

# ========================================================
# State definition
# ========================================================
class QuestioningAgentState(TypedDict):
    """
    State for QuestioningAgent.
    
    Input fields (from Orchestrator):
        system_prompt: System prompt for questioning
        question_list: List of questions to ask
        dialogue_idx: Current question index
        answer_list: Existing answers (may be empty initially)
        followup_count: Current followup count for this question
        
    Output fields (to Orchestrator):
        answer_list: Updated with new answers (appended)
        followup_count: Updated followup count
    """
    system_prompt_followup: str  
    system_prompt_compress: str
    question_list: List[str]
    dialogue_idx: int
    answer_list: List[str]
    followup_count: int

# ========================================================
# Node definition
# ========================================================
def ask_question(state: QuestioningAgentState) -> dict:
    """
    Ask user the current question and collect answer.
    May generate followup questions based on answer quality.
    Implementation in controller.
    """
    return state

def check_followup(state: QuestioningAgentState) -> str:
    """
    Decide whether to ask followup question or move to next question.
    Returns "followup" or "next_question".
    Implementation in controller.
    """
    return "next_question"

# ========================================================
# Schema Definition
# ========================================================
class QuestioningAgentSchema(BaseSchema):
    state_type = QuestioningAgentState

    # State mapping for subgraph invocation from Orchestrator
    state_mapping = {
        "questioning": {
            "input": {
                "system_prompt_followup": "system_prompt_followup",  # Changed
                "system_prompt_compress": "system_prompt_compress",  # New
                "question_list": "question_list",
                "dialogue_idx": "dialogue_idx",
                "answer_list": "answer_list",
                "followup_count": "followup_count"
            },
            "output": {
                "answer_list": "answer_list",
                "followup_count": "followup_count"
            }
        }
    }

    nodes = [
        ("ask_question", ask_question)
    ]
    
    conditional_edges = []
    
    direct_edges = []
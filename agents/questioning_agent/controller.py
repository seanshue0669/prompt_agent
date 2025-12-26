# agents/questioning_agent/controller.py
import json

from agentcore import LLMClient, BaseGraph

from agents.questioning_agent.schema import QuestioningAgentSchema
from agents.questioning_agent.tool import QuestioningAgentTool


class QuestioningAgent(BaseGraph):
    """Agent for questioning agent."""

    def __init__(self, llm_client: LLMClient):
        super().__init__(QuestioningAgentSchema.state_type)

        # --- import schema definitions ---
        self.nodes = QuestioningAgentSchema.nodes
        self.conditional_edges = QuestioningAgentSchema.conditional_edges
        self.direct_edges = QuestioningAgentSchema.direct_edges

        # --- load dependent graphs ---
        DEPENDENT_GRAPHS = {}
        self.subgraphs = {k: v() for k, v in DEPENDENT_GRAPHS.items()}
        self.state_mapping = QuestioningAgentSchema.state_mapping
        
        # --- load tools ---
        self.tools = QuestioningAgentTool(llm_client)
    
    def passthrough(self, state: dict) -> dict:
        """Implement the node here"""
        return state
    
    def compile(self):
        """Compile the QuestioningAgent graph using BaseGraph logic."""
        return super().compile()
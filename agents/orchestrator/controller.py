# agents/orchestrator/controller.py
import json

from agentcore import LLMClient, BaseGraph

from agents.orchestrator.schema import OrchestratorSchema
from agents.orchestrator.tool import OrchestratorTool


class Orchestrator(BaseGraph):
    """Agent for orchestrator."""

    def __init__(self, llm_client: LLMClient):
        super().__init__(OrchestratorSchema.state_type)

        # --- import schema definitions ---
        self.nodes = OrchestratorSchema.nodes
        self.conditional_edges = OrchestratorSchema.conditional_edges
        self.direct_edges = OrchestratorSchema.direct_edges

        # --- load dependent graphs ---
        DEPENDENT_GRAPHS = {}
        self.subgraphs = {k: v() for k, v in DEPENDENT_GRAPHS.items()}
        self.state_mapping = OrchestratorSchema.state_mapping
        
        # --- load tools ---
        self.tools = OrchestratorTool(llm_client)
    
    def passthrough(self, state: dict) -> dict:
        """Implement the node here"""
        return state
    
    def compile(self):
        """Compile the Orchestrator graph using BaseGraph logic."""
        return super().compile()
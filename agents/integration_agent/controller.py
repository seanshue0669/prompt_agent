# agents/integration_agent/controller.py
import json

from agentcore import LLMClient, BaseGraph

from agents.integration_agent.schema import IntegrationAgentSchema
from agents.integration_agent.tool import IntegrationAgentTool


class IntegrationAgent(BaseGraph):
    """Agent for integration agent."""

    def __init__(self, llm_client: LLMClient):
        super().__init__(IntegrationAgentSchema.state_type)

        # --- import schema definitions ---
        self.nodes = IntegrationAgentSchema.nodes
        self.conditional_edges = IntegrationAgentSchema.conditional_edges
        self.direct_edges = IntegrationAgentSchema.direct_edges

        # --- load dependent graphs ---
        DEPENDENT_GRAPHS = {}
        self.subgraphs = {k: v() for k, v in DEPENDENT_GRAPHS.items()}
        self.state_mapping = IntegrationAgentSchema.state_mapping
        
        # --- load tools ---
        self.tools = IntegrationAgentTool(llm_client)
    
    def passthrough(self, state: dict) -> dict:
        """Implement the node here"""
        return state
    
    def compile(self):
        """Compile the IntegrationAgent graph using BaseGraph logic."""
        return super().compile()
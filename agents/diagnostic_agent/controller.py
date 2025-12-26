# agents/diagnostic_agent/controller.py
import json

from agentcore import LLMClient, BaseGraph

from agents.diagnostic_agent.schema import DiagnosticAgentSchema
from agents.diagnostic_agent.tool import DiagnosticAgentTool


class DiagnosticAgent(BaseGraph):
    """Agent for diagnostic agent."""

    def __init__(self, llm_client: LLMClient):
        super().__init__(DiagnosticAgentSchema.state_type)

        # --- import schema definitions ---
        self.nodes = DiagnosticAgentSchema.nodes
        self.conditional_edges = DiagnosticAgentSchema.conditional_edges
        self.direct_edges = DiagnosticAgentSchema.direct_edges

        # --- load dependent graphs ---
        DEPENDENT_GRAPHS = {}
        self.subgraphs = {k: v() for k, v in DEPENDENT_GRAPHS.items()}
        self.state_mapping = DiagnosticAgentSchema.state_mapping
        
        # --- load tools ---
        self.tools = DiagnosticAgentTool(llm_client)
    
    def passthrough(self, state: dict) -> dict:
        """Implement the node here"""
        return state
    
    def compile(self):
        """Compile the DiagnosticAgent graph using BaseGraph logic."""
        return super().compile()
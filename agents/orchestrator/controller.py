# agents/orchestrator/controller.py
import json

from agentcore import LLMClient, BaseGraph

from agents.orchestrator.schema import OrchestratorSchema
from agents.orchestrator.tool import OrchestratorTool

# dependent controller
from agents.diagnostic_agent.controller import DiagnosticAgent
from agents.questioning_agent.controller import QuestioningAgent
from agents.integration_agent.controller import IntegrationAgent

# dependent schema
from agents.diagnostic_agent.schema import DiagnosticAgentSchema
from agents.questioning_agent.schema import QuestioningAgentSchema
from agents.integration_agent.schema import IntegrationAgentSchema

class Orchestrator(BaseGraph):
    """Agent for orchestrator."""

    def __init__(self, llm_client: LLMClient):
        super().__init__(OrchestratorSchema.state_type)

        # --- import schema definitions ---
        self.nodes = OrchestratorSchema.nodes
        self.conditional_edges = OrchestratorSchema.conditional_edges
        self.direct_edges = OrchestratorSchema.direct_edges

        # --- load dependent graphs and schemas ---
        DEPENDENT_GRAPHS_AND_SCHEMA = {
            "diagnostic_agent": {
                "controller": DiagnosticAgent,
                "schema": DiagnosticAgentSchema
            },
            "questioning_agent": {
                "controller": QuestioningAgent,
                "schema": QuestioningAgentSchema
            },
            "integration_agent": {
                "controller": IntegrationAgent,
                "schema": IntegrationAgentSchema
            },
        }

        self.subgraphs = {}
        for k, v in DEPENDENT_GRAPHS_AND_SCHEMA.items():
            subgraph_instance = v["controller"](llm_client)
            self.subgraphs[k] = self.register_subgraph(
                subgraph_instance.compile(),
                v["schema"].state_mapping
            )
        
        # --- load tools ---
        self.tools = OrchestratorTool(llm_client)
    
    def passthrough(self, state: dict) -> dict:
        """Implement the node here"""
        return state
    
    def compile(self):
        """Compile the Orchestrator graph using BaseGraph logic."""
        return super().compile()
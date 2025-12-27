# agents/integration_agent/controller.py
from agentcore import LLMClient, BaseGraph
from agents.integration_agent.schema import IntegrationAgentSchema
from agents.integration_agent.tool import IntegrationAgentTool


class IntegrationAgent(BaseGraph):
    """
    Agent for integrating user answers into the prompt to improve it.
    
    This agent receives user answers from a diagnostic stage,
    analyzes them, and integrates the insights into the prompt
    to make it more complete and precise.
    """

    def __init__(self, llm_client: LLMClient):
        super().__init__(IntegrationAgentSchema.state_type)

        # Import schema definitions
        self.nodes = IntegrationAgentSchema.nodes
        self.conditional_edges = IntegrationAgentSchema.conditional_edges
        self.direct_edges = IntegrationAgentSchema.direct_edges
        self.state_mapping = IntegrationAgentSchema.state_mapping
        
        # Initialize tool
        self.tool = IntegrationAgentTool(llm_client)
    
    def integrate(self, state: dict) -> dict:
        """
        Integrate user answers into the prompt to improve it.
        
        Args:
            state: IntegrationAgentState containing:
                - system_prompt: Integration strategy
                - current_prompt: Current prompt version
                - answer_list: User answers to integrate
                
        Returns:
            Updated state with improved current_prompt
        """
        system_prompt = state["system_prompt"]
        current_prompt = state["current_prompt"]
        answer_list = state["answer_list"]
        
        # Use tool to generate improved prompt
        improved_prompt = self.tool.integrate_answers(
            system_prompt, 
            current_prompt, 
            answer_list
        )
        
        # Update state with improved prompt
        state["current_prompt"] = improved_prompt
        
        return state
    
    def compile(self):
        """Compile the IntegrationAgent graph using BaseGraph logic."""
        return super().compile()
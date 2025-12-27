# agents/diagnostic_agent/controller.py
from agentcore import LLMClient, BaseGraph
from agents.diagnostic_agent.schema import DiagnosticAgentSchema
from agents.diagnostic_agent.tool import DiagnosticAgentTool


class DiagnosticAgent(BaseGraph):
    """
    Agent for analyzing prompts and generating diagnostic questions.
    
    This agent receives a prompt and diagnostic criteria (via system_prompt),
    analyzes the prompt, and generates a list of questions to ask the user.
    """

    def __init__(self, llm_client: LLMClient):
        super().__init__(DiagnosticAgentSchema.state_type)

        # Import schema definitions
        self.nodes = DiagnosticAgentSchema.nodes
        self.conditional_edges = DiagnosticAgentSchema.conditional_edges
        self.direct_edges = DiagnosticAgentSchema.direct_edges
        self.state_mapping = DiagnosticAgentSchema.state_mapping
        
        # Initialize tool
        self.tool = DiagnosticAgentTool(llm_client)
    
    def diagnose(self, state: dict) -> dict:
        """
        Analyze the current prompt and generate diagnostic questions.
        
        Args:
            state: DiagnosticAgentState containing:
                - system_prompt: Diagnostic criteria
                - current_prompt: Prompt to analyze
                
        Returns:
            Updated state with question_list populated
        """
        system_prompt = state["system_prompt"]
        current_prompt = state["current_prompt"]
        
        # Use tool to generate questions
        question_list = self.tool.diagnose_prompt(system_prompt, current_prompt)
        
        # Update state
        state["question_list"] = question_list
        
        return state
    
    def compile(self):
        """Compile the DiagnosticAgent graph using BaseGraph logic."""
        return super().compile()
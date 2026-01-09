# agents/orchestrator/controller.py
from agentcore import LLMClient, BaseGraph
from agents.orchestrator.schema import OrchestratorSchema
from agents.orchestrator.tool import OrchestratorTool
from config.runtime_config import RuntimeConfig
from langgraph.graph import END

# Import dependent agents
from agents.diagnostic_agent.controller import DiagnosticAgent
from agents.questioning_agent.controller import QuestioningAgent
from agents.integration_agent.controller import IntegrationAgent

# Import dependent schemas for state_mapping
from agents.diagnostic_agent.schema import DiagnosticAgentSchema
from agents.questioning_agent.schema import QuestioningAgentSchema
from agents.integration_agent.schema import IntegrationAgentSchema


class Orchestrator(BaseGraph):
    """
    Top-level orchestrator for the prompt optimization system.
    
    Manages the complete workflow across 6 diagnostic stages:
    1. Input/Output Skeleton
    2. Execution Strategy Skeleton
    3. Input/Output Disambiguation
    4. Execution Strategy Disambiguation
    5. Execution Strategy Robustness
    6. Input/Output Robustness
    
    Each stage follows: Diagnostic -> Questioning -> Integration
    """

    def __init__(self, llm_client: LLMClient, initial_prompt: str):
        super().__init__(OrchestratorSchema.state_type)

        # Import schema definitions
        self.nodes = OrchestratorSchema.nodes
        self.conditional_edges = OrchestratorSchema.conditional_edges
        self.direct_edges = OrchestratorSchema.direct_edges
        
        # Initialize tool (loads config)
        self.tool = OrchestratorTool()

        # Override conditional edges so routing uses controller logic.
        self.conditional_edges = [
            ("call_diagnostic", self.route_after_diagnostic, {
                "call_questioning": "call_questioning",
                "update_stage": "update_stage"
            }),
            ("increment_dialogue_idx", self.route_after_questioning, {
                "call_questioning": "call_questioning",
                "call_integration": "call_integration"
            }),
            ("update_stage", self.route_after_integration, {
                "init_stage": "init_stage",
                END: END
            })
        ]
        # Store initial prompt
        self.initial_prompt = initial_prompt
        
        # Load and compile dependent agents
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
    
    # ========================================================
    # Node implementations
    # ========================================================
    
    def init_stage(self, state: dict) -> dict:
        """
        Initialize a new diagnostic stage.
        
        Resets answer_list, question_list, dialogue_idx, followup_count.
        Increments stage_idx if this is not the first stage.
        """
        updates = {
            "question_list": [],
            "answer_list": [],
            "dialogue_idx": 0,
            "followup_count": 0
        }
        
        # If this is the very first call, initialize everything
        if "current_prompt" not in state or not state["current_prompt"]:
            updates["current_prompt"] = self.initial_prompt
            updates["stage_idx"] = 1
        
        return updates
    
    def call_diagnostic(self, state: dict) -> dict:
        """
        Call DiagnosticAgent to generate questions for current stage.
        """
        stage_idx = state["stage_idx"]

        cli = RuntimeConfig.cli_interface
        if cli is not None:
            cli.update_stage(stage_idx=stage_idx, substage="診斷")
            cli.show_waiting_message("正在分析本階段問題...")
        
        # Inject diagnostic system prompt into state for subgraph
        state["system_prompt"] = self.tool.get_system_prompt(stage_idx, "diagnostic")
        
        # Call DiagnosticAgent subgraph
        result = self.subgraphs["diagnostic_agent"]("diagnostic", state)
        
        # Return only the fields that changed
        return {
            "question_list": result.get("question_list", [])
        }
    
    def call_questioning(self, state: dict) -> dict:
        """
        Call QuestioningAgent to ask current question and collect answer.
        Note: Does NOT increment dialogue_idx (handled by increment_dialogue_idx node)
        """
        stage_idx = state["stage_idx"]
        
        # Inject BOTH questioning system prompts into state for subgraph
        state["system_prompt_followup"] = self.tool.get_system_prompt(stage_idx, "questioning_followup")
        state["system_prompt_compress"] = self.tool.get_system_prompt(stage_idx, "questioning_compress")
        
        # Call QuestioningAgent subgraph
        result = self.subgraphs["questioning_agent"]("questioning", state)
        
        # Return only the fields that changed
        return {
            "answer_list": result.get("answer_list", []),
            "followup_count": result.get("followup_count", 0)
        }
    
    def increment_dialogue_idx(self, state: dict) -> dict:
        """
        Increment dialogue_idx after asking a question.
        This node separates the questioning action from the index update,
        allowing route_after_questioning to make correct routing decisions.
        """
        return {
            "dialogue_idx": state["dialogue_idx"] + 1,
            "followup_count": 0  # Reset followup count for next question
        }
    
    def call_integration(self, state: dict) -> dict:
        """
        Call IntegrationAgent to integrate answers into prompt.
        """
        stage_idx = state["stage_idx"]

        cli = RuntimeConfig.cli_interface
        if cli is not None:
            cli.update_stage(stage_idx=stage_idx, substage="統合")
            cli.show_waiting_message("正在進行最終統合...")
        
        # Inject integration system prompt into state for subgraph
        state["system_prompt"] = self.tool.get_system_prompt(stage_idx, "integration")
        
        # Call IntegrationAgent subgraph
        result = self.subgraphs["integration_agent"]("integration", state)
        
        # Return only the fields that changed
        return {
            "current_prompt": result.get("current_prompt", state["current_prompt"])
        }
    
    def update_stage(self, state: dict) -> dict:
        """
        Update stage_idx to move to next stage.
        """
        return {
            "stage_idx": state["stage_idx"] + 1
        }
    
    # ========================================================
    # Edge routing implementations
    # ========================================================
    
    def route_after_diagnostic(self, state: dict) -> str:
        """
        Route after DiagnosticAgent.
        
        Returns:
            "call_questioning": If questions exist
            "update_stage": If no questions exist
        """
        question_list = state["question_list"]
        
        if len(question_list) == 0:
            return "update_stage"
        else:
            return "call_questioning"


    def route_after_questioning(self, state: dict) -> str:
        """
        Route after incrementing dialogue_idx.
        
        Returns:
            "call_questioning": If more questions remain
            "call_integration": If all questions answered
        """
        dialogue_idx = state["dialogue_idx"]
        question_list = state["question_list"]
        
        if dialogue_idx < len(question_list):
            return "call_questioning"
        else:
            return "call_integration"
    
    def route_after_integration(self, state: dict) -> str:
        """
        Route after IntegrationAgent based on whether all stages completed.
        
        Returns:
            "init_stage": If more stages remain
            END: If all stages completed
        """
        stage_idx = state["stage_idx"]
        total_stages = len(self.tool.stage_names)
        
        if stage_idx <= total_stages:
            return "init_stage"
        else:
            return END
    
    # ========================================================
    # Compile
    # ========================================================
    
    def compile(self):
        """Compile the Orchestrator graph using BaseGraph logic."""
        return super().compile()

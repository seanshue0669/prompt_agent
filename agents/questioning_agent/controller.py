# agents/questioning_agent/controller.py
from agentcore import LLMClient, BaseGraph
from agents.questioning_agent.schema import QuestioningAgentSchema
from agents.questioning_agent.tool import QuestioningAgentTool
from config.runtime_config import RuntimeConfig


class QuestioningAgent(BaseGraph):
    """
    Agent for asking questions to users and collecting answers.
    
    This agent presents questions from the diagnostic phase,
    collects user answers via CLI, handles followup questions,
    and compresses the entire conversation into a concise Q&A.
    """

    def __init__(self, llm_client: LLMClient):
        super().__init__(QuestioningAgentSchema.state_type)

        # Import schema definitions
        self.nodes = QuestioningAgentSchema.nodes
        self.conditional_edges = QuestioningAgentSchema.conditional_edges
        self.direct_edges = QuestioningAgentSchema.direct_edges
        self.state_mapping = QuestioningAgentSchema.state_mapping
        
        # Initialize tool
        self.tool = QuestioningAgentTool(llm_client)
        
        # Get max_followup_count from config
        config = RuntimeConfig.config_data
        if config is None:
            raise Exception("Config not loaded in RuntimeConfig")
        self.max_followup_count = config.get("max_followup_count", 2)
    
    def ask_question(self, state: dict) -> dict:
        """
        Ask the current question and handle the entire conversation flow.
        
        This method delegates all logic to the tool, which handles:
        - Asking the original question
        - Managing followup conversation
        - Compressing the entire conversation
        
        Args:
            state: QuestioningAgentState containing:
                - system_prompt_followup: Prompt for followup logic
                - system_prompt_compress: Prompt for compression
                - question_list: All questions for this stage
                - dialogue_idx: Current question index
                - answer_list: Existing answers
                - followup_count: Current followup count (not used anymore, kept for compatibility)
                
        Returns:
            Updated state with compressed answer appended to answer_list
        """
        system_prompt_followup = state["system_prompt_followup"]
        system_prompt_compress = state["system_prompt_compress"]
        question_list = state["question_list"]
        dialogue_idx = state["dialogue_idx"]
        answer_list = state["answer_list"]
        
        # Get current question
        if dialogue_idx >= len(question_list):
            raise Exception(f"dialogue_idx {dialogue_idx} out of range (only {len(question_list)} questions)")
        
        current_question = question_list[dialogue_idx]
        
        # Determine stage_idx for CLI display (extract from orchestrator state if available)
        # For now, use a placeholder - will be properly set when called from Orchestrator
        stage_idx = state.get("stage_idx", 1)
        
        # Delegate entire conversation handling to tool
        compressed = self.tool.handle_question_conversation(
            system_prompt_followup=system_prompt_followup,
            system_prompt_compress=system_prompt_compress,
            question=current_question,
            stage_idx=stage_idx,
            question_idx=dialogue_idx + 1,
            total_questions=len(question_list),
            max_followup=self.max_followup_count
        )
        
        # Append compressed result to answer_list
        updated_answer_list = answer_list + [compressed]

        # Return only the fields that changed
        return {
            "answer_list": updated_answer_list,
            "followup_count": 0
        }
    
    def compile(self):
        """Compile the QuestioningAgent graph using BaseGraph logic."""
        return super().compile()
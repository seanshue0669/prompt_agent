# agents/questioning_agent/controller.py
from agentcore import LLMClient, BaseGraph
from agents.questioning_agent.schema import QuestioningAgentSchema
from agents.questioning_agent.tool import QuestioningAgentTool
from config.runtime_config import RuntimeConfig


class QuestioningAgent(BaseGraph):
    """
    Agent for asking questions to users and collecting answers.
    
    This agent presents questions from the diagnostic phase,
    collects user answers via CLI, and handles followup questions
    when answers need clarification.
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
        Ask the current question and collect user's answer.
        May generate and ask followup questions based on answer quality.
        
        Args:
            state: QuestioningAgentState containing:
                - system_prompt: Prompt for question handling
                - question_list: All questions for this stage
                - dialogue_idx: Current question index
                - answer_list: Existing answers
                - followup_count: Current followup count
                
        Returns:
            Updated state with new answer appended to answer_list
        """
        system_prompt = state["system_prompt"]
        question_list = state["question_list"]
        dialogue_idx = state["dialogue_idx"]
        answer_list = state["answer_list"]
        followup_count = state["followup_count"]
        
        # Get current question
        if dialogue_idx >= len(question_list):
            raise Exception(f"dialogue_idx {dialogue_idx} out of range (only {len(question_list)} questions)")
        
        current_question = question_list[dialogue_idx]
        
        # Determine stage_idx for CLI display (extract from system_prompt or use default)
        # For now, use a placeholder - will be set by Orchestrator
        stage_idx = 1  # This will be properly set when called from Orchestrator
        
        # Ask question and collect answer
        answer = self.tool.ask_question_and_collect(
            system_prompt=system_prompt,
            question=current_question,
            stage_idx=stage_idx,
            question_idx=dialogue_idx + 1,  # Display as 1-indexed
            total_questions=len(question_list)
        )
        
        # Append answer to list
        answer_list.append(answer)
        
        # Check if followup is needed
        followup_result = self.tool.should_followup(
            system_prompt=system_prompt,
            question=current_question,
            answer=answer,
            followup_count=followup_count,
            max_followup=self.max_followup_count
        )
        
        if followup_result["need_followup"]:
            # Generate followup question
            followup_question = followup_result["followup_question"]
            
            # Increment followup count
            state["followup_count"] = followup_count + 1
            
            # Ask followup question recursively
            followup_answer = self.tool.ask_question_and_collect(
                system_prompt=system_prompt,
                question=followup_question,
                stage_idx=stage_idx,
                question_idx=dialogue_idx + 1,
                total_questions=len(question_list)
            )
            
            # Append followup answer
            answer_list.append(followup_answer)
            
            # Check if another followup is needed (recursive)
            # For simplicity, limit to one level of followup per call
            # Orchestrator will call again if needed
        else:
            # Reset followup count for next question
            state["followup_count"] = 0
        
        # Update state
        state["answer_list"] = answer_list
        
        return state
    
    def compile(self):
        """Compile the QuestioningAgent graph using BaseGraph logic."""
        return super().compile()
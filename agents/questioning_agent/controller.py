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
        Compresses entire conversation at the end.
        
        Args:
            state: QuestioningAgentState containing:
                - system_prompt_followup: Prompt for followup logic
                - system_prompt_compress: Prompt for compression
                - question_list: All questions for this stage
                - dialogue_idx: Current question index
                - answer_list: Existing answers
                - followup_count: Current followup count
                
        Returns:
            Updated state with compressed answer appended to answer_list
        """
        system_prompt_followup = state["system_prompt_followup"]
        system_prompt_compress = state["system_prompt_compress"]
        question_list = state["question_list"]
        dialogue_idx = state["dialogue_idx"]
        answer_list = state["answer_list"]
        followup_count = state["followup_count"]
        
        # Get current question
        if dialogue_idx >= len(question_list):
            raise Exception(f"dialogue_idx {dialogue_idx} out of range (only {len(question_list)} questions)")
        
        current_question = question_list[dialogue_idx]
        
        # Determine stage_idx for CLI display
        stage_idx = 1  # Will be properly set when called from Orchestrator
        
        # Track entire conversation
        conversation_history = []
        
        # Ask original question
        answer = self.tool.ask_question_and_collect(
            system_prompt=system_prompt_followup,
            question=current_question,
            stage_idx=stage_idx,
            question_idx=dialogue_idx + 1,
            total_questions=len(question_list)
        )
        
        conversation_history.append({
            "question": current_question,
            "answer": answer
        })
        
        # Followup loop
        current_followup_count = followup_count
        
        while current_followup_count < self.max_followup_count:
            # Check if followup needed
            followup_result = self.tool.should_followup(
                system_prompt=system_prompt_followup,
                question=current_question,
                answer=answer,
                followup_count=current_followup_count,
                max_followup=self.max_followup_count
            )
            
            if not followup_result["need_followup"]:
                break
            
            # Generate and ask followup
            followup_question = followup_result["followup_question"]
            current_followup_count += 1
            
            followup_answer = self.tool.ask_question_and_collect(
                system_prompt=system_prompt_followup,
                question=followup_question,
                stage_idx=stage_idx,
                question_idx=dialogue_idx + 1,
                total_questions=len(question_list)
            )
            
            conversation_history.append({
                "question": followup_question,
                "answer": followup_answer
            })
            
            # Update answer for next iteration check
            answer = followup_answer
        
        # Compress entire conversation
        compressed = self.tool.compress_conversation(
            system_prompt=system_prompt_compress,
            original_question=current_question,
            conversation_history=conversation_history
        )
        
        # Append compressed result
        answer_list.append(compressed)
        
        # Update state
        state["answer_list"] = answer_list
        state["followup_count"] = current_followup_count
        
        return state
    
    def compile(self):
        """Compile the QuestioningAgent graph using BaseGraph logic."""
        return super().compile()
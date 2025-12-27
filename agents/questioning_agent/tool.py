# agents/questioning_agent/tool.py
import json
from typing import List
from agentcore import LLMClient, BaseTool, auto_wrap_error
from config.runtime_config import RuntimeConfig


class QuestioningAgentTool(BaseTool):
    """
    Tool for asking questions to users and collecting answers.
    Handles CLI interaction and followup question generation.
    """
    
    def __init__(self, client: LLMClient):
        super().__init__()
        self.client = client
    
    @auto_wrap_error
    def ask_question_and_collect(
        self,
        system_prompt: str,
        question: str,
        stage_idx: int,
        question_idx: int,
        total_questions: int
    ) -> str:
        """
        Ask a question to the user via CLI and collect the answer.
        
        Args:
            system_prompt: System prompt for question formatting
            question: The question to ask
            stage_idx: Current stage number (for CLI display)
            question_idx: Current question number (for CLI display)
            total_questions: Total number of questions (for CLI display)
            
        Returns:
            User's answer as string
        """
        # Get CLI interface from RuntimeConfig
        cli = RuntimeConfig.cli_interface
        if cli is None:
            raise Exception("CLI interface not initialized in RuntimeConfig")
        
        # Update CLI stage display
        cli.update_stage(
            stage_idx=stage_idx,
            substage="對話",
            question_idx=question_idx,
            total_questions=total_questions
        )
        
        # Format the question using LLM (optional enhancement)
        # For now, use the question directly
        formatted_question = question
        
        # Get user input via CLI
        answer = cli.get_user_input(formatted_question)
        
        return answer
    
    @auto_wrap_error
    def should_followup(
        self,
        system_prompt: str,
        question: str,
        answer: str,
        followup_count: int,
        max_followup: int
    ) -> dict:
        """
        Determine if a followup question is needed based on answer quality.
        
        Args:
            system_prompt: System prompt for followup criteria
            question: Original question
            answer: User's answer
            followup_count: Current followup count for this question
            max_followup: Maximum allowed followup count
            
        Returns:
            Dict with:
                - need_followup: bool (whether followup is needed)
                - followup_question: str or None (the followup question if needed)
        """
        # If already at max followup, no more followups
        if followup_count >= max_followup:
            return {
                "need_followup": False,
                "followup_question": None
            }
        
        # Construct user prompt for LLM to decide
        user_prompt = f"""Original question: {question}

User's answer: {answer}

Based on the followup criteria in the system prompt, determine if this answer is clear enough or if a followup question is needed."""
        
        # Configure for JSON output
        config_override = {
            "response_format": {"type": "json_object"}
        }
        
        # Call LLM
        response = self.client.invoke(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            config_override=config_override
        )
        
        # Parse JSON response
        raw = (response.get("content") or "").strip()
        if not raw:
            raise Exception("LLM returned empty content")
        
        try:
            result = json.loads(raw)
        except json.JSONDecodeError as e:
            raise Exception(f"Failed to parse LLM response as JSON: {e}")
        
        # Expected format: {"need_followup": true/false, "followup_question": "..."}
        if "need_followup" not in result:
            raise Exception("LLM response missing 'need_followup' field")
        
        need_followup = result["need_followup"]
        
        if not isinstance(need_followup, bool):
            raise Exception("'need_followup' field must be a boolean")
        
        # If followup needed, extract the question
        if need_followup:
            if "followup_question" not in result:
                raise Exception("LLM indicated followup needed but missing 'followup_question' field")
            
            followup_question = result["followup_question"]
            
            if not isinstance(followup_question, str):
                raise Exception("'followup_question' field must be a string")
            
            if not followup_question.strip():
                raise Exception("LLM generated empty followup question")
            
            return {
                "need_followup": True,
                "followup_question": followup_question
            }
        else:
            return {
                "need_followup": False,
                "followup_question": None
            }
    # Add new method for compression
@auto_wrap_error
def compress_conversation(
    self,
    system_prompt: str,
    original_question: str,
    conversation_history: List[dict]
) -> str:
    """
    Compress entire conversation into a concise Q&A pair.
    
    Args:
        system_prompt: System prompt for compression (with CoT)
        original_question: The original question asked
        conversation_history: List of {"question": str, "answer": str} dicts
        
    Returns:
        Compressed Q&A string like "Q: ... A: ..."
        
    Raises:
        Exception: If LLM call fails or returns invalid JSON
    """
    # Format conversation history
    formatted_history = ""
    for i, turn in enumerate(conversation_history, 1):
        formatted_history += f"{i}. Q: {turn['question']}\n"
        formatted_history += f"   A: {turn['answer']}\n\n"
    
    # Construct user prompt
    user_prompt = f"""Original question: {original_question}

Conversation history:
{formatted_history}

Based on the entire conversation, compress this into a single concise Q&A pair."""
    
    # Configure for JSON output with CoT
    config_override = {
        "response_format": {"type": "json_object"},
        "max_completion_tokens": 1000  # More tokens for CoT
    }
    
    # Call LLM
    response = self.client.invoke(
        user_prompt=user_prompt,
        system_prompt=system_prompt,
        config_override=config_override
    )
    
    # Parse JSON response
    raw = (response.get("content") or "").strip()
    if not raw:
        raise Exception("LLM returned empty content")
    
    try:
        result = json.loads(raw)
    except json.JSONDecodeError as e:
        raise Exception(f"Failed to parse LLM response as JSON: {e}")
    
    # Extract compressed Q&A
    # Expected format: {"compressed": "Q: ... A: ..."}
    if "compressed" not in result:
        raise Exception("LLM response missing 'compressed' field")
    
    compressed = result["compressed"]
    
    if not isinstance(compressed, str):
        raise Exception("'compressed' field must be a string")
    
    if not compressed.strip():
        raise Exception("LLM generated empty compressed Q&A")
    
    return compressed
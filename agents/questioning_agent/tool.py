# agents/questioning_agent/tool.py
import json
import re
from typing import List, Dict
from agentcore import LLMClient, BaseTool, auto_wrap_error
from config.runtime_config import RuntimeConfig


class QuestioningAgentTool(BaseTool):
    """
    Tool for managing question-answer conversations with users.
    
    Handles the complete conversation flow including:
    - Asking questions via CLI
    - Determining when followup is needed
    - Generating followup questions (with options if needed)
    - Compressing conversation history
    """
    
    def __init__(self, client: LLMClient):
        super().__init__()
        self.client = client

    @staticmethod
    def _sanitize_text(text: str) -> str:
        return "".join(ch for ch in text if not (0xD800 <= ord(ch) <= 0xDFFF))

    @staticmethod
    def _expand_option_answer(answer: str, options: List[str] | None) -> str:
        if not answer or not options:
            return answer

        raw = answer.strip()
        if not raw:
            return answer

        if not re.fullmatch(r"[A-Za-z](?:\s*[,，/、\s]+\s*[A-Za-z])*$", raw):
            return answer

        option_map = {}
        for opt in options:
            match = re.match(r"\s*([A-Za-z])\s*[\)\）]\s*(.*)", opt)
            if match:
                option_map[match.group(1).upper()] = match.group(2).strip()

        codes = [c.upper() for c in re.findall(r"[A-Za-z]", raw)]
        selected = []
        seen = set()
        for code in codes:
            text = option_map.get(code)
            if text and text not in seen:
                seen.add(text)
                selected.append(text)

        if not selected:
            return answer

        return f"{raw}（{'、'.join(selected)}）"
    
    @auto_wrap_error
    def handle_question_conversation(
        self,
        system_prompt_followup: str,
        system_prompt_compress: str,
        question: str,
        stage_idx: int,
        question_idx: int,
        total_questions: int,
        max_followup: int
    ) -> str:
        """
        Handle the entire conversation for a single question.
        
        This method:
        1. Asks the original question
        2. Manages followup conversation loop
        3. Compresses the entire conversation
        4. Returns the compressed Q&A result
        
        Args:
            system_prompt_followup: System prompt for followup decision/generation
            system_prompt_compress: System prompt for conversation compression
            question: The question to ask
            stage_idx: Current stage number (for CLI display)
            question_idx: Current question number (for CLI display)
            total_questions: Total number of questions (for CLI display)
            max_followup: Maximum number of followup questions allowed
            
        Returns:
            Compressed Q&A string in format "Q: ... A: ..."
        """
        # Track entire conversation
        conversation_history = []

        cli = RuntimeConfig.cli_interface
        if cli is not None:
            cli.clear_conversation()

        # Ask original question (open-ended, no options)
        answer = self._ask_question_via_cli(
            question=question,
            stage_idx=stage_idx,
            question_idx=question_idx,
            total_questions=total_questions,
            options=None  # First question has no options
        )
        
        conversation_history.append({
            "question": question,
            "answer": answer,
            "options": None
        })

        if cli is not None:
            cli.show_waiting_message()
        
        # Followup loop
        followup_count = 0
        while followup_count < max_followup:
            # Ask LLM if followup is needed
            followup_result = self._check_followup_needed(
                system_prompt=system_prompt_followup,
                original_question=question,
                conversation_history=conversation_history,
                followup_count=followup_count,
                max_followup=max_followup
            )
            
            if not followup_result["need_followup"]:
                break
            
            # Generate and ask followup question (with options)
            followup_question = followup_result["followup_question"]
            followup_options = followup_result.get("options", None)  # May have options
            
            followup_answer = self._ask_question_via_cli(
                question=followup_question,
                stage_idx=stage_idx,
                question_idx=question_idx,
                total_questions=total_questions,
                options=followup_options  # Pass options to CLI
            )
            followup_answer = self._expand_option_answer(
                followup_answer,
                followup_options
            )

            conversation_history.append({
                "question": followup_question,
                "answer": followup_answer,
                "options": followup_options
            })
            
            followup_count += 1
        
        # Compress entire conversation
        if cli is not None:
            cli.show_waiting_message("正在節錄摘要並進行道下一個部分")

        compressed = self._compress_conversation(
            system_prompt=system_prompt_compress,
            original_question=question,
            conversation_history=conversation_history
        )
        
        return compressed
    
    @auto_wrap_error
    def _ask_question_via_cli(
        self,
        question: str,
        stage_idx: int,
        question_idx: int,
        total_questions: int,
        options: List[str] = None
    ) -> str:
        """
        Ask a question to the user via CLI and collect the answer.
        
        Args:
            question: The question to ask
            stage_idx: Current stage number (for CLI display)
            question_idx: Current question number (for CLI display)
            total_questions: Total number of questions (for CLI display)
            options: Optional list of options to display (e.g., ["A) ...", "B) ...", ...])
            
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
        
        # Get user input via CLI (with options if provided)
        answer = cli.get_user_input(question, options=options)
        
        return answer
    
    @auto_wrap_error
    def _check_followup_needed(
        self,
        system_prompt: str,
        original_question: str,
        conversation_history: List[Dict[str, str]],
        followup_count: int,
        max_followup: int
    ) -> Dict[str, any]:
        """
        Use LLM to determine if a followup question is needed.
        
        Args:
            system_prompt: System prompt for followup criteria
            original_question: The original question asked
            conversation_history: List of {"question": str, "answer": str, "options": List[str] | None} dicts
            followup_count: Current followup count
            max_followup: Maximum allowed followup count
            
        Returns:
            Dict with:
                - need_followup: bool (whether followup is needed)
                - followup_question: str or None (the followup question if needed)
                - options: List[str] or None (list of options if this is a multiple choice followup)
        """
        # If already at max followup, no more followups
        if followup_count >= max_followup:
            return {
                "need_followup": False,
                "followup_question": None,
                "options": None
            }
        
        # Construct user prompt for LLM to decide
        formatted_history = f"Original question: {original_question}\n\n"
        for i, turn in enumerate(conversation_history, 1):
            formatted_history += f"Turn {i}:\n"
            formatted_history += f"Question: {turn['question']}\n"
            if turn.get("options"):
                formatted_history += "Options:\n"
                for opt in turn["options"]:
                    formatted_history += f"  {opt}\n"
            formatted_history += f"User's answer: {turn['answer']}\n\n"

        user_prompt = f"""<user_prompt>
        {formatted_history}Current followup count: {followup_count}
        Maximum followup allowed: {max_followup}

        Based on the complete conversation history and the followup criteria in the system prompt, determine if another followup question is needed.
        </user__prompt>"""

        user_prompt = self._sanitize_text(user_prompt)
        system_prompt = self._sanitize_text(system_prompt)
        
        # Configure for JSON output with strict schema
        config_override = {
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "followup_response",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "need_followup": {
                                "type": "boolean",
                                "description": "Whether a followup question is needed"
                            },
                            "followup_question": {
                                "type": "string",
                                "description": "The followup question text (required if need_followup is true)"
                            },
                            "options": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of options for multiple choice (required if need_followup is true)"
                            }
                        },
                        "required": ["need_followup"],
                        "additionalProperties": False
                    }
                }
            },
            "max_completion_tokens": 100000,
            "temperature": 0.7,
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
        
        # Validate response structure
        if "need_followup" not in result:
            raise Exception("LLM response missing 'need_followup' field")
        
        need_followup = result["need_followup"]
        
        if not isinstance(need_followup, bool):
            raise Exception("'need_followup' field must be a boolean")
        
        # If followup needed, extract the question and options
        if need_followup:
            if "followup_question" not in result:
                raise Exception("LLM indicated followup needed but missing 'followup_question' field")
            
            followup_question = result["followup_question"]
            
            if not isinstance(followup_question, str):
                raise Exception("'followup_question' field must be a string")
            
            if not followup_question.strip():
                raise Exception("LLM generated empty followup question")
            
            # options 是可選的（可能是選項式或開放式）
            options = result.get("options", None)
            
            # 如果有提供 options，才驗證格式
            if options is not None:
                if not isinstance(options, list):
                    raise Exception("'options' field must be a list")
                
                if len(options) == 0:
                    raise Exception("LLM generated empty options list")
                
                # Validate all options are strings
                for i, opt in enumerate(options):
                    if not isinstance(opt, str):
                        raise Exception(f"Option {i} must be a string")
            
            return {
                "need_followup": True,
                "followup_question": followup_question,
                "options": options  # 可能是 list 或 None
            }
        else:
            return {
                "need_followup": False,
                "followup_question": None,
                "options": None
            }
    
    @auto_wrap_error
    def _compress_conversation(
        self,
        system_prompt: str,
        original_question: str,
        conversation_history: List[Dict[str, str]]
    ) -> str:
        """
        Compress entire conversation into a concise Q&A pair using LLM.
        
        Args:
            system_prompt: System prompt for compression (with CoT)
            original_question: The original question asked
            conversation_history: List of {"question": str, "answer": str, "options": List[str] | None} dicts
            
        Returns:
            Compressed Q&A string like "Q: ... A: ..."
            
        Raises:
            Exception: If LLM call fails or returns invalid JSON
        """
        # Format conversation history
        formatted_history = ""
        for i, turn in enumerate(conversation_history, 1):
            formatted_history += f"{i}. Q: {turn['question']}\n"
            if turn.get("options"):
                formatted_history += f"   選項：{', '.join(turn['options'])}\n"
            formatted_history += f"   A: {turn['answer']}\n\n"
        
        # Construct user prompt
        user_prompt = f"""<user_prompt>
Original question: {original_question}

        Conversation history:
        {formatted_history}

        Based on the entire conversation, compress this into a single concise Q&A pair.
</user__prompt>"""

        user_prompt = self._sanitize_text(user_prompt)
        system_prompt = self._sanitize_text(system_prompt)
                
        # Configure for JSON output with CoT
        config_override = {
            "response_format": {"type": "json_object"},
            "reasoning_effort": "high",
            "max_completion_tokens": 10000  # More tokens for CoT
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
        
        # Expected format: {"思考過程": {...}, "compressed": "Q: ... A: ..."}
        if "compressed" not in result:
            raise Exception("LLM response missing 'compressed' field")
        
        compressed = result["compressed"]
        
        if not isinstance(compressed, str):
            raise Exception("'compressed' field must be a string")
        
        if not compressed.strip():
            raise Exception("LLM generated empty compressed Q&A")
        
        return compressed

# agents/integration_agent/tool.py
import json
from agentcore import LLMClient, BaseTool, auto_wrap_error


class IntegrationAgentTool(BaseTool):
    """
    Tool for integrating user answers into the prompt to improve it.
    """
    
    def __init__(self, client: LLMClient):
        super().__init__()
        self.client = client

    @staticmethod
    def _sanitize_text(text: str) -> str:
        return "".join(ch for ch in text if not (0xD800 <= ord(ch) <= 0xDFFF))
    
    @auto_wrap_error
    def integrate_answers(
        self, 
        system_prompt: str, 
        current_prompt: str, 
        answer_list: list[str]
    ) -> str:
        """
        Integrate user answers into the current prompt to improve it.
        
        Args:
            system_prompt: System prompt defining integration strategy
            current_prompt: Current version of the prompt
            answer_list: List of user answers from this diagnostic stage
            
        Returns:
            Improved prompt text
            
        Raises:
            Exception: If LLM call fails or returns invalid JSON
        """
        answer_list_payload = self._normalize_answer_list(answer_list)
        answer_list_json = json.dumps(answer_list_payload, ensure_ascii=False, indent=2)

        # Construct user prompt
        user_prompt = f"""<user_prompt>
        Current prompt:
        ```
        {current_prompt}
        ```

        Answer list (JSON):
        {answer_list_json}

        Use the answer list only as reference to update the prompt sections. Do not copy answer content or labels into the output prompt.
        Return only the JSON object requested by the system prompt.
        </user__prompt>"""

        user_prompt = self._sanitize_text(user_prompt)
        system_prompt = self._sanitize_text(system_prompt)
                
        # Configure for JSON output
        config_override = {
            "response_format": {"type": "json_object"},
            "reasoning_effort": "high",
            "max_completion_tokens": 100000 ,
            "temperature": 0.7
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
        
        # Extract improved prompt
        # Expected format: {"current_prompt": "..."} (fallback to "improved_prompt")
        if "current_prompt" in result:
            prompt_field = "current_prompt"
        elif "improved_prompt" in result:
            prompt_field = "improved_prompt"
        else:
            raise Exception("LLM response missing 'current_prompt' or 'improved_prompt' field")

        improved_prompt = result[prompt_field]
        
        if not isinstance(improved_prompt, str):
            raise Exception(f"'{prompt_field}' field must be a string")
        
        if not improved_prompt.strip():
            raise Exception(f"LLM generated empty {prompt_field}")
        
        return improved_prompt

    def _normalize_answer_list(self, answer_list: list[str]) -> list[dict]:
        normalized = []

        for item in answer_list:
            if isinstance(item, dict):
                question = str(item.get("question", "")).strip()
                answer = str(item.get("answer", "")).strip()
                normalized.append({"question": question, "answer": answer})
                continue

            if not isinstance(item, str):
                item = str(item)

            text = item.strip()
            if not text:
                normalized.append({"question": "", "answer": ""})
                continue

            question, answer = self._split_qa(text)
            normalized.append({"question": question, "answer": answer})

        return normalized

    def _split_qa(self, text: str) -> tuple[str, str]:
        q_marker = "Q:"
        a_marker = "A:"
        q_index = text.find(q_marker)
        if q_index != -1:
            a_index = text.find(a_marker, q_index + len(q_marker))
            if a_index != -1:
                question = text[q_index + len(q_marker):a_index].strip()
                answer = text[a_index + len(a_marker):].strip()
                return question, answer

        return "", text

# agents/diagnostic_agent/tool.py
import json
from agentcore import LLMClient, BaseTool, auto_wrap_error


class DiagnosticAgentTool(BaseTool):
    """
    Tool for analyzing prompts and generating diagnostic questions.
    """
    
    def __init__(self, client: LLMClient):
        super().__init__()
        self.client = client
    
    @auto_wrap_error
    def diagnose_prompt(self, system_prompt: str, current_prompt: str) -> list[str]:
        """
        Analyze the current prompt and generate diagnostic questions.
        
        Args:
            system_prompt: System prompt defining the diagnostic criteria
            current_prompt: The prompt to be analyzed
            
        Returns:
            List of diagnostic questions
            
        Raises:
            Exception: If LLM call fails or returns invalid JSON
        """
        # Construct user prompt with the current prompt to analyze
        user_prompt = f"""Please analyze the following prompt:
        ```
        {current_prompt}
        ```
        Based on the diagnostic criteria in the system prompt, generate questions to ask the user."""
        
        # Configure for JSON output
        config_override = {
                    "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "test_response",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "questions": {
                                "type": "array",
                                "items": {"type": "string"}
                            }
                        },
                        "required": ["questions"],
                        "additionalProperties": False
                    }
                }
            }
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
        
        # Extract question list
        # Expected format: {"questions": ["question1", "question2", ...]}
        if "questions" not in result:
            raise Exception("LLM response missing 'questions' field")
        
        questions = result["questions"]
        
        if not isinstance(questions, list):
            raise Exception("'questions' field must be a list")
        
        if len(questions) == 0:
            raise Exception("LLM generated empty question list")
        
        return questions
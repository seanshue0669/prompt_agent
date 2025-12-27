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
        # Format answers for the user prompt
        formatted_answers = "\n".join([
            f"{i+1}. {answer}" 
            for i, answer in enumerate(answer_list)
        ])
        
        # Construct user prompt
        user_prompt = f"""Current prompt:
        ```
        {current_prompt}
        ```

        User answers from diagnostic questions:
        {formatted_answers}

        Based on these answers, please improve the prompt according to the integration strategy in the system prompt."""
                
        # Configure for JSON output
        config_override = {
            "response_format": {"type": "json_object"},
            "reasoning_effort": "high",
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
        # Expected format: {"improved_prompt": "..."}
        if "current_prompt" not in result:
            raise Exception("LLM response missing 'improved_prompt' field")
        
        improved_prompt = result["current_prompt"]
        
        if not isinstance(improved_prompt, str):
            raise Exception("'improved_prompt' field must be a string")
        
        if not improved_prompt.strip():
            raise Exception("LLM generated empty current_prompt")
        
        return improved_prompt
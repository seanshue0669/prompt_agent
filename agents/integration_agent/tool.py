# agents/integration_agent/tool.py
import json
import os

from agentcore import LLMClient, BaseTool, auto_wrap_error

class IntegrationAgentTool(BaseTool):
    def __init__(self, client: LLMClient):
        super().__init__()
        self.client = client

    def _readjson(self, path: str) -> dict:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)      
    
    def _invoke_and_parse(self, user_prompt: str, system_prompt: str, cfg: dict) -> dict:
        response = self.client.invoke(user_prompt, system_prompt, cfg)
        raw = (response.get("content") or "").strip()
        if not raw:
            raise Exception("LLM returned empty content")
        return json.loads(raw)

    @auto_wrap_error
    def do_something(self, text: str) -> dict:
        return text
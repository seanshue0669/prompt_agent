# agents/orchestrator/tool.py
from agentcore import BaseTool, auto_wrap_error
from config.config_loader import load_config, validate_config
from config.runtime_config import RuntimeConfig


class OrchestratorTool(BaseTool):
    """
    Tool for Orchestrator to manage configuration and system prompts.
    """
    
    def __init__(self):
        super().__init__()
        
        # Load and validate config
        config = load_config("config/config.json")
        validate_config(config)
        
        # Store in RuntimeConfig for global access
        RuntimeConfig.config_data = config
        
        # Extract key config values
        self.stage_names = config["stage_names"]
        self.stage_prompts = config["stage_prompts"]
        self.max_followup_count = config["max_followup_count"]
    
    @auto_wrap_error
    def get_system_prompt(self, stage_idx: int, agent_type: str) -> str:
        """
        Get the appropriate system prompt for a given stage and agent type.
        
        Args:
            stage_idx: Current stage number (1-6)
            agent_type: Type of agent ("diagnostic", "questioning", "integration")
            
        Returns:
            System prompt text
            
        Raises:
            Exception: If stage_idx is out of range or agent_type invalid
        """
        # Validate stage_idx
        if stage_idx < 1 or stage_idx > len(self.stage_names):
            raise Exception(
                f"stage_idx {stage_idx} out of range (1-{len(self.stage_names)})"
            )
        
        # Get stage name (convert to 0-indexed)
        stage_name = self.stage_names[stage_idx - 1]
        
        # Validate agent_type
        valid_types = ["diagnostic", "questioning", "integration"]
        if agent_type not in valid_types:
            raise Exception(
                f"Invalid agent_type '{agent_type}'. Must be one of: {valid_types}"
            )
        
        # Get prompt
        if stage_name not in self.stage_prompts:
            raise Exception(f"Stage '{stage_name}' not found in stage_prompts")
        
        stage_prompt_set = self.stage_prompts[stage_name]
        
        if agent_type not in stage_prompt_set:
            raise Exception(
                f"Agent type '{agent_type}' not found for stage '{stage_name}'"
            )
        
        return stage_prompt_set[agent_type]
    
    @auto_wrap_error
    def get_stage_name(self, stage_idx: int) -> str:
        """
        Get the stage name for a given stage index.
        
        Args:
            stage_idx: Stage number (1-6)
            
        Returns:
            Stage name string
        """
        if stage_idx < 1 or stage_idx > len(self.stage_names):
            raise Exception(
                f"stage_idx {stage_idx} out of range (1-{len(self.stage_names)})"
            )
        
        return self.stage_names[stage_idx - 1]
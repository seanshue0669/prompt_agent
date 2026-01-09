# config/config_loader.py
"""
Configuration loader utility.
Loads config.json and resolves all prompt file paths.
"""

import json
import os
from typing import Dict, Any


def _get_config_root(config_path: str) -> str:
    config_dir = os.path.dirname(os.path.abspath(config_path))
    if os.path.basename(config_dir) == "json_config":
        return os.path.abspath(os.path.join(config_dir, os.pardir))
    return config_dir


def load_config(config_path: str = "config/json_config/config.json") -> Dict[str, Any]:
    """
    Load configuration from JSON file and resolve all prompt file paths.
    
    Args:
        config_path: Path to config.json file in config/json_config
        
    Returns:
        Dict containing:
            - max_followup_count: int
            - stage_names: List[str]
            - stage_prompts: Dict[str, Dict[str, str]] (with resolved prompt content)
            
    Raises:
        FileNotFoundError: If config file or prompt files not found
        json.JSONDecodeError: If config file is invalid JSON
    """
    # Resolve prompt paths relative to the config root (parent of json_config)
    config_root = _get_config_root(config_path)
    
    # Load config.json
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # Resolve all prompt file paths
    stage_prompts = config.get("stage_prompts", {})
    resolved_prompts = {}
    
    for stage_name, prompts in stage_prompts.items():
        resolved_prompts[stage_name] = {}
        
        for prompt_type, relative_path in prompts.items():
            # Construct full path relative to config root
            if os.path.isabs(relative_path):
                full_path = relative_path
            else:
                full_path = os.path.join(config_root, relative_path)
            
            # Read prompt file content
            try:
                with open(full_path, 'r', encoding='utf-8') as pf:
                    prompt_content = pf.read().strip()
                    resolved_prompts[stage_name][prompt_type] = prompt_content
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"Prompt file not found: {full_path} "
                    f"(stage: {stage_name}, type: {prompt_type})"
                )
    
    # Replace paths with actual content
    config["stage_prompts"] = resolved_prompts
    
    return config


def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate that config has all required fields.
    
    Args:
        config: Configuration dictionary
        
    Raises:
        ValueError: If required fields are missing
    """
    required_fields = ["max_followup_count", "stage_names", "stage_prompts"]
    
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required config field: {field}")
    
    # Validate stage_names matches stage_prompts
    stage_names = set(config["stage_names"])
    prompt_stages = set(config["stage_prompts"].keys())
    
    if stage_names != prompt_stages:
        missing = stage_names - prompt_stages
        extra = prompt_stages - stage_names
        
        msg = "Stage names and prompt stages mismatch."
        if missing:
            msg += f" Missing prompts for: {missing}."
        if extra:
            msg += f" Extra prompts for: {extra}."
        
        raise ValueError(msg)
    
    # Validate each stage has all prompt types
    required_prompt_types = ["diagnostic", "questioning_followup", "questioning_compress", "integration"]
    for stage_name, prompts in config["stage_prompts"].items():
        for prompt_type in required_prompt_types:
            if prompt_type not in prompts:
                raise ValueError(
                    f"Missing {prompt_type} prompt for stage: {stage_name}"
                )

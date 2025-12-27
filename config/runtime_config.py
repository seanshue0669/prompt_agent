# config/runtime_config.py
"""
Runtime configuration for managing global shared resources.
This module holds singleton-like global objects that need to be accessed
across different parts of the application.
"""

class RuntimeConfig:
    """
    Global runtime configuration and shared resources.
    
    Attributes:
        cli_interface: Global CLI interface instance (real or mock)
        config_data: Loaded configuration data from config.json
    """
    cli_interface = None
    config_data = None

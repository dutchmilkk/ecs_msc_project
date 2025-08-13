import os
import yaml
from pathlib import Path
from typing import Dict, Any

class ConfigLoader:
    def __init__(self, config_dir: str = "configs"):
        """Initialize ConfigLoader with base config directory."""
        self.config_dir = Path(config_dir)
        
    def load_configs(self) -> Dict[str, Any]:
        """Load all YAML config files and organize by filename (without extension)."""
        config = {}
        
        if not self.config_dir.exists():
            raise FileNotFoundError(f"Config directory '{self.config_dir}' not found")
        
        # Load root level configs (like experiment_settings.yaml)
        for yaml_file in self.config_dir.glob("*.yaml"):
            file_config = self._load_yaml(yaml_file)
            # Use filename (without extension) as the top-level key
            config_key = yaml_file.stem  # e.g., "experiment_settings" from "experiment_settings.yaml"
            config[config_key] = file_config
        
        # Load subdirectory configs
        for subdir in self.config_dir.iterdir():
            if subdir.is_dir():
                for yaml_file in subdir.glob("*.yaml"):
                    file_config = self._load_yaml(yaml_file)
                    # Use subdirectory/filename as the key
                    config_key = f"{subdir.name}_{yaml_file.stem}"  # e.g., "models_graph" from "models/graph.yaml"
                    config[config_key] = file_config
        
        return config

    def _load_yaml(self, file_path: Path) -> Dict[str, Any]:
        """Load and parse a YAML file with environment variable interpolation."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # Simple environment variable substitution
                content = os.path.expandvars(content)
                return yaml.safe_load(content) or {}
        except Exception as e:
            raise ValueError(f"Error loading config file '{file_path}': {e}")
    
    def get_section(self, config: Dict[str, Any], section: str) -> Dict[str, Any]:
        """Get a specific config section."""
        return config.get(section, {})

    def get_config_value(self, config: Dict[str, Any], section: str, key: str, default=None):
        """Get a specific config value with dot notation support."""
        section_config = self.get_section(config, section)
        keys = key.split('.')
        
        current = section_config
        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return default
        
        return current
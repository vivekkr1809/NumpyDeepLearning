"""Configuration management."""

import yaml
import json
from pathlib import Path
from typing import Dict, Any, Union


class Config:
    """
    Configuration class for managing experiment settings.

    Supports loading from YAML/JSON files and dictionary access.
    """

    def __init__(self, config_dict: Dict[str, Any] = None):
        """
        Initialize Config.

        Args:
            config_dict: Dictionary of configuration parameters
        """
        self._config = config_dict or {}

    @classmethod
    def from_file(cls, file_path: Union[str, Path]) -> 'Config':
        """
        Load configuration from YAML or JSON file.

        Args:
            file_path: Path to configuration file

        Returns:
            Config object
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Config file not found: {file_path}")

        with open(file_path, 'r') as f:
            if file_path.suffix in ['.yaml', '.yml']:
                config_dict = yaml.safe_load(f)
            elif file_path.suffix == '.json':
                config_dict = json.load(f)
            else:
                raise ValueError(f"Unsupported file format: {file_path.suffix}")

        return cls(config_dict)

    def save(self, file_path: Union[str, Path]):
        """
        Save configuration to file.

        Args:
            file_path: Path to save configuration
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, 'w') as f:
            if file_path.suffix in ['.yaml', '.yml']:
                yaml.dump(self._config, f, default_flow_style=False)
            elif file_path.suffix == '.json':
                json.dump(self._config, f, indent=2)
            else:
                raise ValueError(f"Unsupported file format: {file_path.suffix}")

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value.

        Args:
            key: Configuration key (supports nested keys with '.')
            default: Default value if key not found

        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self._config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def set(self, key: str, value: Any):
        """
        Set configuration value.

        Args:
            key: Configuration key (supports nested keys with '.')
            value: Value to set
        """
        keys = key.split('.')
        config = self._config

        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value

    def update(self, updates: Dict[str, Any]):
        """
        Update configuration with dictionary.

        Args:
            updates: Dictionary of updates
        """
        self._deep_update(self._config, updates)

    def _deep_update(self, base: dict, updates: dict):
        """Recursively update nested dictionaries."""
        for key, value in updates.items():
            if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                self._deep_update(base[key], value)
            else:
                base[key] = value

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.

        Returns:
            Configuration dictionary
        """
        return self._config.copy()

    def __getitem__(self, key: str) -> Any:
        """Dictionary-style access."""
        return self.get(key)

    def __setitem__(self, key: str, value: Any):
        """Dictionary-style setting."""
        self.set(key, value)

    def __contains__(self, key: str) -> bool:
        """Check if key exists."""
        return self.get(key) is not None

    def __repr__(self):
        return f"Config({self._config})"


def create_default_config() -> Config:
    """
    Create a default configuration template.

    Returns:
        Config object with default settings
    """
    default_config = {
        'model': {
            'type': 'MLP',
            'params': {
                'input_size': 784,
                'hidden_sizes': [256, 128],
                'output_size': 10,
                'dropout': 0.5,
            }
        },
        'training': {
            'batch_size': 32,
            'epochs': 10,
            'learning_rate': 0.001,
            'optimizer': 'Adam',
            'optimizer_params': {
                'betas': [0.9, 0.999],
                'eps': 1e-8,
                'weight_decay': 0.0,
            },
            'loss': 'CrossEntropyLoss',
        },
        'data': {
            'dataset': 'MNIST',
            'train_split': 0.8,
            'val_split': 0.1,
            'test_split': 0.1,
            'shuffle': True,
        },
        'device': 'cpu',
        'seed': 42,
        'logging': {
            'log_interval': 10,
            'save_dir': './experiments',
            'experiment_name': 'default',
        }
    }

    return Config(default_config)

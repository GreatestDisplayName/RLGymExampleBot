import os
import yaml
from typing import Dict, Any, List, Callable, Tuple

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

SUPPORTED_AGENT_TYPES: List[str] = ["PPO", "SAC", "TD3", "A2C", "DQN"]

class ConfigValidator:
    """
    Encapsulates the schema and validation logic for the configuration.
    """
    _SCHEMA = {
        "training": {
            "n_envs": (int, lambda x: x > 0),
            "total_timesteps": (int, lambda x: x > 0),
            "batch_size": (int, lambda x: x > 0),
            "n_epochs": (int, lambda x: x > 0),
            "learning_rate": (float, lambda x: x > 0),
            "eval_freq": (int, lambda x: x > 0),
            "save_freq": (int, lambda x: x > 0),
            "seed": (int, None),
            "early_stopping_patience": (int, lambda x: x > 0),
            "test_episodes": (int, lambda x: x > 0),
        },
        "model": {
            "hidden_size": (int, lambda x: x > 0),
            "policy_layers": (list, lambda x: all(isinstance(i, int) and i > 0 for i in x)),
            "value_layers": (list, lambda x: all(isinstance(i, int) and i > 0 for i in x)),
            "activation": (str, lambda x: x in ["relu", "tanh", "leaky_relu"]),
            "dropout": (float, lambda x: 0 <= x < 1),
        },
        "environment": {
            "max_steps": (int, lambda x: x > 0),
            "timeout": (int, lambda x: x > 0),
            "team_size": (int, lambda x: x in [1, 2, 3]),
            "spawn_opponents": (bool, None),
            "obs_builder": (str, None),
            "action_parser": (str, None),
            "reward_function": (str, None),
            "gravity": (float, None),
            "ball_drag": (float, lambda x: x >= 0),
            "car_drag": (float, lambda x: x >= 0),
            "boost_force": (float, lambda x: x >= 0),
            "jump_force": (float, lambda x: x >= 0),
            "flip_force": (float, lambda x: x >= 0),
            "frame_stack": (int, lambda x: x > 0),
            "observation_normalization": (bool, None),
            "reward_normalization": (bool, None),
            "difficulty": (str, lambda x: x in ["easy", "medium", "hard"]),
            "agent_type": (str, lambda x: x in SUPPORTED_AGENT_TYPES),
        },
        "paths": {
            "log_dir": (str, None),
            "model_dir": (str, None),
            "tensorboard_dir": (str, None),
            "config_dir": (str, None),
        },
        "evaluation": {
            "n_eval_episodes": (int, lambda x: x > 0),
            "eval_deterministic": (bool, None),
            "render_eval": (bool, None),
        },
        "logging": {
            "log_level": (str, lambda x: x in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]),
            "save_replay_buffer": (bool, None),
            "save_vecnormalize": (bool, None),
        }
    }

    _OPTIONAL_KEYS = {
        "training": ["seed", "early_stopping_patience", "test_episodes"],
        "environment": ["frame_stack", "observation_normalization", "reward_normalization", "difficulty", "agent_type"],
        "model": [],
        "paths": [],
        "evaluation": [],
        "logging": [],
    }

    @staticmethod
    def _validate_key(section_name: str, key: str, value: Any, expected_type: type, validation_func: Callable[[Any], bool] | None):
        """Validates a single key-value pair."""
        if not isinstance(value, expected_type):
            raise TypeError(f"Key '{key}' in section '{section_name}' must be of type {expected_type.__name__}, but got {type(value).__name__}.")
        
        if validation_func and not validation_func(value):
            raise ValueError(f"Invalid value for key '{key}' in section '{section_name}': {value}. Does not pass validation rules.")

    @classmethod
    def validate(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validates the configuration dictionary against the predefined schema and rules.
        Raises ValueError or TypeError for invalid configurations.
        """
        if not isinstance(config, dict):
            raise TypeError("Configuration must be a dictionary.")

        for section_name, keys_schema in cls._SCHEMA.items():
            if section_name not in config:
                raise ValueError(f"Missing required section: '{section_name}' in configuration.")
            
            if not isinstance(config[section_name], dict):
                raise TypeError(f"Section '{section_name}' must be a dictionary.")

            for key, (expected_type, validation_func) in keys_schema.items():
                if key not in config[section_name]:
                    if key not in cls._OPTIONAL_KEYS.get(section_name, []):
                        raise ValueError(f"Missing required key: '{key}' in section '{section_name}'.")
                    continue
                
                cls._validate_key(section_name, key, config[section_name][key], expected_type, validation_func)
        
        return config

def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validates the configuration dictionary using the ConfigValidator class.
    """
    return ConfigValidator.validate(config)


class Config:
    """
    Configuration management for the Rocket League bot
    """
    
    def __init__(self, config_path: str = None):
        self.config_path = config_path or "config.yaml"
        self.config = self.load_config()
        
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        
        default_config = {
            "training": {
                "n_envs": 4,
                "total_timesteps": 1_000_000,
                "batch_size": 2048,
                "n_epochs": 10,
                "learning_rate": 3e-4,
                "eval_freq": 50_000,
                "save_freq": 10000,
            },
            "model": {
                "hidden_size": 512,
                "policy_layers": [512, 512, 256],
                "value_layers": [512, 512, 256],
                "activation": "relu",
                "dropout": 0.1,
            },
            "environment": {
                "max_steps": 1000,
                "timeout": 300,
                "team_size": 1,
                "spawn_opponents": True,
                "obs_builder": "AdvancedObs",
                "action_parser": "DefaultAction",
                "reward_function": "CombinedReward",
            },
            "paths": {
                "log_dir": "logs",
                "model_dir": "models",
                "tensorboard_dir": "tensorboard_logs",
                "config_dir": "configs",
            },
            "evaluation": {
                "n_eval_episodes": 10,
                "eval_deterministic": True,
                "render_eval": False,
            },
            "logging": {
                "log_level": "INFO",
                "save_replay_buffer": True,
                "save_vecnormalize": True,
            }
        }
        
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                user_config = yaml.safe_load(f)
                # Merge with defaults
                if user_config:
                    default_config.update(user_config)
        
        return validate_config(default_config)
    
    def save_config(self, config: Dict[str, Any] = None):
        """Save configuration to YAML file"""
        if config is not None:
            self.config = validate_config(config)
            
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
    
    def get(self, key: str, default=None):
        """Get configuration value using dot notation"""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """Set configuration value using dot notation"""
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def __getitem__(self, key):
        return self.get(key)
    
    def __setitem__(self, key, value):
        self.set(key, value)

    def __getattr__(self, name):
        if name in self.config:
            value = self.config[name]
            if isinstance(value, dict):
                return AttrDict(value)
            return value
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def get_algorithm_params(self, agent_type: str) -> Dict[str, Any]:
        """Get algorithm-specific parameters."""
        return {}


# Global config instance
config = Config()
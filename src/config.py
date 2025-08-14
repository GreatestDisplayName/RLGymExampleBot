import os
import yaml
from typing import Dict, Any
from pydantic.v1 import ValidationError, BaseModel, PositiveInt, confloat

class ActionConfig(BaseModel):
    action_type: str
    n_bins: PositiveInt = 5
    clip_range: confloat(ge=0.1, le=2.0) = 1.0

class NetworkConfig(BaseModel):
    hidden_size: PositiveInt = 256
    dropout_rate: confloat(ge=0.0, le=0.5) = 0.1
    use_layer_norm: bool = True

class TrainingConfig(BaseModel):
    learning_rate: confloat(ge=1e-5, le=1e-2) = 3e-4
    gamma: confloat(ge=0.8, le=0.999) = 0.99
    batch_size: PositiveInt = 128

def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    validated = {
        'action': ActionConfig(**config.get('action', {})).dict(),
        'network': NetworkConfig(**config.get('network', {})).dict(),
        'training': TrainingConfig(**config.get('training', {})).dict()
    }
    return validated


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
                "save_freq": 100_000,
                "eval_freq": 50_000,
            },
            "model": {
                "hidden_size": 512,
                "policy_layers": [512, 512, 256],
                "value_layers": [512, 512, 256],
                "activation": "relu",
                "dropout": 0.1,
            },
            "environment": {
                "timeout": 300,
                "team_size": 1,
                "spawn_opponents": True,
                "obs_builder": "AdvancedObs",
                "action_parser": "DiscreteAction",
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
                default_config.update(user_config)
        
        try:
            return validate_config(default_config)
        except ValidationError as e:
            raise RuntimeError(f"Invalid configuration: {e}") from e
    
    def save_config(self, config: Dict[str, Any] = None):
        """Save configuration to YAML file"""
        if config is not None:
            try:
                self.config = validate_config(config)
            except ValidationError as e:
                raise RuntimeError(f"Invalid configuration during save: {e}") from e
            
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


# Global config instance
config = Config()

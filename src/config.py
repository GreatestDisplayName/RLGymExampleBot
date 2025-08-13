import os
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from pathlib import Path
import yaml
from logger import logger # Import logger

@dataclass
class TrainingConfig:
    """Training algorithm configuration"""
    algorithm: str = "PPO"
    total_timesteps: int = 1000000
    save_freq: int = 10000
    learning_rate: float = 3e-4
    batch_size: int = 64
    n_environments: int = 4
    eval_freq: int = 10000
    test_episodes: int = 5
    test_max_steps: int = 1000
    seed: int = 42
    
    # PPO specific parameters
    ppo_n_steps: int = 2048
    ppo_n_epochs: int = 10
    ppo_gamma: float = 0.99
    ppo_gae_lambda: float = 0.95
    ppo_clip_range: float = 0.2
    ppo_ent_coef: float = 0.01
    ppo_vf_coef: float = 0.5
    ppo_max_grad_norm: float = 0.5
    
    # SAC specific parameters
    sac_buffer_size: int = 1000000
    sac_learning_starts: int = 100
    sac_tau: float = 0.005
    sac_gamma: float = 0.99
    sac_ent_coef: str = "auto"
    sac_target_update_interval: int = 1
    
    # TD3 specific parameters
    td3_buffer_size: int = 1000000
    td3_learning_starts: int = 100
    td3_tau: float = 0.005
    td3_gamma: float = 0.99
    td3_policy_delay: int = 2
    td3_target_noise_clip: float = 0.5
    
    # A2C specific parameters
    a2c_n_steps: int = 5
    a2c_gamma: float = 0.99
    a2c_ent_coef: float = 0.01
    a2c_vf_coef: float = 0.5
    a2c_max_grad_norm: float = 0.5
    
    # DQN specific parameters
    dqn_buffer_size: int = 100000
    dqn_learning_starts: int = 1000
    dqn_target_update_interval: int = 500
    dqn_exploration_fraction: float = 0.1
    dqn_exploration_initial_eps: float = 1.0
    dqn_exploration_final_eps: float = 0.05
    
    # Model architecture
    hidden_size: int = 256
    n_layers: int = 2
    
    # Early stopping
    early_stopping_patience: int = 50
    early_stopping_min_delta: float = 0.01

@dataclass
class EnvironmentConfig:
    """Environment configuration"""
    max_steps: int = 1000
    difficulty: str = "medium"
    game_speed: int = 100
    tick_skip: int = 8
    gravity: float = -9.8
    ball_drag: float = 0.03
    car_drag: float = 0.02
    boost_force: float = 15.0
    jump_force: float = 8.0
    flip_force: float = 12.0
    observation_normalization: bool = True
    reward_normalization: bool = True


@dataclass
class ModelConfig:
    """Neural network model configuration"""
    hidden_size: int = 256
    activation: str = "relu"
    n_layers: int = 2
    dropout: float = 0.1
    use_batch_norm: bool = False
    use_dropout: bool = False
    dropout_rate: float = 0.1


@dataclass
class LoggingConfig:
    """Logging and monitoring configuration"""
    tensorboard_log_dir: str = "logs"
    model_save_dir: str = "models"
    log_level: str = "INFO"
    save_replay_buffer: bool = False
    log_interval: int = 100


@dataclass
class LeagueConfig:
    """Self-play league configuration"""
    base_rating: float = 1000.0
    rating_k_factor: float = 32.0
    min_matches_for_rating: int = 5
    tournament_size: int = 8
    promotion_threshold: float = 0.7
    demotion_threshold: float = 0.3


class ConfigManager:
    """Centralized configuration manager"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "config.yaml"
        self.training = TrainingConfig()
        self.env = EnvironmentConfig()
        self.model = ModelConfig()
        self.logging = LoggingConfig()
        self.league = LeagueConfig()
        
        # Load configuration if file exists
        self._load_config()
    
    def _load_config(self):
        """Load configuration from file if it exists"""
        if Path(self.config_path).exists():
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config_data = yaml.safe_load(f)
                self._update_from_dict(config_data)
            except ImportError:
                logger.warning("PyYAML not available, using default configuration")
            except Exception as e:
                logger.error(f"Error loading config from {self.config_path}: {e}")
                logger.exception(e) # Log traceback
    
    def _update_from_dict(self, config_data: Dict[str, Any]):
        """Update configuration from dictionary"""
        for section, values in config_data.items():
            if hasattr(self, section):
                section_obj = getattr(self, section)
                for key, value in values.items():
                    if hasattr(section_obj, key):
                        setattr(section_obj, key, value)
    
    def save_config(self):
        """Save current configuration to file"""
        try:
            config_data = {
                'training': self.training.__dict__,
                'env': self.env.__dict__,
                'model': self.model.__dict__,
                'logging': self.logging.__dict__,
                'league': self.league.__dict__
            }
            
            Path(self.config_path).parent.mkdir(parents=True, exist_ok=True) # Ensure parent directory exists
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_data, f, default_flow_style=False, indent=2)
            logger.info(f"Configuration saved to {self.config_path}")
        except Exception as e:
            logger.error(f"Error saving config to {self.config_path}: {e}")
            logger.exception(e) # Log traceback
    
    def get_algorithm_params(self, algorithm: str) -> Dict[str, Any]:
        """Get algorithm-specific parameters"""
        params = {}
        prefix = f"{algorithm.lower()}_"
        for key, value in self.training.__dict__.items():
            if key.startswith(prefix):
                params[key[len(prefix):]] = value
        return params

    def validate(self) -> bool:
        """Validate configuration values"""
        errors = []
        
        if self.training.total_timesteps <= 0:
            errors.append("total_timesteps must be positive")
        if self.training.learning_rate <= 0:
            errors.append("learning_rate must be positive")
        if self.training.batch_size <= 0:
            errors.append("batch_size must be positive")
        if self.env.max_steps <= 0:
            errors.append("max_steps must be positive")
        
        if errors:
            logger.error("Configuration validation errors:")
            for error in errors:
                logger.error(f"  - {error}")
            return False
        
        return True


# Global configuration instance
config = ConfigManager()

# Environment variables override
def load_env_overrides():
    """Load configuration from environment variables"""
    if os.getenv("RLGYM_ALGORITHM"):
        config.training.algorithm = os.getenv("RLGYM_ALGORITHM")
    if os.getenv("RLGYM_TOTAL_TIMESTEPS"):
        config.training.total_timesteps = int(os.getenv("RLGYM_TOTAL_TIMESTEPS"))
    if os.getenv("RLGYM_LEARNING_RATE"):
        config.training.learning_rate = float(os.getenv("RLGYM_LEARNING_RATE"))
    if os.getenv("RLGYM_BATCH_SIZE"):
        config.training.batch_size = int(os.getenv("RLGYM_BATCH_SIZE"))
    if os.getenv("RLGYM_N_ENVIRONMENTS"):
        config.training.n_environments = int(os.getenv("RLGYM_N_ENVIRONMENTS"))

# Load environment overrides
load_env_overrides()

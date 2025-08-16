import pytest
from src.config import validate_config
import copy # Import copy module

# A valid configuration for testing
# Define it once, but use deepcopy in tests
_BASE_VALID_CONFIG = {
    "training": {
        "n_envs": 4,
        "total_timesteps": 1000000,
        "batch_size": 2048,
        "n_epochs": 10,
        "learning_rate": 0.0003,
        "eval_freq": 50000,
        "save_freq": 10000,
        "seed": 42,
        "early_stopping_patience": 100,
        "test_episodes": 10,
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
        "gravity": -650.0,
        "ball_drag": 0.03,
        "car_drag": 0.1,
        "boost_force": 1000.0,
        "jump_force": 500.0,
        "flip_force": 500.0,
        "frame_stack": 4,
        "observation_normalization": True,
        "reward_normalization": True,
        "difficulty": "medium",
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

def get_valid_config_copy():
    return copy.deepcopy(_BASE_VALID_CONFIG)

def test_valid_config():
    """Test with a complete and valid configuration."""
    config = get_valid_config_copy()
    try:
        validated_config = validate_config(config)
        assert validated_config == config
    except Exception as e:
        pytest.fail(f"Valid config raised an unexpected exception: {e}")

def test_missing_section():
    """Test with a missing top-level section."""
    config = get_valid_config_copy()
    del config["training"]
    with pytest.raises(ValueError, match="Missing required section: 'training' in configuration."):
        validate_config(config)

def test_incorrect_type_top_level_section():
    """Test with a top-level section having incorrect type."""
    config = get_valid_config_copy()
    config["training"] = "not_a_dict"
    with pytest.raises(TypeError, match="Section 'training' must be a dictionary."):
        validate_config(config)

def test_missing_key():
    """Test with a missing required key within a section."""
    config = get_valid_config_copy()
    del config["training"]["n_envs"]
    with pytest.raises(ValueError, match="Missing required key: 'n_envs' in section 'training'."):
        validate_config(config)

def test_incorrect_type_key():
    """Test with a key having an incorrect data type."""
    config = get_valid_config_copy()
    config["training"]["total_timesteps"] = "1000000" # Should be int
    with pytest.raises(TypeError, match="Key 'total_timesteps' in section 'training' must be of type int, but got str."):
        validate_config(config)

def test_invalid_value_range():
    """Test with a key having a value outside the valid range."""
    config = get_valid_config_copy()
    config["training"]["n_envs"] = 0 # Must be > 0
    with pytest.raises(ValueError, match="Invalid value for key 'n_envs' in section 'training': 0. Does not pass validation rules."):
        validate_config(config)

def test_invalid_activation_function():
    """Test with an invalid activation function string."""
    config = get_valid_config_copy()
    config["model"]["activation"] = "invalid_activation"
    with pytest.raises(ValueError, match="Invalid value for key 'activation' in section 'model': invalid_activation. Does not pass validation rules."):
        validate_config(config)

def test_invalid_dropout_rate():
    """Test with a dropout rate outside the valid range [0, 1)."""
    config = get_valid_config_copy()
    config["model"]["dropout"] = 1.5
    with pytest.raises(ValueError, match="Invalid value for key 'dropout' in section 'model': 1.5. Does not pass validation rules."):
        validate_config(config)

def test_invalid_policy_layers_type():
    """Test with policy_layers containing non-integer values."""
    config = get_valid_config_copy()
    config["model"]["policy_layers"] = [512, "256", 256]
    # The error message will be more specific about the inner elements,
    # but the overall type check for the list itself should pass.
    # The inner lambda will catch the non-int.
    with pytest.raises(ValueError, match="Invalid value for key 'policy_layers' in section 'model': \[512, '256', 256\]. Does not pass validation rules."):
        validate_config(config)

def test_invalid_policy_layers_value():
    """Test with policy_layers containing non-positive integer values."""
    config = get_valid_config_copy()
    config["model"]["policy_layers"] = [512, 0, 256]
    with pytest.raises(ValueError, match="Invalid value for key 'policy_layers' in section 'model': \[512, 0, 256\]. Does not pass validation rules."):
        validate_config(config)

def test_invalid_team_size():
    """Test with an invalid team_size."""
    config = get_valid_config_copy()
    config["environment"]["team_size"] = 4 # Must be 1, 2, or 3
    with pytest.raises(ValueError, match="Invalid value for key 'team_size' in section 'environment': 4. Does not pass validation rules."):
        validate_config(config)

def test_invalid_difficulty():
    """Test with an invalid difficulty string."""
    config = get_valid_config_copy()
    config["environment"]["difficulty"] = "super_hard"
    with pytest.raises(ValueError, match="Invalid value for key 'difficulty' in section 'environment': super_hard. Does not pass validation rules."):
        validate_config(config)

def test_invalid_log_level():
    """Test with an invalid log_level string."""
    config = get_valid_config_copy()
    config["logging"]["log_level"] = "ULTRA_DEBUG"
    with pytest.raises(ValueError, match="Invalid value for key 'log_level' in section 'logging': ULTRA_DEBUG. Does not pass validation rules."):
        validate_config(config)

def test_optional_keys_missing():
    """Test that optional keys can be missing without raising an error."""
    config = get_valid_config_copy()
    del config["training"]["seed"]
    del config["training"]["early_stopping_patience"]
    del config["training"]["test_episodes"]
    del config["environment"]["frame_stack"]
    del config["environment"]["observation_normalization"]
    del config["environment"]["reward_normalization"]
    del config["environment"]["difficulty"]
    try:
        validate_config(config)
    except Exception as e:
        pytest.fail(f"Missing optional keys raised an unexpected exception: {e}")

def test_invalid_agent_type_in_environment():
    """Test with an invalid agent_type in the environment section."""
    config = get_valid_config_copy()
    config["environment"]["agent_type"] = "UNSUPPORTED_AGENT"
    with pytest.raises(ValueError, match="Invalid agent_type 'UNSUPPORTED_AGENT' in section 'environment'. Supported types are: PPO, SAC, TD3, A2C, DQN"):
        validate_config(config)
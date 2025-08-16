import pytest
from src.config import Config
from src.rlbot_support import validate_config
from configparser import ConfigParser

def test_config_instantiation():
    """
    Tests that the Config class can be instantiated.
    """
    try:
        _ = Config()
    except Exception as e:
        pytest.fail(f"Failed to instantiate Config: {e}")

def test_validate_config_missing_section():
    """
    Tests that validate_config raises ValueError for a missing section.
    """
    config = ConfigParser()
    with pytest.raises(ValueError, match="Missing required section: \[Bot Parameters\] in bot.cfg"):
        validate_config(config)

def test_validate_config_missing_option():
    """
    Tests that validate_config raises ValueError for a missing option.
    """
    config = ConfigParser()
    config.add_section('Bot Parameters')
    config.add_section('Locations')
    with pytest.raises(ValueError, match="Missing required option: 'obs_builder' in \[Bot Parameters\] in bot.cfg"):
        validate_config(config)

def test_validate_config_invalid_tick_skip():
    """
    Tests that validate_config raises ValueError for an invalid tick_skip.
    """
    config = ConfigParser()
    config.add_section('Bot Parameters')
    config.add_section('Locations')
    config.set('Bot Parameters', 'obs_builder', 'test')
    config.set('Bot Parameters', 'act_parser', 'test')
    config.set('Bot Parameters', 'tick_skip', 'not-an-int')
    config.set('Locations', 'model_path', 'test')
    with pytest.raises(ValueError, match="Invalid value for 'tick_skip' in \[Bot Parameters\]"):
        validate_config(config)

def test_validate_config_valid_config():
    """
    Tests that validate_config does not raise an exception for a valid config.
    """
    config = ConfigParser()
    config.add_section('Bot Parameters')
    config.add_section('Locations')
    config.set('Bot Parameters', 'obs_builder', 'test')
    config.set('Bot Parameters', 'act_parser', 'test')
    config.set('Bot Parameters', 'tick_skip', '8')
    config.set('Locations', 'model_path', 'test')
    try:
        validate_config(config)
    except ValueError:
        pytest.fail("validate_config raised ValueError unexpectedly")
import glob
import importlib
import os
import re # Added
from configparser import ConfigParser
from pathlib import Path
from typing import Optional

import numpy as np

from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.utils.structures.game_data_struct import GameTickPacket

from agent import Agent
from rlgym_compat import GameState

def validate_config(config: ConfigParser):
    """
    Validates the bot configuration.
    """
    # Check for required sections
    required_sections = ['Bot Parameters', 'Locations']
    for section in required_sections:
        if not config.has_section(section):
            raise ValueError(f"Missing required section: [{section}] in bot.cfg")

    # Check for required options in [Bot Parameters]
    required_bot_params = ['obs_builder', 'act_parser', 'tick_skip']
    for param in required_bot_params:
        if not config.has_option('Bot Parameters', param):
            raise ValueError(f"Missing required option: '{param}' in [Bot Parameters] in bot.cfg")

    # Check for required options in [Locations]
    if not config.has_option('Locations', 'model_path'):
        raise ValueError("Missing required option: 'model_path' in [Locations] in bot.cfg")

    # Check tick_skip value
    try:
        tick_skip = config.getint('Bot Parameters', 'tick_skip')
        if tick_skip <= 0:
            raise ValueError("'tick_skip' must be a positive integer.")
    except ValueError as e:
        raise ValueError(f"Invalid value for 'tick_skip' in [Bot Parameters]: {e}") from e

class RLBot(BaseAgent):
    def __init__(self, name, team, index):
        super().__init__(name, team, index)
        self.agent = None
        self.obs_builder = None
        self.act_parser = None
        self.tick_skip = None

        self.game_state: GameState = None
        self.controls = None
        self.action = None
        self.update_action = True
        self.ticks = 0
        self.prev_time = 0

    def initialize_agent(self):
        # Load config
        config_path = Path(self.agent_metadata.bot_directory) / 'bot.cfg' # pylint: disable=no-member
        config = ConfigParser()
        config.read(config_path)
        validate_config(config)

        # Load obs builder
        obs_builder_str = config.get('Bot Parameters', 'obs_builder', fallback='DefaultObs')
        self.obs_builder = self.load_class(obs_builder_str, 'obs')()

        # Load act parser
        act_parser_str = config.get('Bot Parameters', 'act_parser', fallback='DefaultAction')
        self.act_parser = self.load_class(act_parser_str, 'action')()

        # Load model
        model_path_str = config.get('Locations', 'model_path')
        model_dir = config.get('Locations', 'model_dir', fallback='../models')

        if model_path_str == 'latest':
            model_path = self.get_latest_model(model_dir)
        else:
            model_path = Path(self.agent_metadata.bot_directory) / model_path_str # pylint: disable=no-member

        

        # Load agent
        self.agent = Agent(input_size=self.obs_builder.get_obs_size(), model_path=model_path)

        # Load tick skip
        self.tick_skip = config.getint('Bot Parameters', 'tick_skip', fallback=8)

        # Initialize game state
        self.game_state = GameState(self.get_field_info())
        self.ticks = self.tick_skip
        self.prev_time = 0
        self.controls = SimpleControllerState()
        self.action = np.zeros(self.act_parser.get_action_size())
        self.update_action = True

    def get_output(self, packet: GameTickPacket) -> SimpleControllerState:
        cur_time = packet.game_info.seconds_elapsed
        delta = cur_time - self.prev_time
        self.prev_time = cur_time

        ticks_elapsed = round(delta * 120)
        self.ticks += ticks_elapsed
        self.game_state.decode(packet, ticks_elapsed)

        if self.update_action:
            self.update_action = False

            obs = self.obs_builder.build_obs(self.game_state.players[self.index], self.game_state, self.action)
            self.action = self.act_parser.parse_actions(self.agent.act(obs), self.game_state)[0]

        if self.ticks >= self.tick_skip - 1:
            self.update_controls(self.action)

        if self.ticks >= self.tick_skip:
            self.ticks = 0
            self.update_action = True

        return self.controls

    def update_controls(self, action):
        self.controls.throttle = action[0]
        self.controls.steer = action[1]
        self.controls.pitch = action[2]
        self.controls.yaw = action[3]
        self.controls.roll = action[4]
        self.controls.jump = action[5] > 0
        self.controls.boost = action[6] > 0
        self.controls.handbrake = action[7] > 0

    def load_class(self, class_name: str, module_folder: str) -> type:
        """Dynamically loads a class from a module.

        Args:
            class_name: The name of the class to load (e.g., 'DefaultObs').
            module_folder: The sub-directory within 'src' (e.g., 'obs').

        Returns:
            The loaded class type.

        Raises:
            ImportError: If the class or module cannot be found.
        """
        # Convert CamelCase to snake_case for the file name (e.g., DefaultObs -> default_obs)
        module_name = re.sub(r'(?<!^)(?=[A-Z])', '_', class_name).lower()
        full_module_path = f"src.{module_folder}.{module_name}"

        try:
            module = importlib.import_module(full_module_path)
            return getattr(module, class_name)
        except ImportError as e:
            raise ImportError(f"Could not import module '{full_module_path}': {e}") from e
        except AttributeError:
            raise ImportError(f"Class '{class_name}' not found in module '{full_module_path}'")

    def get_latest_model(self, model_dir: str) -> Path:
        """Finds the most recently modified model file in a directory.

        Args:
            model_dir: The directory to search for models.

        Returns:
            The path to the latest model file.

        Raises:
            FileNotFoundError: If the model directory does not exist or no model files are found.
        """
        # pylint: disable=no-member
        model_search_path = Path(self.agent_metadata.bot_directory) / model_dir
        # pylint: enable=no-member
        
        if not model_search_path.exists():
            raise FileNotFoundError(f"Model directory not found: {model_search_path}. Please ensure the directory exists and contains trained models.")

        list_of_files = list(model_search_path.glob('**/*.pth'))
        if not list_of_files:
            raise FileNotFoundError(f"No .pth model files found in {model_search_path}. Please train a model or ensure .pth files are present.")
            
        latest_file = max(list_of_files, key=lambda p: p.stat().st_mtime)
        return latest_file

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from stable_baselines3 import A2C, DQN, PPO, SAC, TD3
from stable_baselines3.common.utils import set_random_seed
from training_env import SimpleRocketLeagueEnv
from logger import logger
from config import config

# Define supported agent types as a global constant for easy reference and validation
SUPPORTED_AGENT_TYPES: List[str] = ["PPO", "SAC", "TD3", "A2C", "DQN"]


def create_vec_env(n_envs: Optional[int] = None, frame_stack: Optional[int] = None, agent_type: str = None) -> SimpleRocketLeagueEnv:
    """
    Creates a vectorized environment for training.
    This function sets up multiple parallel environments, applies monitoring,
    and optionally normalizes observations/rewards and stacks frames.
    
    Args:
        n_envs: Number of parallel environments to create
        frame_stack: Number of frames to stack (None to disable)
        agent_type: Type of agent (e.g., 'DQN', 'PPO')
        
    Returns:
        Vectorized environment ready for training
    """
    # Use config values if not specified
    if n_envs is None:
        n_envs = getattr(config.training, 'n_envs', 1)
    if frame_stack is None:
        frame_stack = getattr(config.environment, 'frame_stack', None)
        
    # Special handling for DQN
    if agent_type == "DQN":
        n_envs = 1
        frame_stack = None
        logger.info("Using single environment for DQN (no parallelization or frame stacking)")

    def make_env():
        """
        Inner function to create a single instance of the training environment.
        This is used by DummyVecEnv to create multiple parallel environments.
        """
        env = SimpleRocketLeagueEnv(
            max_steps=config.environment.get('max_steps', 1000),
            difficulty=config.environment.get('difficulty', 1.0),
            agent_type=agent_type
        )
        return env

    # Create the vectorized environment
    env = make_env()
    
    # Apply observation and reward normalization if enabled
    if config.environment.get('observation_normalization', False) or \
       config.environment.get('reward_normalization', False):
        logger.warning("Observation/reward normalization is not supported without VecNormalize.")
    
    # Apply frame stacking if specified and not using DQN
    if frame_stack is not None and agent_type != "DQN" and frame_stack and frame_stack > 1:
        logger.warning("Frame stacking is not supported without VecFrameStack.")
    elif frame_stack and agent_type == "DQN":
        logger.warning("Frame stacking is not supported for DQN and will be disabled")

    logger.info(f"Created environment with {n_envs} parallel environments (Note: Parallel environments are not actually created without DummyVecEnv)")
    return env


def create_model(agent_type: str, env: SimpleRocketLeagueEnv, model_dir: Union[str, Path]) -> Union[PPO, SAC, TD3, A2C, DQN]:
    """
    Creates and configures the specified Stable-Baselines3 model.
    This function dynamically initializes the correct RL algorithm based on `agent_type`
    and applies common and algorithm-specific hyperparameters from the global configuration.
    
    Args:
        agent_type (str): The type of agent to create (e.g., "PPO", "SAC", "TD3", "A2C", "DQN").
                          Must be one of `SUPPORTED_AGENT_TYPES`.
        env (SimpleRocketLeagueEnv): The environment for training.
        model_dir (Union[str, Path]): The directory where the model will be saved.
                                      Used for setting up TensorBoard logging paths.
                                      
    Returns:
        Union[PPO, SAC, TD3, A2C, DQN]: The created Stable-Baselines3 model instance.
        
    Raises:
        ValueError: If an unsupported `agent_type` is provided.
    """
    # Ensure the model directory exists for saving logs and checkpoints
    model_dir_path = Path(model_dir)
    model_dir_path.mkdir(parents=True, exist_ok=True)

    # Set a random seed for reproducibility of training runs.
    # This ensures that experiments can be replicated with the same results.
    set_random_seed(config.training.seed if hasattr(config.training, 'seed') else 42)

    # Retrieve algorithm-specific parameters from the global configuration.
    # These parameters are defined in `config.py` and are tailored for each RL algorithm.
    algo_params: Dict[str, Any] = config.get_algorithm_params(agent_type)

    # Map string agent types to their corresponding Stable-Baselines3 class constructors.
    agent_classes: Dict[str, Any] = {
        "PPO": PPO,
        "SAC": SAC,
        "TD3": TD3,
        "A2C": A2C,
        "DQN": DQN,
    }

    # Validate the provided agent_type against the list of supported types.
    if agent_type not in agent_classes:
        raise ValueError(f"Unknown agent type: {agent_type}. "
                         f"Supported types: {', '.join(agent_classes.keys())}")

    model_class = agent_classes[agent_type] # Get the constructor for the specified agent type

    # Define common parameters that apply to most Stable-Baselines3 models.
    common_params: Dict[str, Any] = {
        "policy": "MlpPolicy", # Use a Multi-layer Perceptron policy network
        "env": env,
        "learning_rate": config.training.learning_rate,
        "verbose": 1,
        "tensorboard_log": str(Path("logs") / agent_type),
    }

    # Define policy network architecture.
    # `net_arch` specifies the number and size of hidden layers in the policy and value networks.
    policy_kwargs: Dict[str, Any] = {
        "net_arch": config.model.policy_layers
    }

    # Special handling for PPO's policy_kwargs: PPO often uses separate networks for policy (pi) and value (vf) functions.
    if agent_type == "PPO":
        policy_kwargs["net_arch"] = [
            dict(pi=config.model.policy_layers,
                 vf=config.model.value_layers)
        ]

    # Add batch_size parameter for algorithms that require it (e.g., off-policy algorithms like SAC, TD3, DQN, and PPO)
    if agent_type in ["PPO", "SAC", "TD3", "DQN"]:
        common_params["batch_size"] = config.training.batch_size

    # Instantiate the Stable-Baselines3 model with all collected parameters.
    model = model_class(
        **common_params,
        policy_kwargs=policy_kwargs,
        **algo_params
    )

    logger.info(f"Created {agent_type} model with policy architecture: "
               f"{len(config.model.policy_layers)} layers of {config.model.hidden_size} units")

    return model

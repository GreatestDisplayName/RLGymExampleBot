import argparse
import os
import time
from pathlib import Path

import numpy as np
from stable_baselines3 import A2C, DQN, PPO, SAC, TD3

from logger import logger, timed_operation
from training_env import SimpleRocketLeagueEnv


@timed_operation("Agent vs. Agent Simulation")
def simulate_agents(
    model_path_a: str,
    model_path_b: str,
    agent_type_a: str = "PPO",
    agent_type_b: str = "PPO",
    n_episodes: int = 1,
    render: bool = False,
):
    """Simulate two agents playing against each other in a multi-agent environment."""

    logger.info(f"Starting Agent vs. Agent Simulation for {n_episodes} episodes.")
    logger.info(f"Agent A: {model_path_a} ({agent_type_a})")
    logger.info(f"Agent B: {model_path_b} ({agent_type_b})")

    # Map agent types to their respective Stable Baselines3 classes
    agent_classes = {
        "PPO": PPO,
        "SAC": SAC,
        "TD3": TD3,
        "A2C": A2C,
        "DQN": DQN,
    }

    if agent_type_a not in agent_classes:
        raise ValueError(
            f"Unsupported agent type A: {agent_type_a}. "
            f"Supported types: {', '.join(agent_classes.keys())}"
        )
    if agent_type_b not in agent_classes:
        raise ValueError(
            f"Unsupported agent type B: {agent_type_b}. "
            f"Supported types: {', '.join(agent_classes.keys())}"
        )

    env = None
    try:
        env = SimpleRocketLeagueEnv(difficulty="medium")  # Multi-agent environment

        # Load models without env to bypass observation space validation
        model_a = agent_classes[agent_type_a].load(model_path_a)
        model_b = agent_classes[agent_type_b].load(model_path_b)

        for episode in range(n_episodes):
            obs, info = env.reset()
            done = False
            truncated = False
            total_reward_a = 0.0
            total_reward_b = 0.0
            episode_length = 0

            logger.info(f"--- Episode {episode + 1} ---")

            while not done and not truncated:
                print(f"Type of obs: {type(obs)}") # Debugging line
                print(f"Content of obs: {obs}") # Debugging line
                obs_a, obs_b = obs
                action_a, _ = model_a.predict(np.array(obs_a).reshape(1, -1), deterministic=True)
                action_b, _ = model_b.predict(np.array(obs_b).reshape(1, -1), deterministic=True)

                obs, rewards, done, truncated, info = env.step((action_a.flatten(), action_b.flatten()))
                reward_a, reward_b = rewards

                total_reward_a += reward_a
                total_reward_b += reward_b

                episode_length += 1

                if render:
                    env.render()

            logger.info(f"Episode {episode + 1} finished.")
            logger.info(f"  Total Reward Agent A: {total_reward_a:.2f}")
            logger.info(f"  Total Reward Agent B: {total_reward_b:.2f}")
            logger.info(f"  Episode Length: {episode_length} steps")

    except FileNotFoundError as e:
        logger.error(f"Model not found: {e}")
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate two agents in a multi-agent manner.")
    parser.add_argument(
        "--model_a", type=str, required=True, help="Path to the first agent's model"
    )
    parser.add_argument(
        "--model_b", type=str, required=True, help="Path to the second agent's model"
    )
    parser.add_argument(
        "--type_a",
        type=str,
        default="PPO",
        choices=["PPO", "SAC", "TD3", "A2C", "DQN"],
        help="Type of the first agent (e.g., PPO, SAC)",
    )
    parser.add_argument(
        "--type_b",
        type=str,
        default="PPO",
        choices=["PPO", "SAC", "TD3", "A2C", "DQN"],
        help="Type of the second agent (e.g., PPO, SAC)",
    )
    parser.add_argument(
        "--episodes", type=int, default=1, help="Number of episodes to simulate"
    )
    parser.add_argument("--render", action="store_true", help="Render the simulation")

    args = parser.parse_args()

    simulate_agents(
        model_path_a=args.model_a,
        model_path_b=args.model_b,
        agent_type_a=args.type_a,
        agent_type_b=args.type_b,
        n_episodes=args.episodes,
        render=args.render,
    )
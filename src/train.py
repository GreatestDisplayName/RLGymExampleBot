import json
import os
import time
import traceback # Added
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import A2C, DQN, PPO, SAC, TD3 # Added
from stable_baselines3.common.callbacks import (
    BaseCallback, CheckpointCallback, EvalCallback, EveryNTimesteps,
    StopTrainingOnNoModelImprovement
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecNormalize
import tensorboard

from config import config
from logger import logger, timed_operation
from metrics import EarlyStopping, monitor
from training_env import get_env_info, make_training_env
from utils import create_model, create_vec_env


class CustomCallback(BaseCallback):
    """Custom callback for additional monitoring and control"""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.training_start_time = time.time()
        self.last_log_time = time.time()
        self.log_interval = 100
        
    def _on_step(self) -> bool:
        current_time = time.time()
        
        # Log training progress every log_interval steps
        if self.num_timesteps % self.log_interval == 0:
            elapsed_time = current_time - self.training_start_time
            steps_per_sec = self.num_timesteps / elapsed_time if elapsed_time > 0 else 0
            
            logger.info(f"Training progress: {self.num_timesteps} steps, "
                       f"{steps_per_sec:.1f} steps/sec, "
                       f"elapsed: {elapsed_time:.1f}s")
            
            self.last_log_time = current_time
        
        return True





def create_callbacks(model_dir, agent_type, eval_env=None):
    """Create training callbacks"""
    callbacks = []
    
    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=config.get("training.save_freq", 10000), # Use .get with default
        save_path=model_dir,
        name_prefix=f"{agent_type}_model"
    )
    callbacks.append(checkpoint_callback)
    
    # Custom monitoring callback
    custom_callback = CustomCallback()
    callbacks.append(custom_callback)
    
    # Evaluation callback if eval environment provided
    if eval_env is not None:
        # Stop training if no improvement
        # Assuming early_stopping_patience is still directly accessible or has a default
        early_stopping_patience = config.get("training.early_stopping_patience", 10) 
        stop_callback = StopTrainingOnNoModelImprovement(
            max_no_improvement_evals=early_stopping_patience,
            min_evals=10
        )
        
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=f"{model_dir}/best_model/",
            log_path=f"{model_dir}/eval_logs/",
            eval_freq=config.get("training.eval_freq", 10000), # Use .get with default
            n_eval_episodes=config.get("training.test_episodes", 10), # Use .get with default
            deterministic=True,
            render=False,
            callback_after_eval=stop_callback
        )
        callbacks.append(eval_callback)
    
    logger.info(f"Created {len(callbacks)} training callbacks")
    return callbacks


@timed_operation("Agent Training")
def train_agent(agent_type="PPO", total_timesteps=1000000,
                resume_from=None, eval_env=None):
    """Train the agent using the specified algorithm"""
    env = None
    model = None
    try:
        logger.training_start(agent_type, total_timesteps)

        # Create environment with agent-specific settings
        env = create_vec_env(agent_type=agent_type)

        # Get environment info
        get_env_info(env)

        # Create model directory
        model_dir = f"models/{agent_type}"
        os.makedirs(model_dir, exist_ok=True)

        # Create or load model
        if resume_from and os.path.exists(resume_from):
            logger.info(f"Loading existing model from {resume_from}")
            try:
                model = create_model(agent_type, env, model_dir)
                model = model.load(resume_from, env=env)
            except FileNotFoundError:
                logger.error(f"Could not find model to resume from at {resume_from}")
                raise
        else:
            model = create_model(agent_type, env, model_dir)

        # Create callbacks
        callbacks = create_callbacks(model_dir, agent_type, eval_env)

        # Start training
        logger.info(f"Starting training for {total_timesteps} timesteps...")
        start_time = time.time()

        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True
        )

        training_time = time.time() - start_time

        # Save final model
        final_model_path = f"{model_dir}/final_model"
        model.save(final_model_path)

        # Save training metadata
        metadata = {
            "agent_type": agent_type,
            "total_timesteps": total_timesteps,
            "training_time_seconds": training_time,
            "final_model_path": final_model_path,
            "config": {
                "learning_rate": config.training.learning_rate,
                "batch_size": config.training.batch_size,
                "n_environments": config.training.n_envs
            }
        }

        with open(f"{model_dir}/training_metadata.json", "w", encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)

        logger.info("Training completed successfully!")
        logger.info(f"Final model saved to: {final_model_path}")
        logger.info(f"Training time: {training_time:.1f} seconds")
        logger.info(f"Average speed: {total_timesteps/training_time:.1f} steps/second")

        return final_model_path

    except KeyboardInterrupt:
        logger.warning("Training interrupted by user. Saving model...")
        if model is not None:
            interrupted_model_path = f"{model_dir}/final_model_interrupted"
            model.save(interrupted_model_path)
            logger.info(f"Model saved to {interrupted_model_path}")
        raise
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise
    finally:
        if env is not None:
            env.close()





@timed_operation("Agent Testing")
def test_agent(model_path, agent_type="PPO", n_episodes=None, render=False):
    """Test a trained agent"""
    
    if n_episodes is None:
        n_episodes = config.get("training.test_episodes", 10)
    
    test_env = None
    try:
        logger.info(f"Testing agent from {model_path}")
        
        # Create test environment
        test_env = make_training_env(
            max_steps=config.get("training.test_max_steps", 3000),
            difficulty=config.get("env.difficulty", "medium")
        )
        
        # Map agent types to their respective Stable Baselines3 classes
        agent_classes = {
            "PPO": PPO,
            "SAC": SAC,
            "TD3": TD3,
            "A2C": A2C,
            "DQN": DQN,
        }

        if agent_type not in agent_classes:
            raise ValueError(f"Unsupported agent type: {agent_type}. "
                             f"Supported types: {', '.join(agent_classes.keys())}")

        model_class = agent_classes[agent_type]
        try:
            model = model_class.load(model_path, env=test_env)
        except FileNotFoundError:
            logger.error(f"Could not find model to test at {model_path}")
            raise
        
        # Test the agent
        total_reward = 0
        episode_lengths = []
        
        for episode in range(n_episodes):
            obs, _ = test_env.reset()
            episode_reward = 0
            episode_length = 0
            
            while True:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, _ = test_env.step(action)
                episode_reward += reward
                episode_length += 1
                
                if render:
                    test_env.render()
                
                if done or truncated:
                    break
            
            total_reward += episode_reward
            episode_lengths.append(episode_length)
            
            logger.info(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, "
                       f"Length = {episode_length}")
        
        # Calculate statistics
        avg_reward = total_reward / n_episodes
        avg_length = np.mean(episode_lengths)
        
        logger.info("Testing completed!")
        logger.info(f"Average reward: {avg_reward:.2f}")
        logger.info(f"Average episode length: {avg_length:.1f}")
        
        return {
            "avg_reward": avg_reward,
            "avg_length": avg_length,
            "episode_rewards": list(episode_lengths)
        }
        
    except Exception as e:
        logger.error(f"Testing failed: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise
    finally:
        if test_.env is not None:
            test_env.close()


def main():
    """Main training function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train a Rocket League agent")
    parser.add_argument("--agent", type=str, default="PPO", 
                       choices=["PPO", "SAC", "TD3", "A2C", "DQN"],
                       help="Agent type to train")
    parser.add_argument("--timesteps", type=int, default=1000000,
                       help="Total training timesteps")
    
    parser.add_argument("--resume", type=str, default=None,
                       help="Resume training from checkpoint")
    parser.add_argument("--test", action="store_true",
                       help="Test the agent after training")
    parser.add_argument("--render", action="store_true",
                       help="Render during testing")
    parser.add_argument("--profile", action="store_true",
                        help="Enable performance profiling")
    
    args = parser.parse_args()
    
    if args.profile:
        import cProfile
        import pstats
        profiler = cProfile.Profile()
        profiler.enable()

    try:
        # Train the agent
        model_path = train_agent(
            agent_type=args.agent,
            total_timesteps=args.timesteps,
            
            resume_from=args.resume,
            eval_env=None # Explicitly pass None for eval_env
        )
        
        # Test if requested
        if args.test:
            logger.info("Testing trained agent...")
            test_results = test_agent(
                model_path=model_path,
                agent_type=args.agent,
                render=args.render
            )
            
            logger.info("Testing completed successfully!")
            
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        return 1
    finally:
        if args.profile:
            profiler.disable()
            stats = pstats.Stats(profiler).sort_stats('cumtime')
            stats.print_stats()
            stats.dump_stats('training.prof')
            logger.info("Profiling data saved to training.prof")

    return 0


if __name__ == "__main__":
    exit(main())

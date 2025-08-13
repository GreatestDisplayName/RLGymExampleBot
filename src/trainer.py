import os
import time
import torch
import numpy as np
import signal
import sys
from typing import Optional, Dict, Any, List
from torch.utils.tensorboard import SummaryWriter
from rlgym_compat import GameState
from rlgym.utils.terminal_conditions.common_conditions import TimeoutCondition, GoalScoredCondition
from rlgym.utils.action_parsers import DiscreteAction
from rlgym.utils.obs_builders import DefaultObs
from rlgym.utils.reward_functions import DefaultReward
from rlgym.utils.state_setters import DefaultState
from rlgym_tools.sb3_utils import SB3MultipleInstanceEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList, BaseCallback
from stable_baselines3.common.vec_env import VecNormalize, VecCheckNan
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

from src.agent import Agent
from src.rewards import CombinedReward
from src.obs.advanced_obs import AdvancedObs
from src.config import Config
from src.utils.logger import setup_logger


class GracefulExitCallback(BaseCallback):
    """Callback to handle graceful shutdown on interrupt signals"""
    
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.exit_requested = False
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle interrupt signals"""
        print("\nGraceful shutdown requested...")
        self.exit_requested = True
    
    def _on_step(self) -> bool:
        """Check if exit was requested"""
        return not self.exit_requested


class RLGymTrainer:
    """
    Enhanced training manager for the Rocket League bot using PPO
    """

    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Setup directories
        self.log_dir = config.get("paths.log_dir", "logs")
        self.model_dir = config.get("paths.model_dir", "models")
        self.tensorboard_dir = config.get("paths.tensorboard_dir", "tensorboard_logs")
        
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.tensorboard_dir, exist_ok=True)
        
        # Setup logger
        self.logger = setup_logger(self.log_dir, config.get("experiment_name"))
        
        # Training parameters
        self.n_envs = config.get("training.n_envs", 4)
        self.total_timesteps = config.get("training.total_timesteps", 1_000_000)
        self.batch_size = config.get("training.batch_size", 2048)
        self.n_epochs = config.get("training.n_epochs", 10)
        self.learning_rate = config.get("training.learning_rate", 3e-4)
        self.save_freq = config.get("training.save_freq", 100_000)
        self.eval_freq = config.get("training.eval_freq", 50_000)
        
        # Environment parameters
        self.timeout = config.get("environment.timeout", 300)
        self.team_size = config.get("environment.team_size", 1)
        self.spawn_opponents = config.get("environment.spawn_opponents", True)
        
        # Setup environment
        self.env = None
        self.eval_env = None
        self.model = None
        
        # Set random seed for reproducibility
        seed = config.get("training.seed", None)
        if seed is not None:
            set_random_seed(seed)
        
    def setup_environment(self, eval_mode: bool = False):
        """Setup the training or evaluation environment"""
        
        # Environment configuration
        env_config = {
            "reward_fn": CombinedReward(),
            "obs_builder": AdvancedObs(),
            "action_parser": DiscreteAction(),
            "terminal_conditions": [TimeoutCondition(self.timeout), GoalScoredCondition()],
            "state_setter": DefaultState(),
            "team_size": self.team_size,
            "spawn_opponents": self.spawn_opponents,
        }
        
        if eval_mode:
            # Single environment for evaluation
            env = SB3MultipleInstanceEnv(
                match_config=env_config,
                num_instances=1,
                wait_time=15
            )
        else:
            # Multiple environments for training
            env = SB3MultipleInstanceEnv(
                match_config=env_config,
                num_instances=self.n_envs,
                wait_time=15
            )
        
        # Add monitoring and normalization
        env = VecCheckNan(env, raise_exception=True)
        env = VecNormalize(
            env, 
            norm_obs=True, 
            norm_reward=not eval_mode,  # Don't normalize rewards for eval
            clip_obs=10.0,
            training=not eval_mode
        )
        
        return env
    
    def create_model(self, load_path: Optional[str] = None):
        """Create or load the PPO model"""
        
        policy_kwargs = dict(
            net_arch=[dict(
                pi=config.get("model.policy_layers", [512, 512, 256]), 
                vf=config.get("model.value_layers", [512, 512, 256])
            )],
            activation_fn=getattr(torch.nn, config.get("model.activation", "ReLU")),
        )
        
        if load_path and os.path.exists(load_path):
            print(f"Loading model from {load_path}")
            self.model = PPO.load(
                load_path, 
                env=self.env,
                tensorboard_log=self.tensorboard_dir,
                device=self.device
            )
        else:
            print("Creating new model...")
            self.model = PPO(
                "MlpPolicy",
                self.env,
                learning_rate=self.learning_rate,
                n_steps=self.batch_size // self.n_envs,
                batch_size=self.batch_size // 4,
                n_epochs=self.n_epochs,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                clip_range_vf=None,
                ent_coef=0.01,
                vf_coef=0.5,
                max_grad_norm=0.5,
                use_sde=False,
                sde_sample_freq=-1,
                target_kl=None,
                tensorboard_log=self.tensorboard_dir,
                policy_kwargs=policy_kwargs,
                verbose=1,
                seed=config.get("training.seed"),
                device=self.device,
            )
        
        return self.model
    
    def setup_callbacks(self) -> List[BaseCallback]:
        """Setup training callbacks"""
        
        callbacks = []
        
        # Graceful exit callback
        graceful_exit = GracefulExitCallback()
        callbacks.append(graceful_exit)
        
        # Checkpoint callback
        checkpoint_callback = CheckpointCallback(
            save_freq=self.save_freq // self.n_envs,
            save_path=os.path.join(self.model_dir, "checkpoints"),
            name_prefix="rl_model",
            save_replay_buffer=True,
            save_vecnormalize=True,
        )
        callbacks.append(checkpoint_callback)
        
        # Evaluation callback
        if self.eval_freq > 0:
            eval_env = self.setup_environment(eval_mode=True)
            eval_callback = EvalCallback(
                eval_env,
                best_model_save_path=os.path.join(self.model_dir, "best_model"),
                log_path=os.path.join(self.log_dir, "evaluations"),
                eval_freq=self.eval_freq // self.n_envs,
                deterministic=True,
                render=False,
                n_eval_episodes=config.get("evaluation.n_eval_episodes", 10),
            )
            callbacks.append(eval_callback)
        
        return CallbackList(callbacks)
    
    def train(self, resume_from: Optional[str] = None):
        """Start or resume the training process"""
        
        try:
            print("Setting up training environment...")
            self.env = self.setup_environment()
            
            print("Creating model...")
            self.create_model(resume_from)
            
            print("Setting up callbacks...")
            callbacks = self.setup_callbacks()
            
            # Log hyperparameters
            self.logger.log_hyperparameters({
                "n_envs": self.n_envs,
                "total_timesteps": self.total_timesteps,
                "batch_size": self.batch_size,
                "n_epochs": self.n_epochs,
                "learning_rate": self.learning_rate,
                "device": str(self.device),
                **self.config.config
            })
            
            print(f"Starting training for {self.total_timesteps} timesteps...")
            print(f"Using device: {self.device}")
            print(f"Environment: {self.n_envs} parallel instances")
            
            start_time = time.time()
            
            self.model.learn(
                total_timesteps=self.total_timesteps,
                callback=callbacks,
                progress_bar=True,
                reset_num_timesteps=resume_from is None,
            )
            
            training_time = time.time() - start_time
            print(f"Training completed in {training_time:.2f} seconds")
            
            # Save final model
            final_model_path = os.path.join(self.model_dir, "final_model")
            self.model.save(final_model_path)
            if hasattr(self.model, 'env') and self.model.env:
                self.model.env.save(os.path.join(self.model_dir, "final_vecnormalize.pkl"))
            
            print(f"Final model saved to {final_model_path}")
            
        except Exception as e:
            print(f"Error during training: {e}")
            raise
        finally:
            self.logger.close()
            if self.env:
                self.env.close()
    
    def load_model(self, model_path: str, env=None):
        """Load a trained model for inference"""
        if env is None:
            env = self.setup_environment(eval_mode=True)
        
        self.model = PPO.load(model_path, env=env, device=self.device)
        return self.model
    
    def evaluate(self, model_path: Optional[str] = None, n_episodes: int = None):
        """Evaluate a trained model"""
        
        if n_episodes is None:
            n_episodes = config.get("evaluation.n_eval_episodes", 10)
        
        if model_path:
            self.load_model(model_path)
        
        if self.model is None:
            raise ValueError("No model loaded. Train or load a model first.")
        
        if self.eval_env is None:
            self.eval_env = self.setup_environment(eval_mode=True)
        
        episode_rewards = []
        episode_lengths = []
        
        try:
            for episode in range(n_episodes):
                obs = self.eval_env.reset()
                done = False
                episode_reward = 0
                episode_length = 0
                
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, done, info = self.eval_env.step(action)
                    episode_reward += reward
                    episode_length += 1
                
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                
                print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Length = {episode_length}")
        
        except KeyboardInterrupt:
            print("Evaluation interrupted by user")
        
        finally:
            if self.eval_env:
                self.eval_env.close()
        
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        mean_length = np.mean(episode_lengths)
        
        print(f"\nEvaluation over {len(episode_rewards)} episodes:")
        print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
        print(f"Mean episode length: {mean_length:.1f}")
        
        # Log evaluation results
        self.logger.log_evaluation(
            episode=0,  # Will be updated by caller
            mean_reward=mean_reward,
            std_reward=std_reward,
            metrics={"mean_length": mean_length}
        )
        
        return {
            "mean_reward": mean_reward,
            "std_reward": std_reward,
            "mean_length": mean_length,
            "episode_rewards": episode_rewards,
            "episode_lengths": episode_lengths
        }
    
    def benchmark(self, model_path: Optional[str] = None, n_episodes: int = 100):
        """Run extensive benchmarking"""
        print(f"Running benchmark with {n_episodes} episodes...")
        
        results = self.evaluate(model_path, n_episodes)
        
        # Save benchmark results
        benchmark_file = os.path.join(self.log_dir, "benchmark_results.json")
        import json
        with open(benchmark_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"Benchmark results saved to {benchmark_file}")
        return results


def main():
    """Main entry point for training"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Rocket League bot")
    parser.add_argument("--config", type=str, default="config.yaml", help="Configuration file")
    parser.add_argument("--resume", type=str, help="Resume training from checkpoint")
    parser.add_argument("--evaluate", type=str, help="Evaluate model at path")
    parser.add_argument("--benchmark", type=str, help="Run benchmark on model")
    parser.add_argument("--n-episodes", type=int, default=10, help="Number of evaluation episodes")
    
    args = parser.parse_args()
    
    # Load configuration
    config = Config(args.config)
    
    # Create trainer
    trainer = RLGymTrainer(config)
    
    if args.evaluate:
        trainer.evaluate(args.evaluate, args.n_episodes)
    elif args.benchmark:
        trainer.benchmark(args.benchmark, args.n_episodes)
    elif args.resume:
        trainer.train(resume_from=args.resume)
    else:
        trainer.train()


if __name__ == "__main__":
    main()

import os
import time
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from rlgym_compat import GameState
from rlgym.utils.terminal_conditions.common_conditions import TimeoutCondition, GoalScoredCondition
from rlgym.utils.action_parsers import DiscreteAction
from rlgym.utils.obs_builders import DefaultObs
from rlgym.utils.reward_functions import DefaultReward
from rlgym.utils.state_setters import DefaultState
from rlgym_tools.sb3_utils import SB3MultipleInstanceEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import VecNormalize, VecCheckNan
from stable_baselines3.common.monitor import Monitor

from src.agent import Agent
from src.rewards import CombinedReward
from src.obs.advanced_obs import AdvancedObs


class RLGymTrainer:
    """
    Training manager for the Rocket League bot using PPO
    """

    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Setup directories
        self.log_dir = config.get("log_dir", "logs")
        self.model_dir = config.get("model_dir", "models")
        self.tensorboard_dir = config.get("tensorboard_dir", "tensorboard_logs")
        
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.tensorboard_dir, exist_ok=True)
        
        # Training parameters
        self.n_envs = config.get("n_envs", 4)
        self.total_timesteps = config.get("total_timesteps", 1_000_000)
        self.batch_size = config.get("batch_size", 2048)
        self.n_epochs = config.get("n_epochs", 10)
        self.learning_rate = config.get("learning_rate", 3e-4)
        
        # Setup environment
        self.env = None
        self.model = None
        self.writer = None
        
    def setup_environment(self):
        """Setup the training environment"""
        
        # Environment configuration
        env_config = {
            "reward_fn": CombinedReward(),
            "obs_builder": AdvancedObs(),
            "action_parser": DiscreteAction(),
            "terminal_conditions": [TimeoutCondition(300), GoalScoredCondition()],
            "state_setter": DefaultState(),
            "team_size": 1,
            "spawn_opponents": True,
        }
        
        # Create multiple instances for parallel training
        self.env = SB3MultipleInstanceEnv(
            match_config=env_config,
            num_instances=self.n_envs,
            wait_time=15
        )
        
        # Add monitoring and normalization
        self.env = VecCheckNan(self.env, raise_exception=True)
        self.env = VecNormalize(self.env, norm_obs=True, norm_reward=True, clip_obs=10.0)
        
        return self.env
    
    def create_model(self):
        """Create the PPO model"""
        
        policy_kwargs = dict(
            net_arch=[dict(pi=[512, 512, 256], vf=[512, 512, 256])],
            activation_fn=torch.nn.ReLU,
        )
        
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
            seed=None,
            device=self.device,
        )
        
        return self.model
    
    def setup_callbacks(self):
        """Setup training callbacks"""
        
        # Checkpoint callback
        checkpoint_callback = CheckpointCallback(
            save_freq=100_000 // self.n_envs,
            save_path=os.path.join(self.model_dir, "checkpoints"),
            name_prefix="rl_model",
            save_replay_buffer=True,
            save_vecnormalize=True,
        )
        
        # Evaluation callback
        eval_env = self.setup_environment()
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=os.path.join(self.model_dir, "best_model"),
            log_path=os.path.join(self.log_dir, "evaluations"),
            eval_freq=50_000 // self.n_envs,
            deterministic=True,
            render=False,
            n_eval_episodes=10,
        )
        
        return [checkpoint_callback, eval_callback]
    
    def train(self):
        """Start the training process"""
        
        print("Setting up training environment...")
        self.setup_environment()
        
        print("Creating model...")
        self.create_model()
        
        print("Setting up callbacks...")
        callbacks = self.setup_callbacks()
        
        print(f"Starting training for {self.total_timesteps} timesteps...")
        print(f"Using device: {self.device}")
        
        start_time = time.time()
        
        self.model.learn(
            total_timesteps=self.total_timesteps,
            callback=callbacks,
            progress_bar=True,
        )
        
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        
        # Save final model
        final_model_path = os.path.join(self.model_dir, "final_model")
        self.model.save(final_model_path)
        self.env.save(os.path.join(self.model_dir, "final_vecnormalize.pkl"))
        
        print(f"Final model saved to {final_model_path}")
        
    def load_model(self, model_path: str):
        """Load a trained model"""
        self.model = PPO.load(model_path)
        return self.model
    
    def evaluate(self, n_episodes=10):
        """Evaluate the trained model"""
        
        if self.model is None:
            raise ValueError("No model loaded. Train or load a model first.")
        
        episode_rewards = []
        
        for episode in range(n_episodes):
            obs = self.env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, info = self.env.step(action)
                episode_reward += reward
            
            episode_rewards.append(episode_reward)
        
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        
        print(f"Evaluation over {n_episodes} episodes:")
        print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
        
        return mean_reward, std_reward


if __name__ == "__main__":
    # Example training configuration
    config = {
        "n_envs": 4,
        "total_timesteps": 1_000_000,
        "batch_size": 2048,
        "n_epochs": 10,
        "learning_rate": 3e-4,
        "log_dir": "logs",
        "model_dir": "models",
        "tensorboard_dir": "tensorboard_logs",
    }
    
    trainer = RLGymTrainer(config)
    trainer.train()

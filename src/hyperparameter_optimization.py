import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import optuna
import torch
from stable_baselines3 import A2C, DQN, PPO, SAC, TD3
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from config import config
from training_env import make_training_env
from utils import create_model
from logger import logger # Moved here

try:
    import plotly.graph_objects as go
    from optuna.visualization import plot_optimization_history, plot_param_importances
except ImportError:
    go = None
    plot_optimization_history = None
    plot_param_importances = None
    logger.warning("Plotly or Optuna visualization not available, plotting functions will be skipped.")

SUPPORTED_AGENT_TYPES = ["PPO", "SAC", "TD3", "A2C", "DQN"]


class HyperparameterOptimizer:
    """
    Hyperparameter optimizer using Optuna to find optimal hyperparameters for RL agents.
    """
    
    def __init__(self, agent_type: str = "PPO", n_trials: int = 100, study_name: Optional[str] = None):
        """
        Initializes the HyperparameterOptimizer.
        
        Args:
            agent_type (str): The type of agent to optimize (e.g., "PPO", "SAC").
            n_trials (int): The number of optimization trials to run.
            study_name (Optional[str]): A custom name for the Optuna study.
        """
        self.agent_type = agent_type
        self.n_trials = n_trials
        self.study_name = study_name or f"{agent_type}_optimization"
        self.SUPPORTED_AGENT_TYPES = ["PPO", "SAC", "TD3", "A2C", "DQN"]
        
        # Create study directory
        self.study_dir = Path(f"studies/{self.study_name}")
        self.study_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Optuna study
        self.study = optuna.create_study(
            direction="maximize",
            study_name=self.study_name,
            storage=f"sqlite:///{self.study_dir}/study.db",
            load_if_exists=True
        )
        
        logger.info(f"Initialized hyperparameter optimization for {agent_type}")
        logger.info(f"Study name: {self.study_name}")
        logger.info(f"Number of trials: {n_trials}")
    
    def suggest_hyperparameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Suggests hyperparameters for the current Optuna trial based on the agent type.
        
        Args:
            trial (optuna.Trial): The current Optuna trial object.
            
        Returns:
            Dict[str, Any]: A dictionary of suggested hyperparameters.
            
        Raises:
            ValueError: If an unsupported agent type is provided.
        """
        
        self.SUPPORTED_AGENT_TYPES = ["PPO", "SAC", "TD3", "A2C", "DQN"]

        if self.agent_type not in self.SUPPORTED_AGENT_TYPES:
            raise ValueError(f"Unsupported agent type: {self.agent_type}. "
                             f"Supported types: {', '.join(SUPPORTED_AGENT_TYPES)}")

        hyperparams = {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
            "hidden_size": trial.suggest_categorical("hidden_size", [128, 256, 512, 1024]),
            "n_layers": trial.suggest_int("n_layers", 1, 4)
        }

        if self.agent_type == "PPO":
            hyperparams.update({
                "n_steps": trial.suggest_categorical("n_steps", [512, 1024, 2048, 4096]),
                "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
                "n_epochs": trial.suggest_int("n_epochs", 4, 20),
                "gamma": trial.suggest_float("gamma", 0.9, 0.999),
                "gae_lambda": trial.suggest_float("gae_lambda", 0.8, 0.99),
                "clip_range": trial.suggest_float("clip_range", 0.1, 0.4),
                "ent_coef": trial.suggest_float("ent_coef", 0.0, 0.1),
                "vf_coef": trial.suggest_float("vf_coef", 0.1, 1.0),
                "max_grad_norm": trial.suggest_float("max_grad_norm", 0.1, 2.0),
            })
        elif self.agent_type == "SAC":
            hyperparams.update({
                "buffer_size": trial.suggest_categorical("buffer_size", [100000, 500000, 1000000]),
                "learning_starts": trial.suggest_int("learning_starts", 100, 1000),
                "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256, 512]),
                "tau": trial.suggest_float("tau", 0.001, 0.02),
                "gamma": trial.suggest_float("gamma", 0.9, 0.999),
                "target_update_interval": trial.suggest_int("target_update_interval", 1, 10),
            })
        elif self.agent_type == "TD3":
            hyperparams.update({
                "buffer_size": trial.suggest_categorical("buffer_size", [100000, 500000, 1000000]),
                "learning_starts": trial.suggest_int("learning_starts", 100, 1000),
                "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256, 512]),
                "tau": trial.suggest_float("tau", 0.001, 0.02),
                "gamma": trial.suggest_float("gamma", 0.9, 0.999),
                "policy_delay": trial.suggest_int("policy_delay", 1, 4),
                "noise_clip": trial.suggest_float("noise_clip", 0.1, 1.0),
                "target_policy_noise": trial.suggest_float("target_policy_noise", 0.1, 0.5),
            })
        elif self.agent_type == "A2C":
            hyperparams.update({
                "n_steps": trial.suggest_categorical("n_steps", [5, 10, 20, 50]),
                "gamma": trial.suggest_float("gamma", 0.9, 0.999),
                "ent_coef": trial.suggest_float("ent_coef", 0.0, 0.1),
                "vf_coef": trial.suggest_float("vf_coef", 0.1, 1.0),
                "max_grad_norm": trial.suggest_float("max_grad_norm", 0.1, 2.0),
            })
        elif self.agent_type == "DQN":
            hyperparams.update({
                "buffer_size": trial.suggest_categorical("buffer_size", [50000, 100000, 200000]),
                "learning_starts": trial.suggest_int("learning_starts", 500, 2000),
                "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
                "target_update_interval": trial.suggest_int("target_update_interval", 100, 1000),
                "exploration_fraction": trial.suggest_float("exploration_fraction", 0.05, 0.3),
                "exploration_initial_eps": trial.suggest_float("exploration_initial_eps", 0.8, 1.0),
                "exploration_final_eps": trial.suggest_float("exploration_final_eps", 0.01, 0.1),
            })
        return hyperparams
    
    
    
    def objective(self, trial: optuna.Trial) -> float:
        """
        Objective function for Optuna optimization.
        Trains an agent with suggested hyperparameters and returns its mean reward.
        
        Args:
            trial (optuna.Trial): The current Optuna trial object.
            
        Returns:
            float: The mean reward achieved by the agent during evaluation.
        """
        
        try:
            # Get hyperparameters for this trial
            hyperparams = self.suggest_hyperparameters(trial)
            
            # Create environment
            def make_env():
                env = make_training_env(
                    max_steps=config.env.max_steps,
                    difficulty=config.env.difficulty
                )
                env = Monitor(env)
                return env
            
            env = DummyVecEnv([make_env for _ in range(2)])  # Use 2 envs for optimization
            
            # Normalize observations
            env = VecNormalize(env, norm_obs=True, norm_reward=False)
            
            # Create evaluation environment
            eval_env = DummyVecEnv([make_env for _ in range(1)])
            eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False)
            
            # Create model
            model = create_model(self.agent_type, env, self.study_dir)
            
            # Create evaluation callback
            eval_callback = EvalCallback(
                eval_env,
                best_model_save_path=None,  # Don't save during optimization
                log_path=None,
                eval_freq=5000,
                n_eval_episodes=5,
                deterministic=True,
                render=False
            )
            
            # Train for a shorter time during optimization
            training_steps = min(50000, config.training.total_timesteps // 20)
            
            logger.info(f"Trial {trial.number}: Training for {training_steps} steps")
            
            # Train the model
            model.learn(
                total_timesteps=training_steps,
                callback=eval_callback,
                progress_bar=False
            )
            
            # Get final evaluation score
            mean_reward = eval_callback.best_mean_reward
            
            # Clean up
            env.close()
            eval_env.close()
            
            # Log trial results
            logger.info(f"Trial {trial.number}: Mean reward = {mean_reward:.2f}")
            
            return mean_reward if mean_reward is not None else -1000.0
            
        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {str(e)}")
            logger.exception(e) # Log traceback
            return -1000.0  # Return very low score for failed trials
    
    def optimize(self) -> optuna.Trial:
        """
        Runs the hyperparameter optimization.
        
        Returns:
            optuna.Trial: The best trial found during optimization.
        """
        
        logger.info(f"Starting hyperparameter optimization for {self.agent_type}")
        logger.info(f"Running {self.n_trials} trials...")
        
        start_time = time.time()
        
        # Run optimization
        self.study.optimize(
            self.objective,
            n_trials=self.n_trials,
            show_progress_bar=True
        )
        
        optimization_time = time.time() - start_time
        
        # Save results
        self.save_results(optimization_time)
        
        logger.info(f"Optimization completed in {optimization_time:.1f} seconds")
        logger.info(f"Best trial: {self.study.best_trial.number}")
        logger.info(f"Best value: {self.study.best_trial.value:.2f}")
        
        return self.study.best_trial
    
    def save_results(self, optimization_time: float):
        """
        Saves the optimization results to files.
        
        Args:
            optimization_time (float): The total time taken for optimization.
        """
        
        # Save best parameters
        best_params = self.study.best_trial.params
        best_params["best_value"] = self.study.best_trial.value
        
        with open(self.study_dir / "best_parameters.json", "w", encoding='utf-8') as f:
            json.dump(best_params, f, indent=2)
        
        # Save study summary
        study_summary = {
            "study_name": self.study_name,
            "agent_type": self.agent_type,
            "n_trials": self.n_trials,
            "optimization_time_seconds": optimization_time,
            "best_trial": {
                "number": self.study.best_trial.number,
                "value": self.study.best_trial.value,
                "params": self.study.best_trial.params
            },
            "all_trials": []
        }
        
        for trial in self.study.trials:
            if trial.value is not None:
                study_summary["all_trials"].append({
                    "number": trial.number,
                    "value": trial.value,
                    "params": trial.params
                })
        
        with open(self.study_dir / "study_summary.json", "w", encoding='utf-8') as f:
            json.dump(study_summary, f, indent=2)
        
        # Save Optuna study visualization
        try:
            
            # Optimization history
            fig = plot_optimization_history(self.study)
            fig.write_html(str(self.study_dir / "optimization_history.html"))
            
            # Parameter importances
            fig = plot_param_importances(self.study)
            fig.write_html(str(self.study_dir / "param_importances.html"))
            
        except ImportError:
            logger.warning("Plotly not available, skipping visualization")
        except Exception as e:
            logger.error(f"Failed to create visualizations: {e}")
            logger.exception(e) # Log traceback
        
        logger.info(f"Results saved to {self.study_dir}")


def main():
    """Main function for hyperparameter optimization"""
    
    parser = argparse.ArgumentParser(description="Optimize hyperparameters for Rocket League agents")
    parser.add_argument("--agent", type=str, default="PPO", 
                       choices=SUPPORTED_AGENT_TYPES,
                       help="Agent type to optimize")
    parser.add_argument("--trials", type=int, default=100,
                       help="Number of optimization trials")
    parser.add_argument("--study-name", type=str, default=None,
                       help="Name for the optimization study")
    
    args = parser.parse_args()
    
    try:
        # Create optimizer
        optimizer = HyperparameterOptimizer(
            agent_type=args.agent,
            n_trials=args.trials,
            study_name=args.study_name
        )
        
        # Run optimization
        best_trial = optimizer.optimize()
        
        logger.info("\n🎯 Optimization completed!")
        logger.info(f"Best trial: {best_trial.number}")
        logger.info(f"Best mean reward: {best_trial.value:.2f}")
        logger.info("Best parameters:")
        for param, value in best_trial.params.items():
            logger.info(f"  {param}: {value}")
        
        return 0
        
    except Exception as e:
        logger.exception(f"Optimization failed: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

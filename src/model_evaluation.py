#!/usr/bin/env python3
"""
Model Evaluation and Comparison Script
Comprehensive evaluation of trained Rocket League agents
"""

import argparse
import json
import os
import sys # Moved
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from stable_baselines3 import A2C, DQN, PPO, SAC, TD3
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from config import config
from logger import logger
from training_env import make_training_env


class ModelEvaluator:
    """Comprehensive model evaluator for Rocket League agents"""
    
    def __init__(self, models_dir="models", results_dir="evaluation_results"):
        self.models_dir = Path(models_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Supported agent types
        self.agent_types = ["PPO", "SAC", "TD3", "A2C", "DQN"]
        
        # Evaluation metrics
        self.metrics = [
            "mean_reward", "std_reward", "min_reward", "max_reward",
            "mean_episode_length", "std_episode_length",
            "success_rate", "efficiency", "consistency"
        ]
        
        logger.info("Initialized ModelEvaluator")
        logger.info(f"Models directory: {self.models_dir}")
        logger.info(f"Results directory: {self.results_dir}")
    
    def discover_models(self) -> Dict[str, List[str]]:
        """Discover all available trained models"""
        models = {}
        
        for agent_type in self.agent_types:
            agent_dir = self.models_dir / agent_type
            if agent_dir.exists():
                model_files = []
                
                # Look for model files
                for model_file in agent_dir.rglob("*.zip"):
                    if "final_model" in str(model_file) or "best_model" in str(model_file):
                        model_files.append(str(model_file))
                
                if model_files:
                    models[agent_type] = model_files
        
        logger.info(f"Discovered {sum(len(files) for files in models.values())} models")
        for agent_type, files in models.items():
            logger.info(f"  {agent_type}: {len(files)} models")
        
        return models
    
    def load_model(self, model_path: str, agent_type: str):
        """Load a trained model"""
        try:
            agent_loaders = {
                "PPO": PPO.load,
                "SAC": SAC.load,
                "TD3": TD3.load,
                "A2C": A2C.load,
                "DQN": DQN.load,
            }
            if agent_type in agent_loaders:
                return agent_loaders[agent_type](model_path)
            else:
                raise ValueError(f"Unsupported agent type: {agent_type}")
        except Exception as e:
            logger.error(f"Failed to load model {model_path}: {str(e)}")
            return None
    
    def create_eval_env(self, n_envs: int = 4):
        """Create evaluation environment"""
        def make_env():
            env = make_training_env(
                max_steps=config.env.max_steps,
                difficulty=config.env.difficulty
            )
            env = Monitor(env)
            return env
        
        env = DummyVecEnv([make_env for _ in range(n_envs)])
        
        # Normalize observations if enabled
        if hasattr(config.env, 'observation_normalization') and config.env.observation_normalization:
            env = VecNormalize(env, norm_obs=True, norm_reward=False)
        
        return env
    
    def evaluate_model(self, model_path: str, agent_type: str, 
                      n_episodes: int = 100, n_envs: int = 4) -> Dict[str, Any]:
        """Evaluate a single model"""
        
        logger.info(f"Evaluating {agent_type} model: {model_path}")
        
        try:
            # Load model
            model = self.load_model(model_path, agent_type)
            if model is None:
                return None
            
            # Create environment
            env = self.create_eval_env(n_envs)
            
            # Evaluation parameters
            eval_freq = max(1, n_episodes // 10)  # Evaluate every 10% of episodes
            
            # Create evaluation callback
            eval_callback = EvalCallback(
                env,
                best_model_save_path=None,
                log_path=None,
                eval_freq=eval_freq,
                n_eval_episodes=min(10, n_episodes // n_envs),
                deterministic=True,
                render=False
            )
            
            # Run evaluation
            start_time = time.time()
            
            # Simulate episodes by running the environment
            episode_rewards = []
            episode_lengths = []
            episode_successes = []
            
            for episode in range(0, n_episodes, n_envs):
                obs = env.reset()
                episode_reward = np.zeros(n_envs)
                episode_length = np.zeros(n_envs)
                episode_done = np.zeros(n_envs, dtype=bool)
                
                step = 0
                done = np.zeros(n_envs, dtype=bool) # Initialize done
                while step < config.env.max_steps and not np.all(episode_done):
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, done_step, truncated, info = env.step(action) # Renamed done to done_step
                    
                    # Update episode tracking
                    episode_reward += reward
                    episode_length += 1
                    episode_done = done_step | truncated # Use done_step
                    
                    step += 1
                
                # Record episode results
                for i in range(n_envs):
                    episode_rewards.append(episode_reward[i])
                    episode_lengths.append(episode_length[i])
                    # Consider episode successful if reward > 0
                    episode_successes.append(episode_reward[i] > 0)
            
            evaluation_time = time.time() - start_time
            
            # Calculate metrics
            metrics = {
                "model_path": model_path,
                "agent_type": agent_type,
                "n_episodes": n_episodes,
                "evaluation_time": evaluation_time,
                "mean_reward": np.mean(episode_rewards),
                "std_reward": np.std(episode_rewards),
                "min_reward": np.min(episode_rewards),
                "max_reward": np.max(episode_rewards),
                "mean_episode_length": np.mean(episode_lengths),
                "std_episode_length": np.std(episode_lengths),
                "success_rate": np.mean(episode_successes),
                "efficiency": np.mean(episode_rewards) / np.mean(episode_lengths) if np.mean(episode_lengths) > 0 else 0,
                "consistency": 1.0 - (
                    np.std(episode_rewards) / (abs(np.mean(episode_rewards)) + 1e-8)
                ),
                "episode_rewards": episode_rewards,
                "episode_lengths": episode_lengths,
                "episode_successes": episode_successes
            }
            
            # Clean up
            env.close()
            
            logger.info(f"Evaluation completed: Mean reward = {metrics['mean_reward']:.2f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Evaluation failed for {model_path}: {str(e)}")
            return None
    
    def evaluate_all_models(self, n_episodes: int = 100, n_envs: int = 4) -> Dict[str, Any]:
        """Evaluate all discovered models"""
        
        logger.info("Starting comprehensive model evaluation")
        
        # Discover models
        models = self.discover_models()
        if not models:
            logger.warning("No models found for evaluation")
            return {}
        
        # Evaluate each model
        results = {}
        total_models = sum(len(files) for files in models.values())
        evaluated_models = 0
        
        for agent_type, model_files in models.items():
            results[agent_type] = {}
            
            for model_path in model_files:
                logger.info(f"Evaluating model {evaluated_models + 1}/{total_models}")
                
                metrics = self.evaluate_model(
                    model_path, agent_type, n_episodes, n_envs
                )
                
                if metrics:
                    model_name = Path(model_path).stem
                    results[agent_type][model_name] = metrics
                
                evaluated_models += 1
        
        # Save results
        self.save_evaluation_results(results)
        
        # Generate summary
        summary = self.generate_summary(results)
        
        logger.info("Model evaluation completed")
        return results
    
    def generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate evaluation summary"""
        
        summary = {
            "total_models": sum(len(agent_results) for agent_results in results.values()),
            "agent_types": list(results.keys()),
            "best_models": {},
            "comparison": {}
        }
        
        # Find best model for each agent type
        for agent_type, agent_results in results.items():
            if agent_results:
                # Find best model by mean reward
                best_model = max(agent_results.values(), key=lambda x: x['mean_reward'])
                summary["best_models"][agent_type] = {
                    "model_name": Path(best_model['model_path']).stem,
                    "mean_reward": best_model['mean_reward'],
                    "success_rate": best_model['success_rate'],
                    "efficiency": best_model['efficiency']
                }
        
        # Generate comparison metrics
        all_metrics = []
        for agent_results in results.values():
            for model_metrics in agent_results.values():
                all_metrics.append({
                    'agent_type': model_metrics['agent_type'],
                    'mean_reward': model_metrics['mean_reward'],
                    'success_rate': model_metrics['success_rate'],
                    'efficiency': model_metrics['efficiency'],
                    'consistency': model_metrics['consistency']
                })
        
        if all_metrics:
            df = pd.DataFrame(all_metrics)
            summary["comparison"] = {
                "overall_best": df.loc[df['mean_reward'].idxmax()].to_dict(),
                "most_consistent": df.loc[df['consistency'].idxmax()].to_dict(),
                "most_efficient": df.loc[df['efficiency'].idxmax()].to_dict(),
                "agent_type_performance": df.groupby('agent_type')['mean_reward'].agg(['mean', 'std']).to_dict()
            }
        
        return summary
    
    def save_evaluation_results(self, results: Dict[str, Any]):
        """Save evaluation results to files"""
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        results_file = self.results_dir / f"evaluation_results_{timestamp}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save summary
        summary = self.generate_summary(results)
        summary_file = self.results_dir / f"evaluation_summary_{timestamp}.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Save CSV for analysis
        csv_data = []
        for agent_type, agent_results in results.items():
            for model_name, metrics in agent_results.items():
                row = {
                    'agent_type': agent_type,
                    'model_name': model_name,
                    'model_path': metrics['model_path'],
                    **{k: v for k, v in metrics.items() if k not in [
                        'episode_rewards', 'episode_lengths', 'episode_successes']}
                }
                csv_data.append(row)
        
        if csv_data:
            df = pd.DataFrame(csv_data)
            csv_file = self.results_dir / f"evaluation_results_{timestamp}.csv"
            df.to_csv(csv_file, index=False)
        
        logger.info(f"Results saved to {self.results_dir}")
        logger.info(f"  Detailed results: {results_file}")
        logger.info(f"  Summary: {summary_file}")
        if csv_data:
            logger.info(f"  CSV data: {csv_file}")
    
    def create_visualizations(self, results: Dict[str, Any]):
        """Create visualization plots"""
        
        try:
            # Prepare data for plotting
            plot_data = []
            for agent_type, agent_results in results.items():
                for model_name, metrics in agent_results.items():
                    plot_data.append({
                        'agent_type': agent_type,
                        'model_name': model_name,
                        'mean_reward': metrics['mean_reward'],
                        'success_rate': metrics['success_rate'],
                        'efficiency': metrics['efficiency'],
                        'consistency': metrics['consistency']
                    })
            
            if not plot_data:
                logger.warning("No data available for visualization")
                return
            
            df = pd.DataFrame(plot_data)
            
            # Set up plotting style
            plt.style.use('seaborn-v0_8')
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Model Evaluation Results', fontsize=16, fontweight='bold')
            
            # 1. Mean reward comparison
            ax1 = axes[0, 0]
            sns.boxplot(data=df, x='agent_type', y='mean_reward', ax=ax1)
            ax1.set_title('Mean Reward by Agent Type')
            ax1.set_xlabel('Agent Type')
            ax1.set_ylabel('Mean Reward')
            ax1.tick_params(axis='x', rotation=45)
            
            # 2. Success rate comparison
            ax2 = axes[0, 1]
            sns.barplot(data=df, x='agent_type', y='success_rate', ax=ax2)
            ax2.set_title('Success Rate by Agent Type')
            ax2.set_xlabel('Agent Type')
            ax2.set_ylabel('Success Rate')
            ax2.tick_params(axis='x', rotation=45)
            
            # 3. Efficiency comparison
            ax3 = axes[1, 0]
            sns.violinplot(data=df, x='agent_type', y='efficiency', ax=ax3)
            ax3.set_title('Efficiency Distribution by Agent Type')
            ax3.set_xlabel('Agent Type')
            ax3.set_ylabel('Efficiency')
            ax3.tick_params(axis='x', rotation=45)
            
            # 4. Consistency vs Mean Reward scatter
            ax4 = axes[1, 1]
            for agent_type in df['agent_type'].unique():
                agent_data = df[df['agent_type'] == agent_type]
                ax4.scatter(agent_data['consistency'], agent_data['mean_reward'], 
                           label=agent_type, alpha=0.7, s=100)
            
            ax4.set_title('Consistency vs Mean Reward')
            ax4.set_xlabel('Consistency')
            ax4.set_ylabel('Mean Reward')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save plot
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            plot_file = self.results_dir / f"evaluation_plots_{timestamp}.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Visualizations saved to {plot_file}")
            
        except Exception as e:
            logger.error(f"Failed to create visualizations: {str(e)}")
    
    def print_summary(self, results: Dict[str, Any]):
        """Print evaluation summary to console"""
        
        summary = self.generate_summary(results)
        
        print("\n" + "="*80)
        print("🎯 MODEL EVALUATION SUMMARY")
        print("="*80)
        
        print("\n📊 Total Models Evaluated: {}".format(summary['total_models']))
        print(f"🤖 Agent Types: {', '.join(summary['agent_types'])}")
        
        print("\n🏆 BEST MODELS BY AGENT TYPE:")
        for agent_type, best in summary['best_models'].items():
            print("  {}: {}".format(agent_type, best['model_name']))
            print(f"    Mean Reward: {best['mean_reward']:.2f}")
            print(f"    Success Rate: {best['success_rate']:.2%}")
            print(f"    Efficiency: {best['efficiency']:.3f}")
        
        if 'comparison' in summary and summary['comparison']:
            print(f"\n🥇 OVERALL BEST:")
            overall_best = summary['comparison']['overall_best']
            print(f"  {overall_best['agent_type']} - Mean Reward: {overall_best['mean_reward']:.2f}")
            
            print(f"\n📈 AGENT TYPE PERFORMANCE:")
            for agent_type, perf in summary['comparison']['agent_type_performance'].items():
                print(f"  {agent_type}: {perf['mean']:.2f} ± {perf['std']:.2f}")
        
        print("\n" + "="*80)


def main():
    """Main evaluation function"""
    
    parser = argparse.ArgumentParser(description="Evaluate trained Rocket League agents")
    parser.add_argument("--models-dir", type=str, default="models",
                       help="Directory containing trained models")
    parser.add_argument("--results-dir", type=str, default="evaluation_results",
                       help="Directory to save evaluation results")
    parser.add_argument("--episodes", type=int, default=100,
                       help="Number of episodes per evaluation")
    parser.add_argument("--envs", type=int, default=4,
                       help="Number of parallel environments")
    parser.add_argument("--no-viz", action="store_true",
                       help="Skip visualization generation")
    
    args = parser.parse_args()
    
    try:
        # Create evaluator
        evaluator = ModelEvaluator(
            models_dir=args.models_dir,
            results_dir=args.results_dir
        )
        
        # Run evaluation
        results = evaluator.evaluate_all_models(
            n_episodes=args.episodes,
            n_envs=args.envs
        )
        
        if results:
            # Print summary
            evaluator.print_summary(results)
            
            # Create visualizations
            if not args.no_viz:
                evaluator.create_visualizations(results)
            
            print("\n✅ Evaluation completed successfully!")
            print(f"📁 Results saved to: {args.results_dir}")
            
        else:
            print("❌ No models were successfully evaluated")
            return 1
        
        return 0
        
    except Exception as e:
        logger.exception(f"Evaluation failed: {str(e)}")
        return 1


import sys

if __name__ == "__main__":
    sys.exit(main())

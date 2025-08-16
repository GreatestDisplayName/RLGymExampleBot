#!/usr/bin/env python3
"""
Complete RLGym Workflow Pipeline
Handles the entire process: Load Map → Train → Export Model → Load Model → Play
"""

import sys
import time
import argparse
import subprocess
import json
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import torch

from logger import logger, timed_operation # Import timed_operation
from train import train_agent, test_agent
from convert_model import convert_sb3_to_pytorch
from agent import Agent, NeuralNetwork # Import NeuralNetwork here
from training_env import make_training_env, get_env_info


class CompleteWorkflow:
    """
    Complete workflow manager for RLGym training and deployment.
    Handles loading environment, training, model conversion, loading, and playing.
    """
    
    def __init__(self, project_root: str = ".."):
        """
        Initializes the CompleteWorkflow manager. 
        
        Args:
            project_root (str): The root directory of the project.
        """
        self.project_root = Path(project_root)
        self.src_dir = self.project_root / "src"
        self.models_dir = self.project_root / "models"
        self.logs_dir = self.project_root / "logs"
        
        # Ensure directories exist
        self.models_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
        
        # Default configuration
        self.default_config = {
            "agent_type": "PPO",
            "timesteps": 100000,
            "save_freq": 10000,
            "test_episodes": 5,
            "render_test": False,
            "auto_convert": True,
            "auto_play": False
        }
        
        logger.info("🚀 RLGym Complete Workflow Manager Initialized!")
        logger.info(f"   Project Root: {self.project_root}")
        logger.info(f"   Models Directory: {self.models_dir}")
        logger.info(f"   Logs Directory: {self.logs_dir}")
    
    @timed_operation("Environment Loading")
    def load_map(self, difficulty: str = "medium", max_steps: int = 1000) -> bool:
        """
        Step 1: Loads and tests the training environment/map.
        
        Args:
            difficulty (str): The difficulty setting for the environment.
            max_steps (int): Maximum steps per episode for the environment.
            
        Returns:
            bool: True if environment loaded successfully, False otherwise.
        """
        logger.info("\n📍 STEP 1: LOADING TRAINING MAP/ENVIRONMENT")
        logger.info("=" * 60)
        
        try:
            env = make_training_env(max_steps=max_steps, difficulty=difficulty)
            obs_space, action_space = get_env_info(env)
            
            logger.info("✅ Environment loaded successfully!")
            logger.info(f"   Difficulty: {difficulty}")
            logger.info(f"   Max Steps: {max_steps}")
            logger.info(f"   Observation Space: {obs_space}")
            logger.info(f"   Action Space: {action_space}")
            
            logger.info("\n🧪 Testing environment with random actions...")
            obs, info = env.reset()
            total_reward = 0.0
            
            for i in range(10):
                action = env.action_space.sample()
                obs, reward, done, truncated, info = env.step(action)
                total_reward += reward
                
                if done or truncated:
                    obs, info = env.reset()
                    total_reward = 0.0
            
            env.close()
            logger.info(f"✅ Environment test completed! Total reward: {total_reward:.3f}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load environment: {e}")
            logger.exception(e)
            return False
    
    def _run_step1_load_map(self, workflow_results: Dict[str, Any]) -> bool:
        logger.info("Starting Step 1: Load Map/Environment")
        if not self.load_map(difficulty="medium", max_steps=1000):
            logger.error("❌ Workflow failed at Step 1: Environment loading failed.")
            return False
        workflow_results['environment_loaded'] = True
        return True
    
    def _run_step2_train_agent(self, config: Dict[str, Any], workflow_results: Dict[str, Any]) -> Optional[str]:
        logger.info("Starting Step 2: Train Agent")
        training_params = {
            "agent_type": config['agent_type'],
            "timesteps": config['timesteps'],
            "save_freq": config['save_freq'],
            "test_after": True
        }
        model_path = self.train_agent(training_params)

        if not model_path:
            logger.error("❌ Workflow failed at Step 2: Training failed.")
            return None
        workflow_results['training_completed'] = True
        workflow_results['model_path'] = model_path
        return model_path
    
    def _run_step3_export_model(self, config: Dict[str, Any], model_path: str, workflow_results: Dict[str, Any]) -> Optional[str]:
        logger.info("Starting Step 3: Export Model")
        converted_path = None
        if config['auto_convert']:
            converted_path = self.export_model(model_path, config['agent_type'])
            if not converted_path:
                logger.error("❌ Workflow failed at Step 3: Model conversion failed.")
                return False # Indicate error
            workflow_results['model_converted'] = True
            workflow_results['converted_path'] = converted_path

            logger.info(f"\n💡 To use the converted model in your bot, update 'model_path' in 'bot.cfg' to: {converted_path}")
            workflow_results['bot_config_update_info'] = f"Update bot.cfg model_path to: {converted_path}"
        return converted_path # Return path or None if skipped
    
    def _run_step4_load_model(self, config: Dict[str, Any], model_path: str, converted_path: Optional[str], workflow_results: Dict[str, Any]) -> Optional[Agent]:
        logger.info("Starting Step 4: Load Model")
        agent = None
        if config['auto_convert'] and converted_path:
            agent = self.load_model(converted_path)
        elif model_path: # Load the SB3 model if not converting
            # Note: Loading SB3 model directly into Agent class might require adjustments
            # For simplicity, assuming Agent can load both .pth and SB3 models if not converted
            agent = self.load_model(model_path)

        if not agent:
            logger.error("❌ Workflow failed at Step 4: Model loading failed.")
            return None
        workflow_results['model_loaded'] = True
        return agent
    
    def _run_step5_play_with_model(self, config: Dict[str, Any], agent: Any, workflow_results: Dict[str, Any]):
        logger.info("Starting Step 5: Play with Model")
        play_results = None
        if config['auto_play']:
            play_results = self.play_with_model(
                agent, 
                n_episodes=config['test_episodes'],
                render=config['render_test']
            )

        if play_results:
            workflow_results['play_testing'] = True
            workflow_results['play_results'] = play_results
        else:
            logger.warning("Play testing skipped or failed.")
            workflow_results['play_testing'] = False

    @timed_operation("Agent Training")
    def train_agent(self, training_params: Dict[str, Any]) -> Optional[str]:
        """
        Step 2: Trains the reinforcement learning agent.
        
        Args:
            agent_type (str): Type of agent to train (e.g., "PPO", "SAC").
            timesteps (int): Total training timesteps.
            save_freq (int): Frequency of saving models during training.
            resume_from (Optional[str]): Path to a model to resume training from.
            test_after (bool): Whether to test the agent after training.
            
        Returns:
            Optional[str]: Path to the trained model if successful, None otherwise.
        """
        agent_type = training_params.get("agent_type", "PPO")
        timesteps = training_params.get("timesteps", 100000)
        save_freq = training_params.get("save_freq", 10000)
        resume_from = training_params.get("resume_from")
        test_after = training_params.get("test_after", True)

        logger.info(f"\n🎯 STEP 2: TRAINING {agent_type} AGENT")
        logger.info("=" * 60)

        try:
            logger.info(f"Starting training with {agent_type} algorithm...")
            logger.info(f"   Total Timesteps: {timesteps:,}")
            logger.info(f"   Save Frequency: {save_freq:,}")
            logger.info(f"   Resume From: {resume_from or 'None'}")
            
            model_path = train_agent(
                agent_type=agent_type,
                total_timesteps=timesteps,
                resume_from=resume_from
            )
            
            logger.info("✅ Training completed successfully!")
            logger.info(f"   Model saved to: {model_path}")
            
            if test_after:
                logger.info("\n🧪 Testing trained agent...")
                test_results = test_agent(
                    model_path=model_path,
                    agent_type=agent_type,
                    n_episodes=5,
                    render=False
                )
                
                logger.info("✅ Testing completed!")
                logger.info(f"   Average reward: {test_results['avg_reward']:.2f}")
                logger.info(f"   Average episode length: {test_results['avg_length']:.1f}")
            
            return model_path
            
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            logger.exception(e)
            return None
    
    @timed_operation("Model Export")
    def export_model(self, sb3_model_path: str, agent_type: str = "PPO") -> Optional[str]:
        """
        Step 3: Exports/converts the trained Stable-Baselines3 model to PyTorch format.
        
        Args:
            sb3_model_path (str): Path to the Stable-Baselines3 model.
            agent_type (str): Type of agent (e.g., "PPO", "SAC").
            
        Returns:
            Optional[str]: Path to the converted PyTorch model if successful, None otherwise.
        """
        logger.info("\n📤 STEP 3: EXPORTING MODEL TO PYTORCH FORMAT")
        logger.info("=" * 60)
        
        try:
            sb3_model_path_obj = Path(sb3_model_path)
            if not sb3_model_path_obj.exists() and not \
               (sb3_model_path_obj.parent / (sb3_model_path_obj.name + ".zip")).exists():
                logger.error(f"❌ Source model not found: {sb3_model_path}")
                return None
            
            model_dir = sb3_model_path_obj.parent
            output_path = model_dir / f"{agent_type}_converted.pth"
            
            logger.info(f"Converting {agent_type} model...")
            logger.info(f"   Source: {sb3_model_path}")
            logger.info(f"   Output: {output_path}")
            
            converted_model = convert_sb3_to_pytorch(
                sb3_model_path, 
                str(output_path), 
                agent_type
            )
            
            logger.info("✅ Model conversion completed successfully!")
            logger.info(f"   PyTorch model saved to: {output_path}")
            
            logger.info("\n🧪 Testing converted model...")
            sample_input = [0.0] * 107  # 107 observation features
            test_output = self._test_converted_model(str(output_path), sample_input)
            
            if test_output is not None:
                logger.info(f"✅ Model test passed! Output shape: {test_output.shape}")
            
            return str(output_path)
            
        except Exception as e:
            logger.error(f"❌ Model conversion failed: {e}")
            logger.exception(e)
            return None
    
    def _test_converted_model(self, model_path: str, test_input: list) -> Optional[Any]:
        """
        Tests the converted PyTorch model with a sample input.
        
        Args:
            model_path (str): Path to the converted PyTorch model.
            test_input (list): A sample input observation.
            
        Returns:
            Optional[Any]: The output of the model if successful, None otherwise.
        """
        try:
            model = NeuralNetwork(len(test_input), 256, 8)
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
            model.eval()
            
            with torch.no_grad():
                test_tensor = torch.FloatTensor(test_input).unsqueeze(0)
                output = model(test_tensor)
            
            return output
            
        except Exception as e:
            logger.error(f"❌ Model test failed: {e}")
            logger.exception(e)
            return None
    
    @timed_operation("Model Loading")
    def load_model(self, model_path: str) -> Optional[Agent]:
        """
        Step 4: Loads the trained model into an agent.
        
        Args:
            model_path (str): Path to the model file.
            
        Returns:
            Optional[Agent]: The loaded agent if successful, None otherwise.
        """
        logger.info("\n📥 STEP 4: LOADING TRAINED MODEL")
        logger.info("=" * 60)
        
        try:
            if not Path(model_path).exists():
                logger.error(f"❌ Model not found: {model_path}")
                return None
            
            logger.info(f"Loading model from: {model_path}")
            
            # Create agent with loaded model (assuming input_size=107)
            agent = Agent(input_size=107, model_path=model_path)
            
            logger.info("✅ Model loaded successfully!")
            logger.info(f"   Model path: {model_path}")
            logger.info(f"   Input size: {agent.input_size}")
            logger.info(f"   Hidden size: {agent.hidden_size}")
            logger.info(f"   Device: {agent.device}")
            
            logger.info("\n🧪 Testing loaded agent...")
            test_obs = np.zeros(agent.input_size) # Use numpy array for test obs
            action = agent.act(test_obs)
            
            logger.info("✅ Agent test passed!")
            logger.info(f"   Test observation shape: {len(test_obs)}")
            logger.info(f"   Action shape: {action.shape}")
            logger.info(f"   Sample action: {action[:3]}...")
            
            return agent
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            logger.exception(e)
            return None
    
    @timed_operation("Play Testing")
    def play_with_model(self, agent: Agent, n_episodes: int = 3, render: bool = False) -> Optional[Dict[str, Any]]:
        """
        Step 5: Plays/tests the loaded model in the training environment.
        
        Args:
            agent (Agent): The agent to test.
            n_episodes (int): Number of episodes to play.
            render (bool): Whether to render the environment during testing.
            
        Returns:
            Optional[Dict[str, Any]]: Dictionary of play testing results if successful, None otherwise.
        """
        logger.info("\n🎮 STEP 5: PLAYING WITH TRAINED MODEL")
        logger.info("=" * 60)
        
        try:
            logger.info("Testing agent in training environment...")
            logger.info(f"   Episodes: {n_episodes}")
            logger.info(f"   Render: {render}")
            
            env = make_training_env(max_steps=1000, difficulty="medium")
            
            total_reward = 0.0
            episode_rewards = []
            
            for episode in range(n_episodes):
                logger.info(f"\n🎯 Episode {episode + 1}/{n_episodes}")
                
                obs, info = env.reset()
                episode_reward = 0.0
                step_count = 0
                
                while True:
                    action = agent.act(obs)
                    
                    obs, reward, done, truncated, info = env.step(action)
                    
                    episode_reward += reward
                    step_count += 1
                    
                    if render:
                        pass # Placeholder for visualization
                    
                    if done or truncated:
                        break
                
                total_reward += episode_reward
                episode_rewards.append(episode_reward)
                
                logger.info(f"   Steps: {step_count}, Reward: {episode_reward:.2f}")
            
            env.close()
            
            avg_reward = total_reward / n_episodes
            best_reward = max(episode_rewards)
            worst_reward = min(episode_rewards)
            
            logger.info("✅ Play testing completed!")
            logger.info(f"   Total episodes: {n_episodes}")
            logger.info(f"   Average reward: {avg_reward:.2f}")
            logger.info(f"   Best reward: {best_reward:.2f}")
            logger.info(f"   Worst reward: {worst_reward:.2f}")
            
            return {
                "episodes": n_episodes,
                "total_reward": total_reward,
                "avg_reward": avg_reward,
                "best_reward": best_reward,
                "worst_reward": worst_reward,
                "episode_rewards": episode_rewards
            }
            
        except Exception as e:
            logger.error(f"❌ Play testing failed: {e}")
            logger.exception(e)
            return None
    
    def run_complete_workflow(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """
        Runs the complete workflow from start to finish.
        
        Args:
            config (Optional[Dict[str, Any]]): Configuration dictionary. If None, uses default.
            
        Returns:
            bool: True if workflow completed successfully, False otherwise.
        """
        if config is None:
            config = self.default_config
        
        logger.info("🚀 STARTING COMPLETE RLGym WORKFLOW")
        logger.info("=" * 80)
        logger.info(f"Configuration: {config}")
        logger.info("=" * 80)
        
        workflow_results = {}
        
        try:
            # Step 1: Load Map/Environment
            if not self._run_step1_load_map(workflow_results):
                return False

            # Step 2: Train Agent
            model_path = self._run_step2_train_agent(config, workflow_results)
            if not model_path:
                return False

            # Step 3: Export Model (if auto_convert is enabled)
            converted_path = self._run_step3_export_model(config, model_path, workflow_results)
            if converted_path is False:  # False indicates an error, None indicates skipped
                return False

            # Step 4: Load Model
            agent = self._run_step4_load_model(config, model_path, converted_path, workflow_results)
            if not agent:
                return False

            # Step 5: Play with Model
            self._run_step5_play_with_model(config, agent, workflow_results)
            
            # Workflow completed successfully
            logger.info("\n🎉 COMPLETE WORKFLOW SUCCESSFULLY COMPLETED!")
            logger.info("=" * 80)
            logger.info("Summary:")
            for step, result in workflow_results.items():
                status = "✅" if result else \
                         ("❌" if result is False else "ℹ️")  # Use info for non-boolean results
                logger.info(f"   {step}: {status} {result if not isinstance(result, bool) else ''}")
            
            # Save workflow results
            results_file = self.models_dir / f"workflow_results_{int(time.time())}.json"
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(workflow_results, f, indent=2)
            
            logger.info(f"\n📁 Workflow results saved to: {results_file}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Workflow failed with error: {e}")
            logger.exception(e)
            return False


def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description="Complete RLGym Workflow Manager")
    parser.add_argument("--agent", type=str, default="PPO", 
                       choices=["PPO", "SAC", "TD3"],
                       help="Agent type to train")
    parser.add_argument("--timesteps", type=int, default=100000,
                       help="Total training timesteps")
    parser.add_argument("--save-freq", type=int, default=10000,
                       help="Save frequency during training")
    parser.add_argument("--test-episodes", type=int, default=5,
                       help="Number of episodes for testing")
    parser.add_argument("--render", action="store_true",
                       help="Render during testing")
    parser.add_argument("--no-convert", action="store_true",
                       help="Skip model conversion step")
    parser.add_argument("--no-play", action="store_true",
                       help="Skip play testing step")
    parser.add_argument("--quick", action="store_true",
                       help="Quick test run (10k timesteps)")
    
    args = parser.parse_args()
    
    workflow = CompleteWorkflow()
    
    config = {
        "agent_type": args.agent,
        "timesteps": 10000 if args.quick else args.timesteps,
        "save_freq": args.save_freq,
        "test_episodes": args.test_episodes,
        "render_test": args.render,
        "auto_convert": not args.no_convert,
        "auto_play": not args.no_play
    }
    
    success = workflow.run_complete_workflow(config)
    
    if success:
        logger.info("\n🎯 Next steps:")
        logger.info("   1. Run 'python run.py' to start the bot in Rocket League")
        logger.info("   2. Or use 'python src/league_manager.py' for league management")
        logger.info("   3. Check logs/ directory for training progress")
        sys.exit(0)
    else:
        logger.error("\n❌ Workflow failed. Check the error messages above.")
        sys.exit(1)
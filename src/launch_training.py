#!/usr/bin/env python3
"""
Enhanced Training Launcher for Rocket League Agents
Provides a user-friendly interface for launching training sessions
"""

import sys
import json
import argparse
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional
import time
from logger import logger # Import logger
from utils import SUPPORTED_AGENT_TYPES


class TrainingLauncher:
    """Enhanced training launcher with configuration management"""
    
    def __init__(self, config_dir: str = "configs"):
        """
        Initializes the TrainingLauncher.
        
        Args:
            config_dir (str): Directory where configuration files are stored.
        """
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        
        # Default configurations
        self.default_configs = {
            "quick_test": {
                "description": "Quick test run for debugging",
                "agent_type": "PPO",
                "timesteps": 10000,
                "save_freq": 1000,
                "n_envs": 2,
                "eval_freq": 2000
            },
            "standard_training": {
                "description": "Standard training run",
                "agent_type": "PPO",
                "timesteps": 1000000,
                "save_freq": 10000,
                "n_envs": 4,
                "eval_freq": 10000
            },
            "extended_training": {
                "description": "Extended training for better performance",
                "agent_type": "PPO",
                "timesteps": 5000000,
                "save_freq": 50000,
                "n_envs": 8,
                "eval_freq": 50000
            },
            "sac_experiment": {
                "description": "SAC algorithm experiment",
                "agent_type": "SAC",
                "timesteps": 1000000,
                "save_freq": 10000,
                "n_envs": 4,
                "eval_freq": 10000
            },
            "td3_experiment": {
                "description": "TD3 algorithm experiment",
                "agent_type": "TD3",
                "timesteps": 1000000,
                "save_freq": 10000,
                "n_envs": 4,
                "eval_freq": 10000
            }
        }
        
        # Initialize default configs
        self._init_default_configs()
    
    def _init_default_configs(self):
        """Initializes default configuration files if they don't exist."""
        for name, config_data in self.default_configs.items():
            config_file = self.config_dir / f"{name}.json"
            if not config_file.exists():
                with open(config_file, 'w', encoding='utf-8') as f:
                    json.dump(config_data, f, indent=2)
                logger.info(f"Initialized default config: {config_file.name}")
    
    def list_configs(self):
        """Lists all available training configurations."""
        logger.info(f"\n📋 Available Training Configurations:")
        logger.info("=" * 60)
        
        # List default configs
        for name, config_data in self.default_configs.items():
            logger.info(f"\n🔧 {name.upper()}")
            logger.info(f"   Description: {config_data['description']}")
            logger.info(f"   Agent: {config_data['agent_type']}")
            logger.info(f"   Timesteps: {config_data['timesteps']:,}")
            logger.info(f"   Environments: {config_data['n_envs']}")
            logger.info(f"   Save Frequency: {config_data['save_freq']:,}")
        
        # List custom configs
        custom_configs = list(self.config_dir.glob("*.json"))
        custom_configs = [f for f in custom_configs if f.stem not in self.default_configs]
        
        if custom_configs:
            logger.info(f"\n📁 Custom Configurations:")
            for config_file in custom_configs:
                try:
                    with open(config_file, 'r', encoding='utf-8') as f:
                        config_data = json.load(f)
                    logger.info(f"\n   🎯 {config_file.stem}")
                    logger.info(f"      Agent: {config_data.get('agent_type', 'Unknown')}")
                    logger.info(f"      Timesteps: {config_data.get('timesteps', 'Unknown'):,}")
                except Exception as e:
                    logger.info(f"   ❌ {config_file.stem} (invalid or unreadable): {e}")
        
        logger.info("\n" + "=" * 60)
    
    def create_config(self, name: str, **kwargs) -> Path:
        """
        Creates a new training configuration file.
        
        Args:
            name (str): The name of the new configuration.
            **kwargs: Configuration parameters to include in the new config.
            
        Returns:
            Path: The path to the newly created configuration file.
        """
        config_data = {
            "description": kwargs.get("description", f"Custom configuration: {name}"),
            "agent_type": kwargs.get("agent_type", "PPO"),
            "timesteps": kwargs.get("timesteps", 1000000),
            "save_freq": kwargs.get("save_freq", 10000),
            "n_envs": kwargs.get("n_envs", 4),
            "eval_freq": kwargs.get("eval_freq", 10000),
            "learning_rate": kwargs.get("learning_rate", 3e-4),
            "batch_size": kwargs.get("batch_size", 64),
            "hidden_size": kwargs.get("hidden_size", 256),
            "n_layers": kwargs.get("n_layers", 2)
        }
        
        config_file = self.config_dir / f"{name}.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2)
        
        logger.info(f"Created configuration: {name}")
        logger.info(f"   File: {config_file}")
        return config_file
    
    def load_config(self, name: str) -> Dict[str, Any]:
        """
        Loads a training configuration from a file.
        
        Args:
            name (str): The name of the configuration to load.
            
        Returns:
            Dict[str, Any]: The loaded configuration dictionary.
            
        Raises:
            FileNotFoundError: If the configuration file is not found.
        """
        config_file = self.config_dir / f"{name}.json"
        
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration '{name}' not found at {config_file}")
        
        with open(config_file, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        
        return config_data
    
    def validate_config(self, config_data: Dict[str, Any]) -> bool:
        """
        Validates a training configuration dictionary.
        
        Args:
            config_data (Dict[str, Any]): The configuration dictionary to validate.
            
        Returns:
            bool: True if the configuration is valid, False otherwise.
        """
        required_fields = ["agent_type", "timesteps", "save_freq", "n_envs"]
        
        for field_name in required_fields:
            if field_name not in config_data:
                logger.error(f"❌ Missing required field: {field_name}")
                return False
        
        if config_data["agent_type"] not in SUPPORTED_AGENT_TYPES:
            logger.error(f"❌ Invalid agent type: {config_data['agent_type']}")
            logger.error(f"   Valid types: {', '.join(SUPPORTED_AGENT_TYPES)}")
            return False
        
        if config_data["timesteps"] <= 0:
            logger.error("❌ Timesteps must be positive")
            return False
        
        if config_data["save_freq"] <= 0 or config_data["save_freq"] > config_data["timesteps"]:
            logger.error("❌ Save frequency must be positive and <= timesteps")
            return False
        
        if config_data["n_envs"] <= 0:
            logger.error("❌ Number of environments must be positive")
            return False
        
        logger.info("Configuration validation passed")
        return True
    
    def launch_training(self, config_name: str, resume_from: Optional[str] = None,
                       test_after: bool = False, render_test: bool = False, profile: bool = False) -> bool:
        """Launches a training session with the specified configuration.
        
        Args:
            config_name (str): The name of the configuration to use.
            resume_from (Optional[str]): Path to a model checkpoint to resume training from.
            test_after (bool): Whether to test the agent after training.
            render_test (bool): Whether to render the environment during testing.
            profile (bool): Whether to enable performance profiling.
            
        Returns:
            bool: True if training completed successfully, False otherwise.
        """
        
        
        try:
            config_data = self.load_config(config_name)
            
            if not self.validate_config(config_data):
                return False
            
            logger.info(f"\nLaunching Training Session")
            logger.info("=" * 50)
            logger.info(f"Configuration: {config_name}")
            logger.info(f"Agent Type: {config_data['agent_type']}")
            logger.info(f"Timesteps: {config_data['timesteps']:,}")
            logger.info(f"Environments: {config_data['n_envs']}")
            logger.info(f"Save Frequency: {config_data['save_freq']:,}")
            
            if resume_from:
                logger.info(f"Resuming from: {resume_from}")
            
            if profile:
                logger.info("Performance profiling enabled")

            logger.info("=" * 50)
            
            cmd = [
                sys.executable, "train.py",
                "--agent", config_data["agent_type"],
                "--timesteps", str(config_data["timesteps"])
            ]
            
            if resume_from:
                cmd.extend(["--resume", resume_from])
            
            if test_after:
                cmd.append("--test")
                if render_test:
                    cmd.append("--render")

            if profile:
                cmd.append("--profile")
            
            logger.info(f"\nExecuting: {' '.join(cmd)}")
            logger.info(f"\nTraining started... Press Ctrl+C to stop\n")
            
            start_time = time.time()
            
            try:
                result = subprocess.run(cmd, cwd=Path(__file__).parent, check=True, capture_output=True, text=True)
            except subprocess.CalledProcessError as e:
                error_message = (
                    f"Training script failed with exit code {e.returncode}.\n\n"
                    f"Stdout:\n{e.stdout}\n\nStderr:\n{e.stderr}"
                )
                logger.error(error_message)
                raise RuntimeError(error_message) from e
            
            training_time = time.time() - start_time
            
            if result.returncode == 0:
                logger.info(f"\nTraining completed successfully!")
                logger.info(f"Total time: {training_time:.1f} seconds")
            else:
                logger.error(f"\nTraining failed with exit code: {result.returncode}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to launch training: {str(e)}")
            logger.exception(e) # Log traceback
            raise e
    
    def interactive_config(self):
        """
        Guides the user through creating a new training configuration interactively.
        """
        logger.info(f"\nInteractive Configuration Creation")
        logger.info("=" * 40)
        
        name = input("Configuration name: ").strip()
        if not name:
            logger.error("Configuration name cannot be empty")
            return
        
        if (self.config_dir / f"{name}.json").exists():
            overwrite = input(f"Configuration '{name}' already exists. Overwrite? (y/N): ").strip().lower()
            if overwrite != 'y':
                logger.info("Configuration creation cancelled")
                return
        
        logger.info("\nAvailable agent types: PPO, SAC, TD3, A2C, DQN")
        agent_type = input("Agent type [PPO]: ").strip().upper() or "PPO"
        
        try:
            timesteps = int(input("Total timesteps [1000000]: ").strip() or "1000000")
            save_freq = int(input("Save frequency [10000]: ").strip() or "10000")
            n_envs = int(input("Number of environments [4]: ").strip() or "4")
            learning_rate = float(input("Learning rate [3e-4]: ").strip() or "3e-4")
            batch_size = int(input("Batch size [64]: ").strip() or "64")
            hidden_size = int(input("Hidden layer size [256]: ").strip() or "256")
            n_layers = int(input("Number of layers [2]: ").strip() or "2")
        except ValueError:
            logger.error("Invalid numeric input")
            return
        
        description = input("Description (optional): ").strip()
        
        config_file = self.create_config(
            name=name,
            description=description,
            agent_type=agent_type,
            timesteps=timesteps,
            save_freq=save_freq,
            n_envs=n_envs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            hidden_size=hidden_size,
            n_layers=n_layers
        )
        
        logger.info(f"\nConfiguration '{name}' created successfully!")
        
        launch = input(f"\nLaunch training with this configuration? (y/N): ").strip().lower()
        if launch == 'y':
            self.launch_training(name)


def main():
    """
    Main launcher function for command-line execution.
    """
    parser = argparse.ArgumentParser(description="Enhanced Training Launcher for Rocket League Agents")
    parser.add_argument("--list", action="store_true", help="List available configurations")
    parser.add_argument("--create", action="store_true", help="Create new configuration interactively")
    parser.add_argument("--config", type=str, help="Training configuration to use")
    parser.add_argument("--resume", type=str, help="Resume training from checkpoint")
    parser.add_argument("--test", action="store_true", help="Test agent after training")
    parser.add_argument("--render", action="store_true", help="Render during testing")
    parser.add_argument("--config-dir", type=str, default="configs", help="Configuration directory")
    
    args = parser.parse_args()
    
    launcher = TrainingLauncher(config_dir=args.config_dir)
    
    try:
        if args.list:
            launcher.list_configs()
        
        elif args.create:
            launcher.interactive_config()
        
        elif args.config:
            success = launcher.launch_training(
                config_name=args.config,
                resume_from=args.resume,
                test_after=args.test,
                render_test=args.render
            )
            sys.exit(0 if success else 1)
        
        else:
            if len(sys.argv) == 1:
                logger.info("Enhanced Training Launcher for Rocket League Agents")
                logger.info("=" * 60)
                logger.info("Use --help to see all options")
                logger.info("\nQuick start:")
                logger.info("  python launch_training.py --list                    # List configurations")
                logger.info("  python launch_training.py --create                  # Create new config")
                logger.info("  python launch_training.py --config standard_training # Launch training")
                logger.info("  python launch_training.py --config quick_test --test # Quick test run")
                logger.info("\nFor more help: python launch_training.py --help")
                sys.exit(0)
        
        sys.exit(0)
        
    except KeyboardInterrupt:
        logger.info("\n\nTraining interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"\nError: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    exit(main())
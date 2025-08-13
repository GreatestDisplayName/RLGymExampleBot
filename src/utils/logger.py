import logging
import os
import json
from datetime import datetime
from typing import Dict, Any
import numpy as np
from torch.utils.tensorboard import SummaryWriter


class TrainingLogger:
    """
    Comprehensive logging for training progress and metrics
    """
    
    def __init__(self, log_dir: str, experiment_name: str = None):
        self.log_dir = log_dir
        self.experiment_name = experiment_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create experiment directory
        self.experiment_dir = os.path.join(log_dir, self.experiment_name)
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        # Setup tensorboard
        self.writer = SummaryWriter(os.path.join(self.experiment_dir, "tensorboard"))
        
        # Setup file logging
        self.setup_file_logging()
        
        # Metrics storage
        self.metrics = {
            "episodes": [],
            "rewards": [],
            "losses": [],
            "evaluations": [],
            "hyperparameters": {},
        }
        
    def setup_file_logging(self):
        """Setup file-based logging"""
        log_file = os.path.join(self.experiment_dir, "training.log")
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger("RLGymTrainer")
    
    def log_scalar(self, tag: str, value: float, step: int):
        """Log a scalar value to tensorboard"""
        self.writer.add_scalar(tag, value, step)
    
    def log_histogram(self, tag: str, values: np.ndarray, step: int):
        """Log histogram to tensorboard"""
        self.writer.add_histogram(tag, values, step)
    
    def log_episode(self, episode: int, reward: float, length: int, info: Dict[str, Any] = None):
        """Log episode metrics"""
        self.metrics["episodes"].append({
            "episode": episode,
            "reward": reward,
            "length": length,
            "info": info or {},
            "timestamp": datetime.now().isoformat()
        })
        
        self.log_scalar("episode/reward", reward, episode)
        self.log_scalar("episode/length", length, episode)
        
        if info:
            for key, value in info.items():
                self.log_scalar(f"episode/{key}", value, episode)
    
    def log_training_metrics(self, step: int, loss_dict: Dict[str, float]):
        """Log training metrics"""
        self.metrics["losses"].append({
            "step": step,
            **loss_dict,
            "timestamp": datetime.now().isoformat()
        })
        
        for key, value in loss_dict.items():
            self.log_scalar(f"train/{key}", value, step)
    
    def log_evaluation(self, episode: int, mean_reward: float, std_reward: float, metrics: Dict[str, float] = None):
        """Log evaluation results"""
        eval_data = {
            "episode": episode,
            "mean_reward": mean_reward,
            "std_reward": std_reward,
            "metrics": metrics or {},
            "timestamp": datetime.now().isoformat()
        }
        
        self.metrics["evaluations"].append(eval_data)
        
        self.log_scalar("eval/mean_reward", mean_reward, episode)
        self.log_scalar("eval/std_reward", std_reward, episode)
        
        if metrics:
            for key, value in metrics.items():
                self.log_scalar(f"eval/{key}", value, episode)
    
    def log_hyperparameters(self, params: Dict[str, Any]):
        """Log hyperparameters"""
        self.metrics["hyperparameters"] = params
        
        # Log to tensorboard
        param_str = ", ".join([f"{k}={v}" for k, v in params.items()])
        self.writer.add_text("hyperparameters", param_str, 0)
    
    def save_metrics(self):
        """Save all metrics to JSON file"""
        metrics_file = os.path.join(self.experiment_dir, "metrics.json")
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2, default=str)
    
    def close(self):
        """Close the logger and save final metrics"""
        self.save_metrics()
        self.writer.close()
        self.logger.info("Training logger closed")


# Global logger instance
logger = None


def setup_logger(log_dir: str, experiment_name: str = None):
    """Setup global logger"""
    global logger
    logger = TrainingLogger(log_dir, experiment_name)
    return logger

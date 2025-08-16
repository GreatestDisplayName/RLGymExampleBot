"""
Performance Monitoring and Metrics for RLGym Training Project
"""

import json
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from logger import logger


@dataclass
class TrainingMetrics:
    """Training performance metrics"""
    episode_rewards: deque = field(default_factory=lambda: deque(maxlen=1000))
    episode_lengths: deque = field(default_factory=lambda: deque(maxlen=1000))
    losses: deque = field(default_factory=lambda: deque(maxlen=1000))
    value_losses: deque = field(default_factory=lambda: deque(maxlen=1000))
    policy_losses: deque = field(default_factory=lambda: deque(maxlen=1000))
    entropy: deque = field(default_factory=lambda: deque(maxlen=1000))
    learning_rate: deque = field(default_factory=lambda: deque(maxlen=1000))
    explained_variance: deque = field(default_factory=lambda: deque(maxlen=1000))
    
    def add_episode(self, reward: float, length: int):
        """Add episode metrics"""
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
    
    def add_losses(self, total_loss: float, value_loss: float, policy_loss: float):
        """Add loss metrics"""
        self.losses.append(total_loss)
        self.value_losses.append(value_loss)
        self.policy_losses.append(policy_loss)
    
    def add_auxiliary(self, entropy: float, lr: float, explained_var: float):
        """Add auxiliary metrics"""
        self.entropy.append(entropy)
        self.learning_rate.append(lr)
        self.explained_variance.append(explained_var)
    
    def get_stats(self) -> Dict[str, float]:
        """Get current statistics"""
        stats = {}
        
        if self.episode_rewards:
            rewards = list(self.episode_rewards)
            stats.update({
                'mean_reward': np.mean(rewards),
                'std_reward': np.std(rewards),
                'min_reward': np.min(rewards),
                'max_reward': np.max(rewards),
                'recent_reward': np.mean(list(rewards)[-100:]) if len(rewards) >= 100 else np.mean(rewards)
            })
        
        if self.episode_lengths:
            lengths = list(self.episode_lengths)
            stats.update({
                'mean_length': np.mean(lengths),
                'std_length': np.std(lengths),
                'min_length': np.min(lengths),
                'max_length': np.max(lengths)
            })
        
        if self.losses:
            losses = list(self.losses)
            stats.update({
                'mean_loss': np.mean(losses),
                'std_loss': np.std(losses),
                'min_loss': np.min(losses),
                'max_loss': np.max(losses),
                'recent_loss': np.mean(list(losses)[-100:]) if len(losses) >= 100 else np.mean(losses)
            })
        
        return stats


@dataclass
class MatchMetrics:
    """Match performance metrics"""
    wins: int = 0
    losses: int = 0
    draws: int = 0
    goals_scored: int = 0
    goals_conceded: int = 0
    total_matches: int = 0
    win_rate: float = 0.0
    goal_difference: int = 0
    
    def update_match_result(self, result: str, goals_for: int, goals_against: int):
        """Update metrics after a match"""
        self.total_matches += 1
        self.goals_scored += goals_for
        self.goals_conceded += goals_against
        self.goal_difference = self.goals_scored - self.goals_conceded
        
        if result == "win":
            self.wins += 1
        elif result == "loss":
            self.losses += 1
        else:  # draw
            self.draws += 1
        
        self.win_rate = self.wins / self.total_matches if self.total_matches > 0 else 0.0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get match statistics"""
        return {
            'wins': self.wins,
            'losses': self.losses,
            'draws': self.draws,
            'total_matches': self.total_matches,
            'win_rate': self.win_rate,
            'goals_scored': self.goals_scored,
            'goals_conceded': self.goals_conceded,
            'goal_difference': self.goal_difference,
            'avg_goals_per_match': self.goals_scored / self.total_matches if self.total_matches > 0 else 0.0
        }


class PerformanceMonitor:
    """Main performance monitoring class"""
    
    def __init__(self, save_dir: str = "metrics"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        self.training_metrics = TrainingMetrics()
        self.match_metrics = defaultdict(MatchMetrics)
        self.start_time = time.time()
        self.last_save = time.time()
        self.save_interval = 300  # Save every 5 minutes
        
        # Performance tracking
        self.episode_times = deque(maxlen=100)
        self.step_times = deque(maxlen=1000)
        self.memory_usage = deque(maxlen=100)
        
        logger.info("Performance monitor initialized", save_dir=str(self.save_dir))
    
    def start_episode(self) -> float:
        """Start timing an episode"""
        return time.time()
    
    def end_episode(self, start_time: float, reward: float, length: int):
        """End episode timing and record metrics"""
        episode_time = time.time() - start_time
        self.episode_times.append(episode_time)
        self.training_metrics.add_episode(reward, length)
        
        logger.debug(f"Episode completed: reward={reward:.3f}, length={length}, time={episode_time:.3f}s")
    
    def start_step(self) -> float:
        """Start timing a training step"""
        return time.time()
    
    def end_step(self, start_time: float, losses: Optional[Dict[str, float]] = None):
        """End step timing and record metrics"""
        step_time = time.time() - start_time
        self.step_times.append(step_time)
        
        if losses:
            self.training_metrics.add_losses(
                losses.get('total_loss', 0.0),
                losses.get('value_loss', 0.0),
                losses.get('policy_loss', 0.0)
            )
        
        logger.debug(f"Training step completed in {step_time:.4f}s")
    
    def record_match(self, match_data: Dict[str, Any]):
        """Record match results"""
        player1 = match_data['player1']
        player2 = match_data['player2']
        result = match_data['result']
        goals_for = match_data['goals_for']
        goals_against = match_data['goals_against']

        # Update player1 metrics
        if result == "win":
            self.match_metrics[player1].update_match_result("win", goals_for, goals_against)
            self.match_metrics[player2].update_match_result("loss", goals_against, goals_for)
        elif result == "loss":
            self.match_metrics[player1].update_match_result("loss", goals_for, goals_against)
            self.match_metrics[player2].update_match_result("win", goals_against, goals_for)
        else:  # draw
            self.match_metrics[player1].update_match_result("draw", goals_for, goals_against)
            self.match_metrics[player2].update_match_result("draw", goals_against, goals_for)
        
        logger.info(f"Match recorded: {player1} vs {player2} - {result} ({goals_for}-{goals_against})")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        stats = {
            'uptime': time.time() - self.start_time,
            'training': self.training_metrics.get_stats(),
            'performance': {}
        }
        
        # Performance metrics
        if self.episode_times:
            stats['performance']['avg_episode_time'] = np.mean(self.episode_times)
            stats['performance']['avg_step_time'] = np.mean(self.step_times) if self.step_times else 0.0
        
        # Match statistics
        stats['matches'] = {
            player: metrics.get_stats() 
            for player, metrics in self.match_metrics.items()
        }
        
        return stats
    
    def save_metrics(self, force: bool = False):
        """Save metrics to file"""
        current_time = time.time()
        
        if not force and (current_time - self.last_save) < self.save_interval:
            return
        
        try:
            metrics_data = self.get_performance_stats()
            metrics_data['timestamp'] = current_time
            
            # Save as JSON
            json_path = self.save_dir / f"metrics_{int(current_time)}.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(metrics_data, f, indent=2)
            
            # Save latest metrics
            latest_path = self.save_dir / "metrics_latest.json"
            with open(latest_path, 'w', encoding='utf-8') as f:
                json.dump(metrics_data, f, indent=2)
            
            self.last_save = current_time
            logger.debug(f"Metrics saved to {json_path}")
            
        except Exception as e:
            logger.error(f"Failed to save metrics: {e}")
    
    def log_summary(self):
        """Log a summary of current metrics"""
        stats = self.get_performance_stats()
        
        logger.info("Performance Summary", **stats)
        
        # Training progress
        if stats['training']:
            training = stats['training']
            logger.info(f"Training Progress - Mean Reward: {training.get('mean_reward', 0):.3f}, "
                       f"Recent Reward: {training.get('recent_reward', 0):.3f}")
        
        # Match statistics
        if stats['matches']:
            for player, match_stats in stats['matches'].items():
                logger.info(f"Player {player}: Win Rate: {match_stats['win_rate']:.2%}, "
                           f"Matches: {match_stats['total_matches']}")
    
    def reset(self):
        """Reset all metrics"""
        self.training_metrics = TrainingMetrics()
        self.match_metrics.clear()
        self.episode_times.clear()
        self.step_times.clear()
        self.memory_usage.clear()
        self.start_time = time.time()
        self.last_save = time.time()
        
        logger.info("Performance monitor reset")


class EarlyStopping:
    """Early stopping mechanism for training"""
    
    def __init__(self, patience: int = 100, min_delta: float = 0.001, 
                 monitor: str = 'mean_reward', mode: str = 'max'):
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.should_stop = False
        
        logger.info(f"Early stopping initialized: patience={patience}, monitor={monitor}, mode={mode}")
    
    def __call__(self, current_score: float) -> bool:
        """Check if training should stop"""
        if self.best_score is None:
            self.best_score = current_score
            return False
        
        if self.mode == 'max':
            improved = current_score > self.best_score + self.min_delta
        else:
            improved = current_score < self.best_score - self.min_delta
        
        if improved:
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            logger.info(f"Early stopping triggered after {self.patience} epochs without improvement")
            return True
        
        return False


# Global performance monitor instance
monitor = PerformanceMonitor()

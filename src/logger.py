import logging
import sys
import os
from pathlib import Path
from typing import Optional, Dict, Any, Callable
from datetime import datetime
import json
import numpy as np
from torch.utils.tensorboard import SummaryWriter


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for console output"""
    
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }
    
    def format(self, record: logging.LogRecord) -> str:
        # Store original levelname to avoid modifying the record
        original_levelname = record.levelname
        color = self.COLORS.get(original_levelname)
        if color:
            record.levelname = f"{color}{original_levelname}{self.COLORS['RESET']}"
        
        # Format the message
        formatted_message = super().format(record)
        
        # Restore original levelname
        record.levelname = original_levelname
        
        return formatted_message


class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging"""
    
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add extra fields if present, excluding standard LogRecord attributes
        for key, value in record.__dict__.items():
            if key not in ['name', 'levelname', 'pathname', 'lineno', 'funcName', 'created',
                           'msecs', 'relativeCreated', 'thread', 'threadName', 'processName',
                           'process', 'message', 'args', 'exc_info', 'exc_text', 'stack_info']:
                log_entry[key] = value

        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        if record.stack_info:
            log_entry['stack_info'] = self.formatStack(record.stack_info)
        
        return json.dumps(log_entry)


class RLGymLogger:
    """Main logger class for RLGym project"""
    
    def __init__(self, name: str = "RLGym", log_dir: str = "logs", experiment_name: Optional[str] = None):
        self.name = name
        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = self.log_dir / self.experiment_name
        
        self.log_dir.mkdir(exist_ok=True)
        self.experiment_dir.mkdir(exist_ok=True)
        
        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        
        # Setup Tensorboard
        self.writer = SummaryWriter(str(self.experiment_dir))

        # Metrics storage
        self.metrics: Dict[str, Any] = {
            "episodes": [],
            "rewards": [],
            "losses": [],
            "evaluations": [],
            "hyperparameters": {},
        }
        
        # Prevent duplicate handlers
        if not self.logger.handlers:
            self._setup_handlers()
    
    def _setup_handlers(self):
        """Setup logging handlers"""
        # Console handler with colors
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = ColoredFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        
        # File handler for all logs
        file_handler = logging.FileHandler(
            self.experiment_dir / f"{self.name.lower()}.log", encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(funcName)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        
        # JSON file handler for structured logging
        json_handler = logging.FileHandler(
            self.experiment_dir / f"{self.name.lower()}_structured.json", encoding='utf-8'
        )
        json_handler.setLevel(logging.INFO)
        json_formatter = StructuredFormatter()
        json_handler.setFormatter(json_formatter)
        
        # Add handlers
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(json_handler)
    
    def debug(self, message: str, **kwargs: Any):
        """Log debug message"""
        self.logger.debug(message, extra=kwargs)

    def info(self, message: str, **kwargs: Any):
        """Log info message"""
        self.logger.info(message, extra=kwargs)

    def warning(self, message: str, **kwargs: Any):
        """Log warning message"""
        self.logger.warning(message, extra=kwargs)

    def error(self, message: str, exc_info: bool = False, **kwargs: Any):
        """Log error message"""
        self.logger.error(message, exc_info=exc_info, extra=kwargs)

    def critical(self, message: str, exc_info: bool = False, **kwargs: Any):
        """Log critical message"""
        self.logger.critical(message, exc_info=exc_info, extra=kwargs)

    def exception(self, message: str, **kwargs: Any):
        """Log exception with traceback"""
        self.logger.exception(message, extra=kwargs)

    def log_scalar(self, tag: str, value: float, step: int):
        """Log a scalar value to tensorboard"""
        self.writer.add_scalar(tag, value, step)

    def log_histogram(self, tag: str, values: np.ndarray, step: int):
        """Log histogram to tensorboard"""
        self.writer.add_histogram(tag, values, step)

    def log_hyperparameters(self, params: Dict[str, Any]):
        """Log hyperparameters"""
        self.metrics["hyperparameters"] = params
        param_str = ", ".join([f"{k}={v}" for k, v in params.items()])
        self.writer.add_text("hyperparameters", param_str, 0)

    def training_start(self, algorithm: str, timesteps: int, **kwargs: Any):
        """Log training start with structured data"""
        self.info(f"Starting training with {algorithm}", 
                 algorithm=algorithm, 
                 total_timesteps=timesteps,
                 event_type="training_start",
                 **kwargs)
        self.log_hyperparameters(kwargs)
    
    def training_step(self, step: int, reward: float, loss: Optional[float] = None, **kwargs: Any):
        """Log training step progress"""
        self.debug(f"Training step {step}: reward={reward:.3f}", 
                  step=step, 
                  reward=reward, 
                  loss=loss,
                  event_type="training_step",
                  **kwargs)
        self.log_scalar("train/reward", reward, step)
        if loss is not None:
            self.log_scalar("train/loss", loss, step)

    def training_complete(self, total_steps: int, final_reward: float, **kwargs: Any):
        """Log training completion"""
        self.info(f"Training completed: {total_steps} steps, final reward: {final_reward:.3f}", 
                 total_steps=total_steps,
                 final_reward=final_reward,
                 event_type="training_complete",
                 **kwargs)
        self.save_metrics()
    
    def match_result(self, player1: str, player2: str, winner: str, score: str, **kwargs: Any):
        """Log match result"""
        self.info(f"Match result: {player1} vs {player2} - Winner: {winner} ({score})", 
                 player1=player1,
                 player2=player2,
                 winner=winner,
                 score=score,
                 event_type="match_result",
                 **kwargs)
    
    def model_saved(self, model_path: str, algorithm: str, **kwargs: Any):
        """Log model save event"""
        self.info(f"Model saved: {model_path}", 
                 model_path=model_path,
                 algorithm=algorithm,
                 event_type="model_saved",
                 **kwargs)
    
    def environment_info(self, obs_space: Any, action_space: Any, **kwargs: Any):
        """Log environment information"""
        self.info(f"Environment: obs_space={obs_space.shape}, action_space={action_space.shape}", 
                 obs_space_shape=obs_space.shape,
                 action_space_shape=action_space.shape,
                 event_type="environment_info",
                 **kwargs)

    def save_metrics(self):
        """Save all metrics to JSON file"""
        metrics_file = self.experiment_dir / "metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2, default=str)

    def close(self):
        """Close the logger and save final metrics"""
        self.save_metrics()
        self.writer.close()
        self.info("Logger closed")


# Global logger instance
logger = RLGymLogger()

# Convenience functions for backward compatibility
def log_debug(message: str, **kwargs: Any):
    """Log debug message"""
    logger.debug(message, **kwargs)

def log_info(message: str, **kwargs: Any):
    """Log info message"""
    logger.info(message, **kwargs)

def log_warning(message: str, **kwargs: Any):
    """Log warning message"""
    logger.warning(message, **kwargs)

def log_error(message: str, **kwargs: Any):
    """Log error message"""
    logger.error(message, **kwargs)

def log_exception(message: str, **kwargs: Any):
    """Log exception with traceback"""
    logger.exception(message, **kwargs)


# Context manager for timing operations
class TimerContext:
    """Context manager for timing operations"""
    
    def __init__(self, operation_name: str, logger_instance: Optional['RLGymLogger'] = None):
        self.operation_name = operation_name
        self.logger = logger_instance or logger
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.info(f"Starting {self.operation_name}")
        return self
    
    def __exit__(self, exc_type: Optional[type], exc_val: Optional[BaseException], exc_tb: Optional[Any]):
        if self.start_time:
            elapsed = (datetime.now() - self.start_time).total_seconds()
            if exc_type:
                self.logger.error(f"{self.operation_name} failed after {elapsed:.2f}s", 
                                operation=self.operation_name,
                                duration=elapsed,
                                error=str(exc_val))
            else:
                self.logger.info(f"{self.operation_name} completed in {elapsed:.2f}s", 
                               operation=self.operation_name,
                               duration=elapsed)


# Decorator for timing functions
def timed_operation(operation_name: Optional[str] = None) -> Callable:
    """Decorator to time operations"""
    def decorator(func: Callable) -> Callable:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            op_name = operation_name or func.__name__
            with TimerContext(op_name):
                return func(*args, **kwargs)
        return wrapper
    return decorator

# Training Guide

This page provides a guide on training your RLGymExampleBot.

## 1. Training Workflow Overview

The typical training workflow involves several steps:

1.  **Environment Setup**: Configure the observation and action spaces for your agent.
2.  **Training**: Run the main training script with your desired reinforcement learning algorithm.
3.  **Evaluation**: Monitor training progress using TensorBoard and other metrics.
4.  **Model Conversion**: Convert the trained Stable-Baselines3 model to a PyTorch format compatible with RLBot.
5.  **Deployment**: Integrate the converted model into your bot for live gameplay.

## 2. Configuration

Training parameters are centralized in `src/config.py`. This file uses Python dataclasses to organize various configuration settings. You can edit this file to customize:

*   **`TrainingConfig`**: Defines parameters for the reinforcement learning algorithms.
    *   `algorithm`: The specific RL algorithm to use (e.g., `PPO`, `SAC`, `TD3`, `A2C`, `DQN`).
    *   `total_timesteps`: Total number of timesteps the agent will train for.
    *   `save_freq`: How often (in timesteps) the model checkpoints will be saved.
    *   `learning_rate`: The learning rate for the optimizer.
    *   `batch_size`: The batch size used during training updates.
    *   `n_environments`: Number of parallel environments to run during training (for vectorized environments).
    *   `eval_freq`: How often (in timesteps) the agent will be evaluated during training.
    *   `test_episodes`: Number of episodes to run during evaluation.
    *   `test_max_steps`: Maximum steps per episode during evaluation.
    *   `seed`: Random seed for reproducibility.
    *   **Algorithm-Specific Parameters**: Parameters unique to each algorithm (e.g., `ppo_n_steps`, `sac_buffer_size`, `td3_policy_delay`).
    *   **Model Architecture**: `hidden_size` (neurons in hidden layers) and `n_layers` (number of hidden layers).
    *   **Early Stopping**: `early_stopping_patience` and `early_stopping_min_delta` to stop training if performance plateaus.

*   **`EnvironmentConfig`**: Defines parameters for the RLGym environment.
    *   `max_steps`: Maximum steps per episode in the training environment.
    *   `difficulty`: Difficulty setting for the environment (e.g., "medium").
    *   `tick_skip`: Number of game ticks to skip between agent actions.
    *   `gravity`, `ball_drag`, `car_drag`, `boost_force`, `jump_force`, `flip_force`: Physics parameters.
    *   `observation_normalization`, `reward_normalization`: Flags to enable/disable normalization.

*   **`ModelConfig`**: Defines parameters related to the neural network model architecture.
    *   `hidden_size`, `activation`, `n_layers`, `dropout`, `use_batch_norm`, `use_dropout`, `dropout_rate`.

*   **`LoggingConfig`**: Defines parameters for logging and monitoring.
    *   `tensorboard_log_dir`, `model_save_dir`, `log_level`, `save_replay_buffer`, `log_interval`.

*   **`LeagueConfig`**: Defines parameters for the self-play league system.
    *   `base_rating`, `rating_k_factor`, `min_matches_for_rating`, `tournament_size`, `promotion_threshold`, `demotion_threshold`.

## 3. Starting Training

You can start training using the provided batch file (Windows) or by running the Python script directly. The `src/launch_training.py` script provides a user-friendly interface for managing and launching training sessions.

### Using the Batch File (Windows)

Navigate to the project root (`RLGymExampleBot/`) and use `start_training.bat`. This script forwards arguments to `src/launch_training.py`.

```batch
# Run with default settings (PPO, 100k timesteps, test, convert)
start_training.bat

# Run with custom algorithm and timesteps
start_training.bat --algorithm SAC --timesteps 500000

# Run without testing or converting after training
start_training.bat --no-test --no-convert

# Display help for all options
start_training.bat --help
```

### Using Python Directly (`src/launch_training.py`)

The `launch_training.py` script offers several commands for managing your training configurations:

*   **`--list`**: Lists all available training configurations (default and custom).
    ```bash
    python src/launch_training.py --list
    ```

*   **`--create`**: Guides you through an interactive process to create a new custom training configuration.
    ```bash
    python src/launch_training.py --create
    ```

*   **`--config <CONFIG_NAME>`**: Launches a training session using a specified configuration.
    ```bash
    # Run with a default configuration (e.g., standard_training)
    python src/launch_training.py --config standard_training

    # Run with a custom configuration you created
    python src/launch_training.py --config my_custom_ppo_config

    # Launch with a specific configuration and resume from a checkpoint
    python src/launch_training.py --config standard_training --resume models/PPO/PPO_model_120000_steps.zip

    # Launch with a specific configuration and enable testing/rendering
    python src/launch_training.py --config quick_test --test --render
    ```

You can also run `src/launch_training.py` without any arguments to see a quick overview of its usage.
```bash
# Run with default settings (equivalent to --config standard_training if no other args)
python src/launch_training.py
```

## 4. Supported Algorithms

The project supports several popular reinforcement learning algorithms from Stable-Baselines3:

### PPO (Proximal Policy Optimization)
*   **Best for**: Beginners, stable training.
*   **Use case**: General RL tasks, good sample efficiency.
*   **Parameters**: Configurable in `src/config.py`.

### SAC (Soft Actor-Critic)
*   **Best for**: Continuous action spaces.
*   **Use case**: Exploration-heavy tasks.
*   **Parameters**: Configurable in `src/config.py`.

### TD3 (Twin Delayed DDPG)
*   **Best for**: Continuous control, stable Q-learning.
*   **Use case**: When you need deterministic actions.
*   **Parameters**: Configurable in `src/config.py`.

### A2C (Advantage Actor-Critic)
*   **Best for**: Simplicity and good performance for many tasks.
*   **Use case**: Baseline for continuous and discrete action spaces.
*   **Parameters**: Configurable in `src/config.py`.

### DQN (Deep Q-Network)
*   **Best for**: Discrete action spaces.
*   **Use case**: Value-based method for discrete control problems.
*   **Parameters**: Configurable in `src/config.py`.

## 5. Monitoring Training Progress

*   **TensorBoard**: View training metrics in real-time.
    ```bash
    tensorboard --logdir logs
    ```
*   **Console Output**: Progress bars and status updates are printed to the console.
*   **Log Files**: Detailed training logs are saved in the `logs/` directory.
*   **Checkpoints**: Models are automatically saved every `save_freq` steps in the `models/` directory.

[Home](Home.md)
# Hyperparameter Optimization

This page explains how to use the built-in hyperparameter optimization tools to find optimal settings for your RLGymExampleBot's training.

## Understanding Hyperparameter Optimization

Hyperparameters are configuration variables external to the model whose values cannot be estimated from data. They are set before the learning process begins (e.g., learning rate, batch size, number of hidden layers). The performance of a reinforcement learning agent is highly sensitive to its hyperparameters.

Hyperparameter optimization is the process of finding a set of hyperparameters that yields the best performance for a given task. This project integrates `Optuna`, a popular hyperparameter optimization framework.

## `src/hyperparameter_optimization.py`

This script provides a `HyperparameterOptimizer` class that uses Optuna to search for optimal hyperparameters for different RL algorithms (PPO, SAC, TD3, A2C, DQN).

### Key Concepts:

*   **Study**: An optimization session managed by Optuna. It stores all trials and their results.
*   **Trial**: A single run of the training process with a specific set of hyperparameters suggested by Optuna.
*   **Objective Function**: A function that Optuna tries to optimize (maximize or minimize). In our case, it's typically the mean reward achieved by the agent during evaluation.
*   **Samplers**: Algorithms Optuna uses to suggest new hyperparameter combinations (e.g., Tree-structured Parzen Estimator - TPE).
*   **Pruners**: Algorithms that automatically stop unpromising trials early to save computational resources.

## How to Run Hyperparameter Optimization

You can run hyperparameter optimization using the `hyperparameter_optimization.py` script directly.

```bash
# Optimize PPO hyperparameters for 100 trials
python src/hyperparameter_optimization.py --agent PPO --trials 100

# Optimize SAC hyperparameters with a custom study name
python src/hyperparameter_optimization.py --agent SAC --trials 50 --study-name sac_experiment_v1

# Display help for all options
python src/hyperparameter_optimization.py --help
```

### Command-Line Arguments:

*   `--agent`: (Required) The type of agent whose hyperparameters you want to optimize (e.g., `PPO`, `SAC`, `TD3`, `A2C`, `DQN`).
*   `--trials`: (Optional) The number of optimization trials to run. Default is 100.
*   `--study-name`: (Optional) A custom name for the Optuna study. If not provided, a default name based on the agent type will be used.

## Interpreting Results

After the optimization run completes, results are saved in a `studies/` directory within your project root, under a subdirectory named after your study (e.g., `studies/PPO_optimization/`).

Key output files include:

*   **`study.db`**: The SQLite database file where Optuna stores all trial data. You can use Optuna's visualization tools to analyze this database.
*   **`best_parameters.json`**: A JSON file containing the hyperparameters of the best performing trial (highest mean reward).
*   **`study_summary.json`**: A summary of the entire optimization study, including best trial details and overall statistics.
*   **`optimization_history.html`**: An HTML plot showing the optimization history (how the objective value changed over trials).
*   **`param_importances.html`**: An HTML plot showing the importance of each hyperparameter in influencing the objective value.

## Customizing the Optimization Process

You can customize the hyperparameter search space and optimization objective by modifying `src/hyperparameter_optimization.py`:

*   **`suggest_hyperparameters` method**: This method defines the search space for each agent type. You can change the ranges, types (e.g., `suggest_float`, `suggest_int`, `suggest_categorical`), and distributions of the hyperparameters.
*   **`objective` method**: This method defines how a trial is run and how its performance is evaluated. You can modify the training duration for each trial, the evaluation metrics, or the environment used for evaluation.

Hyperparameter optimization is a powerful technique to significantly improve the performance of your RL agents by systematically exploring the vast space of possible configurations.

[Home](Home.md)
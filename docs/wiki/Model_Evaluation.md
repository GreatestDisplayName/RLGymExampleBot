# Model Evaluation and Analysis

This page explains how to evaluate and analyze the performance of your trained RLGymExampleBot models.

## Understanding Model Evaluation

Evaluating your trained models is crucial to understand their performance, identify areas for improvement, and compare different training runs or algorithms. This project provides tools to perform comprehensive evaluations and generate insightful visualizations.

## `src/model_evaluation.py`

This script provides a `ModelEvaluator` class that can discover, evaluate, and compare trained models. It generates detailed metrics and visualizations.

### Key Concepts:

*   **Evaluation Episodes**: The number of games the agent plays to assess its performance.
*   **Metrics**: Mean reward, standard deviation of reward, success rate, efficiency, consistency, etc.
*   **Visualizations**: Plots to compare performance across different agent types or training runs.

## How to Run Model Evaluation

You can run model evaluation using the `model_evaluation.py` script directly.

```bash
# Evaluate all discovered models with default settings (100 episodes, 4 environments)
python src/model_evaluation.py

# Evaluate models with 200 episodes and skip visualization generation
python src/model_evaluation.py --episodes 200 --no-viz

# Specify a different directory for models and results
python src/model_evaluation.py --models-dir my_custom_models --results-dir my_eval_results

# Display help for all options
python src/model_evaluation.py --help
```

### Command-Line Arguments:

*   `--models-dir`: (Optional) Directory containing trained models. Default is `models/`.
*   `--results-dir`: (Optional) Directory to save evaluation results. Default is `evaluation_results/`.
*   `--episodes`: (Optional) Number of episodes per evaluation. Default is 100.
*   `--envs`: (Optional) Number of parallel environments to use during evaluation. Default is 4.
*   `--no-viz`: (Optional) Flag to skip visualization generation.

## Interpreting Results

After the evaluation run completes, results are saved in the `evaluation_results/` directory (or your specified `--results-dir`).

Key output files include:

*   **`evaluation_results_<timestamp>.json`**: Detailed JSON file containing metrics for each evaluated model.
*   **`evaluation_summary_<timestamp>.json`**: A JSON summary of the entire evaluation, including best models and overall performance.
*   **`evaluation_results_<timestamp>.csv`**: A CSV file with detailed evaluation metrics, suitable for further analysis in spreadsheets or data analysis tools.
*   **`evaluation_plots_<timestamp>.png`**: PNG image files containing various plots (e.g., mean reward comparison, success rate, efficiency distribution).

## Customizing Evaluation

You can customize the evaluation process by modifying `src/model_evaluation.py`:

*   **`discover_models` method**: Modify how models are discovered if your naming conventions or directory structure differ.
*   **`evaluate_model` method**: Adjust the evaluation logic, add custom metrics, or change the evaluation environment.
*   **`create_eval_env` method**: Customize the environment used for evaluation (e.g., different difficulty, max steps).
*   **`create_visualizations` method**: Modify the types of plots generated or their appearance.

Model evaluation is an essential step in the reinforcement learning workflow, providing the insights needed to iterate and improve your RLGymExampleBot.

[Home](Home.md)
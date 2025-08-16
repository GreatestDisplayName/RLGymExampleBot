# Introduction

This page provides an introduction to the RLGymExampleBot project.

The **RLGymExampleBot** is a comprehensive framework designed for training and deploying Rocket League bots using Reinforcement Learning (RL). It leverages the `RLGym` library for creating custom Rocket League environments and `Stable-Baselines3` for implementing various RL algorithms.

## Project Goals

The primary goals of this project are:

*   **End-to-End RL Pipeline**: Provide a complete workflow from environment setup and agent training to model conversion and deployment within the RLBot framework, enabling users to go from zero to a functional RL bot.
*   **Modular Design**: Allow easy customization of observation spaces, action spaces, reward functions, and RL algorithms. This modularity promotes experimentation and adaptation to different research or competitive needs.
*   **Reproducibility**: Ensure training processes are reproducible with clear configuration options and comprehensive logging. This is crucial for scientific rigor and sharing results.
*   **Self-Play Capabilities**: Integrate a self-play league system for agents to learn and improve through continuous competition against evolving opponents, mimicking advanced AI training methodologies.

## Key Components (Expanded)

The project is structured into several key components, each serving a specific role in the RL pipeline:

*   **`src/` directory**: This is the heart of the project, containing all core Python logic.
    *   `complete_workflow.py`: Acts as the central orchestrator, automating the entire training-to-deployment pipeline. It manages the sequential execution of environment setup, training, model conversion, and bot integration.
    *   `train.py`: Implements the core reinforcement learning training loops. It interfaces with `Stable-Baselines3` to train agents using specified algorithms and manages callbacks for logging, evaluation, and checkpointing.
    *   `config.py`: Provides a centralized and extensible configuration management system. It uses Python dataclasses to define various parameters for training, environment, model architecture, and logging, allowing for easy modification and overriding.
    *   `agent.py`: Defines the neural network architecture that serves as the policy for the RL agent. This typically involves defining the input (observation) and output (action) layers, and intermediate hidden layers.
    *   `training_env.py`: Contains the custom `RLGym` environment definition. This includes the game state representation (observations), available actions, and the reward function that guides the agent's learning.
    *   `convert_model.py`: A crucial utility for bridging the gap between `Stable-Baselines3` and `RLBot`. It converts trained `Stable-Baselines3` models (often saved as `.zip` files) into a pure PyTorch `.pth` format that can be directly loaded and used by RLBot.
    *   `league_manager.py`: Manages the self-play league system, handling agent registration, match scheduling, ELO rating updates, and training against league opponents.
    *   `bot.py`: The interface between the trained RL model and the RLBot framework. It loads the converted `.pth` model and translates the bot's observations into model inputs, and model outputs into bot actions within the Rocket League game.
*   **`scripts/` directory**: Houses convenient Windows batch files (`.bat`) that simplify the execution of common workflows (e.g., starting training, running the complete pipeline) without needing to type long Python commands.
*   **`models/` directory**: The primary storage location for all trained RL models, including intermediate checkpoints, final models, and converted `.pth` files. It also stores associated training metadata (e.g., hyperparameters, training duration).
*   **`logs/` directory**: Dedicated to storing training logs, including TensorBoard event files for visualizing training progress, performance metrics, and debugging information.
*   **`docs/` directory**: Contains all project documentation, including this wiki.

## Workflow Overview (Expanded)

The typical workflow within the RLGymExampleBot framework follows a structured pipeline:

1.  **Environment Setup**: This initial phase involves defining the Rocket League environment for training. This includes specifying the observation space (what the bot "sees"), the action space (what the bot can take), and the reward function (how the bot is rewarded or penalized for its actions). This is primarily configured in `src/training_env.py` and `src/config.py`.
2.  **Agent Training**: The core learning phase where the RL agent interacts with the environment to learn optimal policies. Users select an RL algorithm (e.g., PPO, SAC, TD3) and define training parameters (timesteps, learning rate, batch size) in `src/config.py`. The `src/train.py` script executes this process, saving model checkpoints and logging progress.
3.  **Model Conversion**: After training, the `Stable-Baselines3` model needs to be transformed into a format directly usable by the RLBot framework. The `src/convert_model.py` utility performs this crucial step, converting the `.zip` model into a lightweight PyTorch `.pth` file.
4.  **Deployment**: The converted `.pth` model is then integrated into an RLBot agent. The `src/bot.py` script loads this model, allowing the bot to use its learned policy to play within Rocket League. This step makes the trained AI tangible and playable.
5.  **Self-Play (Optional)**: For advanced training, the self-play league system (`src/league_manager.py`) allows agents to compete against each other. This continuous competition drives the evolution of agent strategies, leading to more robust and skilled AI. New agents can be trained against the current best-performing agents in the league.

This project aims to provide a robust and flexible foundation for anyone looking to develop and experiment with Rocket League AI using modern reinforcement learning techniques, from initial setup to advanced self-play training.

## Advanced Capabilities

Beyond the core workflow, RLGymExampleBot offers several advanced features to enhance your RL development:

*   **Modular Rewards, Observations, and Actions**: Easily customize how your bot perceives the game state, what actions it can take, and how it's rewarded, allowing for fine-tuned experimentation.
*   **Curriculum Learning**: Implement progressive training strategies where the environment's difficulty or complexity increases over time, helping agents learn complex behaviors more effectively.
*   **Imitation Learning Hooks**: Provides mechanisms to integrate imitation learning, allowing agents to learn from expert demonstrations and accelerate initial policy acquisition.
*   **Kevpert Drill Loader Stubs**: Includes placeholders or basic integrations for loading popular "Kevpert" training drills, enabling focused practice on specific Rocket League mechanics.
*   **Export to ONNX for RLBot**: Facilitates efficient deployment by supporting model export to the ONNX format, which can be highly optimized for inference within the RLBot framework.
*   **Rapid Prototyping**: The framework is designed so you can start training immediately without wiring everything from scratch, significantly reducing setup time and allowing for quick iteration on ideas.

[Home](Home.md)
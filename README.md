# RLGymExampleBot

## Introduction

RLGymExampleBot is an example bot for Rocket League, built using the RLGym library. This project provides a framework for developing, training, and deploying reinforcement learning agents for Rocket League. It's designed to be a starting point for researchers and enthusiasts looking to experiment with RL agents in the game.

## Features

*   **RLGym Integration:** Seamlessly integrates with the RLGym library for environment interaction.
*   **Configurable Agents:** Easily switch between different observation builders, action parsers, and models via configuration.
*   **Model Loading:** Supports loading the latest trained model or a specific model file.
*   **Training Configurations:** Includes example configurations for various training algorithms (e.g., PPO, SAC, TD3).
*   **RLBot Integration:** Designed to run within the RLBot framework.

## Getting Started

### Installation

To set up the RLGymExampleBot, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-repo/RLGymExampleBot.git
    cd RLGymExampleBot
    ```
2.  **Install dependencies:**
    The `run.py` script will automatically check for and install the necessary Python dependencies listed in `requirements.txt` when you first run the bot.

### Running the Bot

To run the RLGymExampleBot, execute the `run.py` script:

```bash
python run.py
```

This will launch the RLBot framework and load the RLGymExampleBot. Ensure that Rocket League is running and the RLBot framework is properly set up.

## Training

This project supports training reinforcement learning models using various algorithms. Training configurations are located in the `configs/` directory. You can launch training using the `src/launch_training.py` script (or similar, depending on your setup).

Example training command (assuming `src/launch_training.py` is the entry point):

```bash
python src/launch_training.py --config configs/standard_training.json
```

Refer to the `configs/` directory for available training configurations and `src/train.py` for the training script details.

## Configuration

The main configuration for the bot is located in `src/bot.cfg`. This file allows you to customize:

*   `model_path`: The path to your trained model. Set to `latest` to automatically load the most recently created model from the `models` directory.
*   `model_dir`: The directory where your trained models are stored (defaults to `../models`).
*   `tick_skip`: The number of game ticks to skip between bot decisions.
*   `obs_builder`: The observation builder class to use (e.g., `DefaultObs`, `AdvancedObs`).
*   `act_parser`: The action parser class to use (e.g., `DefaultAction`, `ContinuousAction`).

## Project Structure

```
RLGymExampleBot/
├── run.py                  # Main script to launch the bot
├── requirements.txt        # Python dependencies
├── src/
│   ├── bot.py              # Main bot class
│   ├── rlbot_support.py    # Core RLBot agent logic, including model loading
│   ├── bot.cfg             # Bot configuration file
│   ├── train.py            # Script for training RL models
│   ├── launch_training.py  # Script to launch training with specific configs
│   ├── action/             # Action parsers
│   ├── obs/                # Observation builders
│   └── ...
├── configs/                # Training configuration files
├── models/                 # Directory for trained models
├── league/                 # (Potentially) League management system
├── gui.py                  # (Potentially) Graphical User Interface
├── docs/                   # Project documentation
└── ...
```
# Deployment Guide

This page explains how to deploy your trained RLGymExampleBot.

## 1. Model Conversion

After training, your model is typically saved in a format specific to the Stable-Baselines3 library (e.g., `.zip` files). For deployment within the RLBot framework, these models need to be converted to a pure PyTorch format (`.pth`).

The `src/convert_model.py` script handles this conversion.

### How to Convert a Model

You can convert a model using the `convert_model.py` script directly.

```bash
# Convert the latest trained PPO model
python src/convert_model.py

# Convert a specific model
python src/convert_model.py --model models/PPO/PPO_model_120000_steps.zip --algorithm PPO

# Display help for all options
python src/convert_model.py --help
```

The converted model will be saved in the same directory as the original model, typically named `PPO_converted.pth` (or similar, depending on the algorithm).

## 2. Integrating with RLBot

Once your model is converted to `.pth` format, you can integrate it with the RLBot framework. The `src/bot.py` file is responsible for loading your trained model and making predictions for the bot's actions.

### Updating Bot Configuration

You need to ensure that your `bot.py` (or the relevant bot configuration file) points to the correct path of your converted model.

The `update_bot_config` method in `CompleteWorkflow` (or similar logic in your bot's setup) can help automate this. Manually, you would edit the `bot.cfg` file or the Python script that loads the model.

### Running the Bot

You can run your trained bot using the `run.py` script from the project root.

```bash
# Run the trained bot in a headless Rocket League instance
python run.py
```

For a graphical user interface to manage and launch bots, you can use `run_gui.py`.

```bash
# Run the RLBot GUI
python run_gui.py
```

## 3. Self-Play League Integration (Optional)

If you are using the self-play league system, your converted models can be added to the league for competitive play and continuous improvement. Refer to the [Self-Play League documentation](SELF_PLAY_LEAGUE.md) for more details.

[Home](Home.md)
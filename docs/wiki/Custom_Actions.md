# Customizing Action Spaces

This page explains how to customize the action space for your RLGymExampleBot, allowing you to define how your bot translates its neural network outputs into game actions.

## Understanding Action Spaces

An action space defines the set of possible actions that your reinforcement learning agent can take in the environment. In Rocket League, actions typically involve throttle, steer, jump, boost, etc.

The `RLGym` library provides a flexible way to define custom action parsers. The core idea is to transform a numerical vector (output from your neural network) into a format that the game can understand.

## Default Action Parsers

The project includes two example action parsers in the `src/action/` directory:

*   **`src/action/continuous_act.py`**: Implements a simple continuous action space where all actions (including binary ones like jump/boost) are represented as values between -1 and 1. Binary actions are then thresholded.
*   **`src/action/discrete_act.py`**: Implements a discrete action space where continuous actions are binned into a fixed number of categories (e.g., -1, 0, 1).

## Creating Your Own Custom Action Parser

To create a custom action parser, you typically define a Python class that implements a `parse_actions` method.

### `parse_actions` Method

The `parse_actions` method is the most important part of your custom action parser. It takes `actions` (the raw output from your neural network, typically a NumPy array) and `state` (the current game state) as input. It should return a NumPy array of actions in the format expected by the RLGym environment.

```python
import numpy as np
from rlgym_compat import GameState # Import necessary types

class MyCustomAction:
    def __init__(self):
        # Initialize any parameters or constants here
        pass

    def parse_actions(self, actions: np.ndarray, state: GameState) -> np.ndarray:
        """
        Parses the raw neural network output into game actions.

        Args:
            actions (np.ndarray): The raw output from the neural network.
                                  Shape typically (batch_size, action_dim).
            state (GameState): The current game state.

        Returns:
            np.ndarray: The parsed actions in a format RLGym understands.
                        Shape typically (batch_size, 8) for Rocket League.
        """
        # Ensure actions are in the correct shape (e.g., (batch_size, action_dim))
        # Example: If your NN outputs 8 values per agent
        actions = actions.reshape((-1, 8))

        # --- Continuous Actions (Throttle, Steer, Pitch, Yaw, Roll) ---
        # These are typically directly mapped from -1 to 1
        throttle = actions[:, 0]
        steer = actions[:, 1]
        pitch = actions[:, 2]
        yaw = actions[:, 3]
        roll = actions[:, 4]

        # --- Binary Actions (Jump, Boost, Handbrake) ---
        # These need to be converted from continuous outputs (e.g., -1 to 1) to binary (0 or 1)
        jump = (actions[:, 5] > 0).astype(float) # Threshold at 0
        boost = (actions[:, 6] > 0).astype(float)
        handbrake = (actions[:, 7] > 0).astype(float)

        # Combine into the final action array
        # Ensure the order matches RLGym's expectation:
        # [throttle, steer, pitch, yaw, roll, jump, boost, handbrake]
        parsed_actions = np.stack([throttle, steer, pitch, yaw, roll, jump, boost, handbrake], axis=-1)

        return parsed_actions
```

### Integrating Your Custom Action Parser

Once you have created your `MyCustomAction` class (e.g., in `src/action/my_custom_act.py`), you need to tell your bot to use it.

1.  **Import your parser**: In `src/bot.py`, change the import:
    ```python
    # from action.default_act import DefaultAction
    from action.my_custom_act import MyCustomAction # Assuming your file is my_custom_act.py
    ```
2.  **Instantiate your parser**: In the `RLGymExampleBot` class's `__init__` method in `src/bot.py`:
    ```python
    # self.act_parser = DefaultAction()
    self.act_parser = MyCustomAction()
    ```

By creating custom action parsers, you can experiment with different ways your bot controls its car, from simple binary inputs to more complex combinations of actions.

[Home](Home.md)
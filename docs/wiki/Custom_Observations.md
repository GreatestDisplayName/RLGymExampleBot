# Customizing Observation Spaces

This page explains how to customize the observation space for your RLGymExampleBot, allowing you to control what information your bot perceives from the game.

## Understanding Observation Spaces

An observation space defines the data that your reinforcement learning agent receives from the environment at each timestep. In Rocket League, this typically includes information about the ball, your car, teammates, and opponents.

The `RLGym` library provides a flexible way to define custom observation builders. The core idea is to transform the raw game state into a numerical vector that your neural network can understand.

## Default Observation Builders

The project includes two example observation builders in the `src/obs/` directory:

*   **`src/obs/default_obs.py`**: A basic observation builder that provides normalized physical quantities of game elements and player-specific data. It's a good starting point for many tasks.
*   **`src/obs/advanced_obs.py`**: An example of a more complex observation space, providing normalized physical quantities and relative positions/velocities of game elements, including more detailed team and opponent information.

## Creating Your Own Custom Observation Builder

To create a custom observation builder, you typically define a Python class that inherits from a base class (though not strictly required by RLGym, it's good practice) and implements a `build_obs` method.

### `build_obs` Method

The `build_obs` method is the most important part of your custom observation builder. It takes `player`, `state`, and `previous_action` as input and should return a NumPy array representing the observation vector.

```python
import numpy as np
from rlgym_compat import GameState, PlayerData, PhysicsObject # Import necessary types

class MyCustomObservation:
    def __init__(self):
        # Initialize any parameters or constants here
        self.POS_STD = 2300 # Example normalization constant

    def reset(self, initial_state: GameState):
        # Optional: Reset any internal state of your observation builder
        pass

    def build_obs(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> np.ndarray:
        """
        Builds the observation vector for the given player and game state.

        Args:
            player (PlayerData): Data for the current player's car.
            state (GameState): The overall game state.
            previous_action (np.ndarray): The action taken in the previous timestep.

        Returns:
            np.ndarray: The observation vector.
        """
        obs = []

        # --- Ball Information ---
        # Ball position (x, y, z)
        obs.extend(state.ball.position / self.POS_STD)
        # Ball linear velocity
        obs.extend(state.ball.linear_velocity / self.POS_STD)
        # Ball angular velocity
        obs.extend(state.ball.angular_velocity / np.pi) # Normalize by PI for angles

        # --- Player Car Information ---
        # Current player's car data (position, velocity, rotation, boost, etc.)
        car_data = player.car_data
        obs.extend(car_data.position / self.POS_STD)
        obs.extend(car_data.linear_velocity / self.POS_STD)
        obs.extend(car_data.angular_velocity / np.pi)
        obs.extend(car_data.forward()) # Car's forward vector
        obs.extend(car_data.up())      # Car's up vector
        obs.append(player.boost_amount / 100) # Normalize boost
        obs.append(float(player.on_ground))
        obs.append(float(player.has_flip))
        obs.append(float(player.is_demoed))

        # --- Relative Information (Example: Ball relative to car) ---
        rel_ball_pos = state.ball.position - car_data.position
        rel_ball_vel = state.ball.linear_velocity - car_data.linear_velocity
        obs.extend(rel_ball_pos / self.POS_STD)
        obs.extend(rel_ball_vel / self.POS_STD)

        # --- Previous Action ---
        obs.extend(previous_action)

        # --- Boost Pad Information (Example: only active pads) ---
        # You might want to include positions of active boost pads
        # for i, pad_active in enumerate(state.boost_pads):
        #     if pad_active:
        #         # Add pad position or a binary flag
        #         obs.append(1.0)
        #     else:
        #         obs.append(0.0)

        # --- Teammate and Opponent Information (Example) ---
        # Iterate through other players in the game state
        for other_player in state.players:
            if other_player.car_id == player.car_id: # Skip current player
                continue

            other_car_data = other_player.car_data
            obs.extend(other_car_data.position / self.POS_STD)
            obs.extend(other_car_data.linear_velocity / self.POS_STD)
            # Add relative info for other cars
            rel_other_pos = other_car_data.position - car_data.position
            obs.extend(rel_other_pos / self.POS_STD)


        return np.array(obs, dtype=np.float32)
```

### Integrating Your Custom Observation Builder

Once you have created your `MyCustomObservation` class (e.g., in `src/obs/my_custom_obs.py`), you need to tell your bot to use it.

1.  **Import your builder**: In `src/bot.py`, change the import:
    ```python
    # from obs.default_obs import DefaultObs
    from obs.my_custom_obs import MyCustomObservation # Assuming your file is my_custom_obs.py
    ```
2.  **Instantiate your builder**: In the `RLGymExampleBot` class's `__init__` method in `src/bot.py`:
    ```python
    # self.obs_builder = DefaultObs()
    self.obs_builder = MyCustomObservation()
    ```
3.  **Update Agent Input Size**: Ensure the `input_size` parameter when initializing your `Agent` in `src/bot.py` matches the actual size of the observation vector returned by your `build_obs` method. You might need to calculate this size carefully.

By following these steps, you can tailor the information your bot receives, which is crucial for developing specialized and high-performing agents.

[Home](Home.md)
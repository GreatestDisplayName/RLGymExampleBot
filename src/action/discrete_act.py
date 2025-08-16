import numpy as np
from rlgym_compat import GameState
from typing import Any, Tuple


class DiscreteAction:
    """
    Represents a discrete action space for Rocket League agents.
    Analog actions (throttle, steer, pitch, yaw, roll) are binned into a specified
    number of discrete values (e.g., -1, 0, 1 for 3 bins).
    """

    # Constants for action dimensions and binned action count
    ACTION_DIM: int = 8 # Total number of actions
    BINNED_ACTION_COUNT: int = 5 # Number of actions that are binned (throttle, steer, pitch, yaw, roll)

    def __init__(self, n_bins: int = 3):
        """
        Initializes the DiscreteAction parser.
        
        Args:
            n_bins (int): The number of bins for analog actions. Must be an odd number
                          to include a zero/neutral action.
        
        Raises:
            AssertionError: If `n_bins` is an even number.
        """
        # Ensure n_bins is odd so there's a clear middle (zero) bin
        assert n_bins % 2 == 1, "n_bins must be an odd number to have a neutral action (0)"
        self._n_bins = n_bins

    def get_action_space(self) -> Any:
        """
        Returns the Gymnasium action space object.
        This method is not implemented as RLGym-Gymnasium compatibility is handled
        by `rlgym_compat` and direct Gymnasium dependency is avoided here.
        
        Raises:
            NotImplementedError: This method is intentionally not implemented.
        """
        raise NotImplementedError("We don't implement get_action_space to remove the gym dependency")

    def parse_actions(self, actions: np.ndarray, state: GameState) -> np.ndarray:
        """
        Parses and converts raw discrete actions from the agent into a format
        suitable for the Rocket League environment.
        
        Args:
            actions (np.ndarray): A numpy array of discrete actions from the agent.
                                  Expected shape: (N, ACTION_DIM) or (ACTION_DIM,)
                                  where N is the number of agents.
                                  Values for binned actions are expected to be integers
                                  from 0 to `n_bins - 1`.
            state (GameState): The current game state (used for context, but not directly
                                for action parsing in this simple implementation).
                                
        Returns:
            np.ndarray: A numpy array of parsed actions, with binned actions mapped
                        to the range [-1, 1]. Shape: (N, ACTION_DIM).
        """
        # Reshape actions to (batch_size, ACTION_DIM) and cast to float32 for calculations.
        actions = actions.reshape((-1, self.ACTION_DIM)).astype(dtype=np.float32)

        # Map the binned analog actions from their integer representation (0 to n_bins-1)
        # to a continuous range of -1 to 1.
        # Example: if n_bins=3, bins are {0, 1, 2}.
        # (actions / (3 // 2)) - 1  => (actions / 1) - 1
        # 0 -> -1
        # 1 -> 0
        # 2 -> 1
        actions[..., :self.BINNED_ACTION_COUNT] = actions[..., :self.BINNED_ACTION_COUNT] / (self._n_bins // 2) - 1

        return actions
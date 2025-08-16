import numpy as np
from rlgym_compat import GameState
from typing import Any, Tuple


class ContinuousAction:
    """
    Represents a continuous action space for Rocket League agents.
    Actions are expected to be in the range of -1 to 1.
    Binary actions (jump, boost, handbrake) are converted from continuous
    values back to binary (0 or 1) based on a threshold.
    """

    # Constants for action dimensions and binary action indexing
    ACTION_DIM: int = 8
    BINARY_ACTION_START_INDEX: int = 5 # Index where binary actions (jump, boost, handbrake) start

    def __init__(self):
        """
        Initializes the ContinuousAction parser.
        No specific parameters are needed for this action space.
        """
        pass

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
        Parses and converts raw continuous actions from the agent into a format
        suitable for the Rocket League environment.
        
        Args:
            actions (np.ndarray): A numpy array of continuous actions from the agent.
                                  Expected shape: (N, ACTION_DIM) or (ACTION_DIM,)
                                  where N is the number of agents.
            state (GameState): The current game state (used for context, but not directly
                                for action parsing in this simple implementation).
                                
        Returns:
            np.ndarray: A numpy array of parsed actions, with binary actions converted.
                        Shape: (N, ACTION_DIM).
        """
        # Ensure actions array has the correct shape (N, ACTION_DIM)
        # -1 infers the batch size, ACTION_DIM is fixed at 8
        actions = actions.reshape((-1, self.ACTION_DIM))

        # Clip the first 5 actions (throttle, steer, pitch, yaw, roll) to the range [-1, 1].
        # This ensures that the continuous actions are within their valid operational limits.
        actions[..., :self.BINARY_ACTION_START_INDEX] = actions[..., :self.BINARY_ACTION_START_INDEX].clip(-1, 1)
        
        # Convert binary actions (jump, boost, handbrake) from continuous to discrete (0 or 1).
        # These actions are inherently binary in Rocket League. A value > 0 is treated as 1 (active),
        # and <= 0 is treated as 0 (inactive).
        actions[..., self.BINARY_ACTION_START_INDEX:] = actions[..., self.BINARY_ACTION_START_INDEX:] > 0

        return actions

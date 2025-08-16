import numpy as np
from rlgym_compat import GameState
from .continuous_act import ContinuousAction
from typing import Union, List, Any


class DefaultAction(ContinuousAction):
    """
    A default continuous action space that extends `ContinuousAction`.
    This class provides additional flexibility by accepting various input formats
    for actions, ensuring compatibility and ease of use.
    """

    def __init__(self):
        """
        Initializes the DefaultAction parser.
        Calls the constructor of the parent class `ContinuousAction`.
        """
        super().__init__()

    def get_action_size(self) -> int:
        return self.ACTION_DIM

    def get_action_space(self) -> Any:
        """
        Returns the Gymnasium action space object.
        This method delegates to the parent class's implementation.
        
        Raises:
            NotImplementedError: This method is intentionally not implemented in the base class.
        """
        return super().get_action_space()

    def parse_actions(self, actions: Union[np.ndarray, List[np.ndarray], List[float]], state: GameState) -> np.ndarray:
        """
        Parses and converts raw actions from various input formats into a standardized
        numpy array format before delegating to the parent class for further processing.
        
        Args:
            actions (Union[np.ndarray, List[np.ndarray], List[float]]): 
                Actions from the agent. Can be a numpy array, a list of numpy arrays,
                or a list of floats.
            state (GameState): The current game state.
            
        Returns:
            np.ndarray: A numpy array of parsed actions, ready for the environment.
                        Shape: (N, ACTION_DIM).
                        
        Raises:
            ValueError: If the input `actions` has an invalid shape.
        """
        # Convert input to numpy array if it's not already.
        # This provides flexibility for agents that might output lists or other array-like structures.
        if not isinstance(actions, np.ndarray):
            actions = np.asarray(actions)

        # Reshape 1D action arrays to 2D (batch_size, action_dim) if necessary.
        # This handles cases where a single action is provided without an explicit batch dimension.
        if len(actions.shape) == 1:
            actions = actions.reshape((-1, ContinuousAction.ACTION_DIM))
        # Raise an error for invalid input shapes (e.g., 3D or higher).
        elif len(actions.shape) > 2:
            raise ValueError('{} is not a valid action shape. Expected 1D or 2D array.'.format(actions.shape))

        # Delegate the actual action parsing (clipping, binary conversion) to the parent class.
        return super().parse_actions(actions, state)
import math
import numpy as np
from typing import Any, List, Tuple
from rlgym_compat import common_values
from rlgym_compat import PlayerData, GameState, PhysicsObject


class AdvancedObs:
    """
    Implements an advanced observation space for RLGym, providing normalized
    physical quantities and relative positions/velocities of game elements.
    This observation builder aims to provide a rich and informative state
    representation for the agent.
    """
    # Normalization constants for position and angular velocity
    POS_STD: float = 2300.0  # Standard deviation for position normalization
    ANG_STD: float = math.pi # Standard deviation for angular velocity normalization (radians)
    
    # Team number for orange team, used for observation inversion
    ORANGE_TEAM_NUM: int = common_values.ORANGE_TEAM

    def __init__(self):
        """
        Initializes the AdvancedObs observation builder.
        """
        super().__init__()

    def reset(self, initial_state: GameState):
        """
        Resets the observation builder.
        This method is called at the beginning of each episode.
        
        Args:
            initial_state (GameState): The initial game state.
        """
        pass # No internal state to reset for this observation builder

    def build_obs(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> np.ndarray:
        """
        Builds the observation array for a given player in the current game state.
        Observations are inverted for the orange team to maintain a consistent
        perspective (always from blue team's side).
        
        Args:
            player (PlayerData): The player for whom to build the observation.
            state (GameState): The current game state.
            previous_action (np.ndarray): The action taken by the player in the previous timestep.
            
        Returns:
            np.ndarray: The concatenated observation array.
        """
        # Determine if observations need to be inverted based on the player's team.
        # Inverting observations ensures that the agent always perceives the field
        # from the same perspective, regardless of which team it's on.
        if player.team_num == self.ORANGE_TEAM_NUM:
            inverted = True
            ball = state.inverted_ball # Use inverted ball data
            pads = state.inverted_boost_pads # Use inverted boost pad data
        else:
            inverted = False
            ball = state.ball # Use normal ball data
            pads = state.boost_pads # Use normal boost pad data

        # Initial observation components: ball's state and previous action
        obs: List[np.ndarray] = [
            ball.position / self.POS_STD,          # Normalized ball position
            ball.linear_velocity / self.POS_STD,   # Normalized ball linear velocity
            ball.angular_velocity / self.ANG_STD,  # Normalized ball angular velocity
            previous_action,                       # Agent's previous action
            pads                                   # Boost pad states
        ]

        # Add the current player's car data to the observation
        player_car = self._add_player_to_obs(obs, player, ball, inverted)

        # Prepare lists for allies and enemies observations
        allies: List[np.ndarray] = []
        enemies: List[np.ndarray] = []

        # Iterate through all other players in the game state
        for other in state.players:
            if other.car_id == player.car_id:
                continue # Skip the current player

            # Assign other players to either allies or enemies list
            if other.team_num == player.team_num:
                team_obs = allies
            else:
                team_obs = enemies

            # Add the other player's car data to their respective team's observation list
            other_car = self._add_player_to_obs(team_obs, other, ball, inverted)

            # Add relative position and velocity between the current player and other players
            team_obs.extend([
                (other_car.position - player_car.position) / self.POS_STD,          # Normalized relative position
                (other_car.linear_velocity - player_car.linear_velocity) / self.POS_STD # Normalized relative velocity
            ])

        # Extend the main observation list with allies and enemies data
        obs.extend(allies)
        obs.extend(enemies)
        
        # Concatenate all observation components into a single 1D numpy array
        return np.concatenate(obs)

    def _add_player_to_obs(self, obs_list: List[np.ndarray], player: PlayerData, ball: PhysicsObject, inverted: bool) -> PhysicsObject:
        """
        Adds a player's car data to the observation list.
        
        Args:
            obs_list (List[np.ndarray]): The list to which observation components will be added.
            player (PlayerData): The player data.
            ball (PhysicsObject): The ball's physics object (inverted or not, depending on context).
            inverted (bool): True if observations are inverted for the orange team.
            
        Returns:
            PhysicsObject: The player's car physics object (inverted or not).
        """
        # Select the correct car data (inverted for orange team)
        if inverted:
            player_car = player.inverted_car_data
        else:
            player_car = player.car_data

        # Calculate relative position and velocity of the ball to the player's car
        rel_pos = ball.position - player_car.position
        rel_vel = ball.linear_velocity - player_car.linear_velocity

        # Extend the observation list with player-specific features
        obs_list.extend([
            rel_pos / self.POS_STD,                 # Normalized relative position to ball
            rel_vel / self.POS_STD,                 # Normalized relative velocity to ball
            player_car.position / self.POS_STD,     # Normalized absolute car position
            player_car.forward(),                   # Car's forward direction vector
            player_car.up(),                        # Car's up direction vector
            player_car.linear_velocity / self.POS_STD, # Normalized car linear velocity
            player_car.angular_velocity / self.ANG_STD, # Normalized car angular velocity
            np.array([                              # Miscellaneous car state information
                player.boost_amount,                # Current boost amount
                int(player.on_ground),              # Is car on ground (binary)
                int(player.has_flip),               # Does car have a flip available (binary)
                int(player.is_demoed)               # Is car demoed (binary)
            ], dtype=np.float32)])

        return player_car
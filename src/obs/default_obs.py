import math
import numpy as np
from typing import Any, List, Tuple
from rlgym_compat import common_values
from rlgym_compat import PlayerData, GameState


class DefaultObs:
    def __init__(self, pos_coef: float = 1/2300, ang_coef: float = 1/math.pi, 
                 lin_vel_coef: float = 1/2300, ang_vel_coef: float = 1/math.pi, max_players: int = 6):
        super().__init__()
        self.POS_COEF = pos_coef
        self.ANG_COEF = ang_coef
        self.LIN_VEL_COEF = lin_vel_coef
        self.ANG_VEL_COEF = ang_vel_coef
        self.max_players = max_players

    def get_obs_size(self):
        # ball data (3+3+3) + pads (34) + previous_action (8) = 51
        # player data (3+3+3+3+3+4) = 19
        return 51 + 19 * self.max_players

    def reset(self, initial_state: GameState):
        pass

    def build_obs(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> np.ndarray:
        if player.team_num == common_values.ORANGE_TEAM:
            inverted = True
            ball = state.inverted_ball
            pads = state.inverted_boost_pads
        else:
            inverted = False
            ball = state.ball
            pads = state.boost_pads

        obs = [ball.position * self.POS_COEF,
               ball.linear_velocity * self.LIN_VEL_COEF,
               ball.angular_velocity * self.ANG_VEL_COEF,
               previous_action,
               pads]

        self._add_player_to_obs(obs, player, inverted)

        allies = []
        enemies = []

        for other in state.players:
            if other.car_id == player.car_id:
                continue

            if other.team_num == player.team_num:
                team_obs = allies
            else:
                team_obs = enemies

            self._add_player_to_obs(team_obs, other, inverted)
        
        # pad observations
        for _ in range(self.max_players - 1 - len(allies) - len(enemies)):
            enemies.append(np.zeros(19))

        obs.extend(allies)
        obs.extend(enemies)

        return np.concatenate(obs)

    def _add_player_to_obs(self, obs_list: List[np.ndarray], player: PlayerData, inverted: bool):
        if inverted:
            player_car = player.inverted_car_data
        else:
            player_car = player.car_data

        obs_list.extend([
            player_car.position * self.POS_COEF,
            player_car.forward(),
            player_car.up(),
            player_car.linear_velocity * self.LIN_VEL_COEF,
            player_car.angular_velocity * self.ANG_VEL_COEF,
            np.array([player.boost_amount, int(player.on_ground), int(player.has_flip), int(player.is_demoed)], dtype=np.float32)])

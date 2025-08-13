import numpy as np
from rlgym_compat import GameState, PlayerData
from rlgym.utils.reward_functions import RewardFunction
from rlgym.utils.gamestates import PhysicsObject


class CombinedReward(RewardFunction):
    """
    A comprehensive reward function that combines multiple objectives:
    - Ball direction towards opponent goal
    - Velocity towards ball
    - Distance to ball
    - Saving shots
    - Scoring goals
    - Boost management
    - Air control
    """

    def __init__(self):
        super().__init__()
        self.goal_reward = 10.0
        self.save_reward = 5.0
        self.ball_direction_weight = 1.0
        self.velocity_to_ball_weight = 0.5
        self.distance_weight = 0.3
        self.boost_weight = 0.1
        self.air_control_weight = 0.2

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        reward = 0.0
        
        # Ball direction towards opponent goal
        ball_pos = state.ball.position
        ball_vel = state.ball.linear_velocity
        
        # Determine which goal is the opponent's
        if player.team_num == 0:
            opponent_goal = np.array([0, 5120, 0])  # Orange goal
        else:
            opponent_goal = np.array([0, -5120, 0])  # Blue goal
        
        ball_to_goal = opponent_goal - ball_pos
        ball_to_goal_norm = ball_to_goal / np.linalg.norm(ball_to_goal)
        
        ball_direction_reward = np.dot(ball_vel, ball_to_goal_norm) / 2300.0
        reward += ball_direction_reward * self.ball_direction_weight
        
        # Velocity towards ball
        player_to_ball = ball_pos - player.car_data.position
        player_to_ball_norm = player_to_ball / np.linalg.norm(player_to_ball)
        velocity_to_ball = np.dot(player.car_data.linear_velocity, player_to_ball_norm)
        reward += velocity_to_ball / 2300.0 * self.velocity_to_ball_weight
        
        # Distance to ball (negative reward for being far)
        distance = np.linalg.norm(player_to_ball)
        distance_reward = -distance / 1000.0 * self.distance_weight
        reward += distance_reward
        
        # Boost management
        boost_reward = player.boost_amount / 100.0 * self.boost_weight
        reward += boost_reward
        
        # Air control (reward for being in air with control)
        if not player.on_ground:
            air_control = 1.0 - abs(player.car_data.angular_velocity).sum() / 10.0
            reward += air_control * self.air_control_weight
        
        return reward

    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        """Called at episode end to give final rewards"""
        reward = 0.0
        
        # Check for goals
        if state.last_touch == player.car_id:
            # Check if ball went into opponent goal
            ball_pos = state.ball.position
            if player.team_num == 0 and ball_pos[1] > 5120:
                reward += self.goal_reward
            elif player.team_num == 1 and ball_pos[1] < -5120:
                reward += self.goal_reward
        
        return reward

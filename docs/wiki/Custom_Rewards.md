# Customizing Reward Functions

This page explains how to customize the reward function for your RLGymExampleBot, which is crucial for guiding your agent's learning process.

## Understanding Reward Functions

In Reinforcement Learning, the reward function defines the goal of the agent. It provides a scalar feedback signal to the agent at each timestep, indicating how well it is performing. A well-designed reward function is essential for efficient and effective learning.

In this project, the reward function is primarily implemented within the `_calculate_reward` method of the `SimpleRocketLeagueEnv` class in `src/training_env.py`.

## Default Reward Components

The `_calculate_reward` method in `src/training_env.py` includes a comprehensive set of reward components designed to encourage various aspects of Rocket League gameplay. These include:

*   **Base Rewards**: Small positive rewards for staying alive or progressing through the game.
*   **Ball Control & Possession**: Rewards for being close to the ball, touching the ball, and maintaining possession.
*   **Scoring & Goals**: Significant positive rewards for scoring goals, and penalties for conceding goals.
*   **Strategic Positioning**: Rewards for defensive positioning (e.g., between ball and goal) and offensive positioning (e.g., in opponent's half).
*   **Movement & Mechanics**: Rewards for efficient movement, aerial play, wall riding, and other advanced mechanics.
*   **Boost Management**: Rewards for efficient boost usage and collecting boost pads.
*   **Ball Physics & Control**: Rewards for controlling ball direction towards the goal and maintaining optimal ball speed.
*   **Temporal & Consistency Rewards**: Rewards for consistent performance and quick responses.
*   **Risk Management**: Penalties for being out of bounds or reckless driving.
*   **Performance Milestones**: Rewards for achieving certain heights or speeds.
*   **Team Play & Coordination**: Rewards for passing plays (if applicable in a multi-agent setup).
*   **Difficulty-Based Rewards**: Adjustments to rewards based on the environment's difficulty setting.
*   **Momentum Rewards**: Rewards for building and maintaining speed.
*   **Skill Combos**: Rewards for executing sequences of advanced actions.
*   **Recovery**: Rewards for recovering from mistakes or bad positions.

## Creating Your Own Custom Reward Function

To customize the reward function, you will primarily modify the `_calculate_reward` method in `src/training_env.py`.

### `_calculate_reward` Method

This method takes the `action` taken by the agent as input and returns a scalar reward. You have access to the environment's internal state variables (e.g., `self.ball_position`, `self.car_position`, `self.score`, `self.boost_amount`) to calculate the reward.

```python
import numpy as np
# ... other imports and class definition ...

class SimpleRocketLeagueEnv(gym.Env):
    # ... __init__ and other methods ...

    def _calculate_reward(self, action):
        reward = 0.0

        # Example: Simple reward for hitting the ball
        # You would need to track ball touches in _update_physics or similar
        # if self.last_touch == "player":
        #     reward += 1.0

        # Example: Reward for moving towards the ball
        # ball_to_car_vec = self.ball_position - self.car_position
        # distance_to_ball = np.linalg.norm(ball_to_car_vec)
        # if distance_to_ball > 0:
        #     # Reward for reducing distance to ball
        #     reward += (self.last_ball_distance - distance_to_ball) * 0.1
        # self.last_ball_distance = distance_to_ball

        # Example: Reward for scoring a goal
        # if self.score > self.last_score:
        #     reward += 10.0
        # self.last_score = self.score

        # Combine all reward components
        # ... (your custom logic here) ...

        # Clamp reward to reasonable bounds
        reward = np.clip(reward, -5.0, 10.0) # Adjust bounds as needed

        return reward
```

### Tips for Designing Effective Reward Functions:

*   **Sparse vs. Dense Rewards**:
    *   **Sparse**: Rewards given only for achieving the ultimate goal (e.g., scoring a goal). Can be hard for agents to learn from.
    *   **Dense**: Rewards given for progress towards the goal (e.g., moving towards the ball, hitting the ball). Easier for agents to learn from, but can lead to unintended behaviors if not carefully designed.
*   **Shaping Rewards**: Provide intermediate rewards that guide the agent towards desired behaviors without explicitly telling it how to achieve the goal.
*   **Penalties**: Use negative rewards (penalties) for undesirable actions or states (e.g., going out of bounds, conceding a goal).
*   **Normalization**: Ensure reward values are scaled appropriately to prevent one reward component from dominating others.
*   **Experimentation**: Reward function design is often an iterative process. Experiment with different components and weights to find what works best for your specific task.

By customizing the reward function, you can directly influence what your bot learns and how it behaves in the Rocket League environment.

[Home](Home.md)
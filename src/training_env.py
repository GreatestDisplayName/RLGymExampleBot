import gymnasium as gym
from gymnasium import spaces
import numpy as np
from logger import logger
from config import config


class SimpleRocketLeagueEnv(gym.Env):
    """
    Enhanced simple Rocket League environment for training and self-play
    
    This environment provides a simplified but more realistic simulation
    of Rocket League gameplay for training reinforcement learning agents.
    """
    
    def __init__(self, max_steps=1000, difficulty=1.0, agent_type=None):
        super().__init__()
        
        # Store agent type for action space handling
        self.agent_type = agent_type or "PPO"  # Default to PPO if not specified
        
        # Environment configuration with defaults
        self.max_steps = max_steps
        self.difficulty = difficulty
        self.current_step = 0
        
        # Game state tracking
        self.ball_position = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # x, y, z
        self.ball_velocity = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.car_position = np.array([0.0, 0.0, 18.0], dtype=np.float32)  # Start slightly above ground
        self.car_velocity = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.car_rotation = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # pitch, yaw, roll
        self.goal_position = np.array([0.0, 5120.0, 0.0], dtype=np.float32)  # Default goal position
        
        # Game mechanics
        self.boost_amount = 100.0
        self.on_ground = True
        self.has_flip = True
        self.last_touch = None
        self.score = 0
        self.opponent_score = 0
        
        # Physics constants with defaults if not in config
        self.gravity = getattr(config.environment, 'gravity', -650.0)
        self.ball_drag = getattr(config.environment, 'ball_drag', 0.03)
        self.car_drag = getattr(config.environment, 'car_drag', 0.1)
        self.boost_force = getattr(config.environment, 'boost_force', 1000.0)
        self.jump_force = getattr(config.environment, 'jump_force', 500.0)
        self.flip_force = getattr(config.environment, 'flip_force', 500.0)
        
        # Define action space based on agent type
        # Agents that require continuous actions
        continuous_agents = ["PPO", "SAC", "TD3"]
        action_parser = getattr(config.environment, 'action_parser', '')

        if self.agent_type in continuous_agents and action_parser != "DiscreteAction":
            # Continuous action space for PPO, SAC, TD3
            self.action_space = spaces.Box(
                low=-1.0, high=1.0, shape=(8,), dtype=np.float32
            )
        else:
            # Discrete action space for DQN, A2C, or if specified in config
            self.action_space = spaces.Discrete(10)

        # Define observation space (43 features)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(43,), dtype=np.float32
        )
        
        # Initialize reward tracking variables
        self.consecutive_touches = 0
        self.consistent_steps = 0
        self.last_ball_touch_time = 0
        self.last_car_position = np.array([0.0, 0.0, 0.0])
        self.last_ball_position = np.array([0.0, 0.0, 0.0])
        self.last_car_speed = 0.0
        self.power_hit_bonus = False
        self.last_boost_amount = 100.0
        self.ball_possession_time = 0
        self.skill_combo_count = 0
        self.last_reward = 0.0

        # Initialize game state
        self._reset_game_state()
        
        logger.info(f"SimpleRocketLeagueEnv initialized with {self.max_steps} max steps, difficulty: {self.difficulty}")
        
        
    
    def _reset_game_state(self):
        """Reset the internal game state"""
        # Reset positions
        self.ball_position = np.array([0.0, 0.0, 0.0])
        self.ball_velocity = np.array([0.0, 0.0, 0.0])
        self.car_position = np.array([0.0, 0.0, 0.0])
        self.car_velocity = np.array([0.0, 0.0, 0.0])
        self.car_rotation = np.array([0.0, 0.0, 0.0])
        
        # Reset game mechanics
        self.boost_amount = 100.0
        self.on_ground = True
        self.has_flip = True
        self.last_touch = None
        self.score = 0
        self.opponent_score = 0
        
        # Reset reward tracking variables
        self.consecutive_touches = 0
        self.consistent_steps = 0
        self.last_ball_touch_time = 0
        self.last_car_position = np.array([0.0, 0.0, 0.0])
        self.last_ball_position = np.array([0.0, 0.0, 0.0])
        self.last_car_speed = 0.0
        self.power_hit_bonus = False
        self.last_boost_amount = 100.0
        self.ball_possession_time = 0
        self.skill_combo_count = 0
        self.last_reward = 0.0
        
        # Set goal position based on difficulty
        # Assuming a single goal at one end of the field for simplicity
        self.goal_position = np.array([0.0, 5120.0, 0.0]) # Example: one end of the field

    def _discrete_to_continuous(self, discrete_action):
        # Create a continuous action array
        continuous_action = np.zeros(8, dtype=np.float32)
        
        # Default is no-op (all zeros)
        if discrete_action == 0:
            return continuous_action
            
        # Movement actions (1-4)
        if discrete_action == 1:  # Forward
            continuous_action[0] = 1.0  # Throttle
        elif discrete_action == 2:  # Backward
            continuous_action[0] = -1.0  # Reverse
        elif discrete_action == 3:  # Left
            continuous_action[1] = -1.0  # Steer left
        elif discrete_action == 4:  # Right
            continuous_action[1] = 1.0   # Steer right
            
        # Rotation actions (5-6)
        elif discrete_action == 5:  # Pitch up
            continuous_action[2] = 1.0  # Pitch up
        elif discrete_action == 6:  # Pitch down
            continuous_action[2] = -1.0  # Pitch down
            
        # Special actions (7-9)
        elif discrete_action == 7:  # Jump
            continuous_action[5] = 1.0  # Jump
        elif discrete_action == 8:  # Boost
            continuous_action[6] = 1.0  # Boost
        elif discrete_action == 9:  # Handbrake
            continuous_action[7] = 1.0  # Handbrake

        return continuous_action
    
    def reset(self, seed=None, options=None):
        """Reset the environment to initial state"""
        super().reset(seed=seed)
        
        # Reset step counter
        self.current_step = 0
        
        # Reset game state
        self._reset_game_state()
        
        # Generate initial observation for both cars
        obs_a = self._get_observation(0)
        obs_b = self._get_observation(1)
        obs = (obs_a, obs_b)
        
        info = {
            "step": self.current_step,
            "score": self.score,
            "opponent_score": self.opponent_score,
            "boost": self.boost_amount,
            "ball_position": self.ball_position.copy(),
            "car_position": self.car_position.copy()
        }
        
        return obs, info
    
    def step(self, action):
        """Execute one step in the environment"""
        if isinstance(self.action_space, spaces.Discrete):
            action = self._discrete_to_continuous(action)
        else:
            # Validate action
            action = np.clip(action, -1.0, 1.0)
        
        # Increment step counter
        self.current_step += 1
        
        # Apply action to game state
        self._apply_action(action)
        
        # Update physics
        self._update_physics()
        
        # Check for game events
        self._check_game_events()
        
        # Generate observation
        obs = self._get_observation()
        
        # Calculate reward
        reward = self._calculate_reward(action)
        
        # Check if episode is done
        done = self._is_episode_done()
        
        # Truncated flag for gymnasium
        truncated = self.current_step >= self.max_steps
        
        # Prepare info
        info = {
            "step": self.current_step,
            "score": self.score,
            "opponent_score": self.opponent_score,
            "boost": self.boost_amount,
            "ball_position": self.ball_position.copy(),
            "car_position": self.car_position.copy(),
            "ball_velocity": self.ball_velocity.copy(),
            "car_velocity": self.car_velocity.copy(),
            "on_ground": self.on_ground,
            "has_flip": self.has_flip
        }
        
        return obs, reward, done, truncated, info
    
    def _apply_action(self, action):
        """Apply the agent's action to the game state"""
        # Extract action components
        throttle, steer, pitch, yaw, roll, jump, boost, handbrake = action
        
        # Apply throttle (forward/backward movement)
        throttle_force = throttle * 10.0
        self.car_velocity[0] += throttle_force * 0.1
        
        # Apply steering (left/right movement)
        steer_force = steer * 5.0
        self.car_velocity[1] += steer_force * 0.1
        
        # Apply rotation changes
        self.car_rotation[0] += pitch * 0.1  # pitch
        self.car_rotation[1] += yaw * 0.1    # yaw
        self.car_rotation[2] += roll * 0.1   # roll
        
        # Apply jump
        if jump > 0.5 and self.on_ground and self.has_flip:
            self.car_velocity[2] += self.jump_force
            self.on_ground = False
            self.has_flip = False
        
        # Apply boost
        if boost > 0.5 and self.boost_amount > 0:
            boost_force = boost * self.boost_force
            # Boost in the direction the car is facing
            boost_direction = np.array([
                np.cos(self.car_rotation[1]),
                np.sin(self.car_rotation[1]),
                0.0
            ])
            self.car_velocity += boost_direction * boost_force * 0.1
            self.boost_amount = max(0, self.boost_amount - 0.5)
        
        # Apply handbrake (affects turning)
        if handbrake > 0.5:
            self.car_velocity[1] *= 0.8  # Reduce lateral movement
    
    def _update_physics(self):
        """Update physics simulation"""
        # Update car position
        self.car_position += self.car_velocity * 0.1
        
        # Apply gravity to car
        if not self.on_ground:
            self.car_velocity[2] += self.gravity * 0.1
        
        # Apply drag to car
        self.car_velocity *= (1.0 - self.car_drag)
        
        # Check if car is on ground
        if self.car_position[2] <= 0:
            self.car_position[2] = 0
            self.car_velocity[2] = 0
            self.on_ground = True
            self.has_flip = True
        
        # Update ball physics
        self.ball_position += self.ball_velocity * 0.1
        
        # Apply gravity to ball
        self.ball_velocity[2] += self.gravity * 0.1
        
        # Apply drag to ball
        self.ball_velocity *= (1.0 - self.ball_drag)
        
        # Ball collision with ground
        if self.ball_position[2] <= 0:
            self.ball_position[2] = 0
            self.ball_velocity[2] = -self.ball_velocity[2] * 0.7  # Bounce with energy loss
        
        # Ball collision with walls (simplified)
        for i in range(3):
            if abs(self.ball_position[i]) > 50:  # Wall boundary
                self.ball_position[i] = np.sign(self.ball_position[i]) * 50
                self.ball_velocity[i] = -self.ball_velocity[i] * 0.8
        
        # Car-ball collision detection
        distance = np.linalg.norm(self.car_position - self.ball_position)
        if distance < 2.0:  # Collision threshold
            # Enhanced collision response based on car velocity and angle
            collision_force = (2.0 - distance) * 5.0
            
            # Add car velocity influence to ball
            car_velocity_contribution = np.dot(self.car_velocity, 
                                             (self.ball_position - self.car_position) / (distance + 1e-6))
            collision_force += max(0, car_velocity_contribution) * 0.5
            
            collision_direction = (self.ball_position - self.car_position) / (distance + 1e-6)
            
            # Apply collision force to ball
            self.ball_velocity += collision_direction * collision_force
            
            # Update last touch and add small reward for successful hit
            self.last_touch = "player"
            
            # Add reward for powerful hits (when car is moving fast)
            if np.linalg.norm(self.car_velocity) > 10.0:
                self.power_hit_bonus = True  # Flag for reward calculation
            else:
                self.power_hit_bonus = False
    
    def _check_game_events(self):
        """Check for game events like goals"""
        # Check if ball is in goal (simplified)
        goal_distance = np.linalg.norm(self.ball_position - self.goal_position)
        if goal_distance < 5.0 and self.ball_velocity[0] > 5.0:  # Ball moving toward goal
            if self.last_touch == "player":
                self.score += 1
                logger.debug(f"Goal scored! Score: {self.score}")
            else:
                self.opponent_score += 1
                logger.debug(f"Opponent goal! Score: {self.opponent_score}")
            
            # Reset ball position
            self.ball_position = np.array([0.0, 0.0, 0.0])
            self.ball_velocity = np.array([0.0, 0.0, 0.0])
            self.last_touch = None
    
    def _calculate_reward(self, action):
        """Calculate comprehensive reward based on game state and actions"""
        reward = 0.0
        
        # ===== BASE REWARDS =====
        # Base reward for staying alive (encourages survival)
        reward += 0.01
        
        # ===== BALL CONTROL & POSSESSION =====
        # Reward for being close to the ball (proximity reward)
        ball_distance = np.linalg.norm(self.car_position - self.ball_position)
        if ball_distance < 10.0:
            reward += (10.0 - ball_distance) * 0.02  # Increased from 0.01
        
        # Reward for touching the ball (possession reward)
        if self.last_touch == "player":
            reward += 0.2  # Increased from 0.1
            
            # Bonus for powerful hits (when car is moving fast)
            if self.power_hit_bonus:
                reward += 0.05  # Extra reward for powerful hits
        
        # Reward for maintaining ball possession (consecutive touches)
        if self.last_touch == "player":
            self.consecutive_touches += 1
            reward += self.consecutive_touches * 0.05  # Bonus for possession chains
        else:
            self.consecutive_touches = 0
        
        # ===== SCORING & GOALS =====
        # Reward for scoring goals (major achievement)
        if self.score > 0:
            reward += self.score * 15.0  # Increased from 10.0
        
        # Penalty for opponent goals (defensive failure)
        if self.opponent_score > 0:
            reward -= self.opponent_score * 8.0  # Increased penalty
        
        # ===== STRATEGIC POSITIONING =====
        # Reward for being between ball and goal (defensive positioning)
        car_to_ball = self.ball_position - self.car_position
        car_to_goal = self.goal_position - self.car_position
        
        # Check if car is in good defensive position
        if np.dot(car_to_ball, car_to_goal) > 0:  # Car is between ball and goal
            reward += 0.05
        
        # Reward for saves (preventing goals)
        # Check if ball was moving toward goal but got stopped
        ball_to_goal_distance = np.linalg.norm(self.ball_position - self.goal_position)
        last_ball_to_goal_distance = np.linalg.norm(self.last_ball_position - self.goal_position)
        
        if (last_ball_to_goal_distance < ball_to_goal_distance and 
            last_ball_to_goal_distance < 15.0 and 
            self.last_touch == "player"):
            reward += 0.1  # Reward for save
        
        # Reward for being in offensive position (closer to opponent goal)
        if self.car_position[0] > 0:  # In opponent half
            reward += 0.02
        
        # Reward for center field positioning (good for transitions)
        center_distance = np.linalg.norm(self.car_position[:2])  # Only x,y
        if center_distance < 20.0:
            reward += (20.0 - center_distance) * 0.01
        
        # ===== MOVEMENT & MECHANICS =====
        # Reward for efficient movement (not standing still)
        car_speed = np.linalg.norm(self.car_velocity)
        if car_speed > 1.0:  # Moving at reasonable speed
            reward += 0.01
        elif car_speed < 0.1:  # Penalty for being stationary
            reward -= 0.02
        
        # Reward for aerial play (advanced skill)
        if self.car_position[2] > 5.0:  # Above ground
            reward += 0.03
        
        # Reward for wall riding (another advanced skill)
        wall_distance = min(abs(self.car_position[0]), abs(self.car_position[1]))
        if wall_distance < 3.0 and self.car_position[2] > 2.0:
            reward += 0.02
        
        # Reward for ceiling touches (very advanced skill)
        if self.car_position[2] > 30.0:
            reward += 0.1  # Significant reward for ceiling play
        
        # Reward for wall hits (ball bouncing off walls)
        if hasattr(self, 'last_ball_position'):
            # Check if ball hit a wall
            for i in range(3):
                if abs(self.ball_position[i]) > 45 and abs(self.last_ball_position[i]) <= 45:
                    if self.last_touch == "player":
                        reward += 0.03  # Reward for wall shots
        
        # Reward for smooth movement (reduced jitter)
        position_change = np.linalg.norm(self.car_position - self.last_car_position)
        if position_change < 2.0:  # Smooth movement
            reward += 0.01
        self.last_car_position = self.car_position.copy()
        
        # ===== BOOST MANAGEMENT =====
        # Reward for boost management (strategic resource)
        if self.boost_amount > 50.0:
            reward += 0.02  # Good boost reserves
        elif self.boost_amount < 10.0:
            reward -= 0.01  # Low boost warning
        
        # Reward for boost efficiency (using boost when needed)
        if self.boost_amount > 0 and car_speed > 5.0:
            reward += 0.01  # Using boost effectively
        
        # Reward for boost pad collection (simulated)
        if self.boost_amount > self.last_boost_amount:
            reward += 0.05  # Reward for collecting boost
        self.last_boost_amount = self.boost_amount
        
        # ===== BALL PHYSICS & CONTROL =====
        # Reward for controlling ball direction (toward goal)
        if self.last_touch == "player":
            ball_to_goal = self.goal_position - self.ball_position
            ball_velocity_magnitude = np.linalg.norm(self.ball_velocity)
            if ball_velocity_magnitude > 0.1:
                ball_direction = self.ball_velocity / ball_velocity_magnitude
                alignment = np.dot(ball_direction, ball_to_goal) / np.linalg.norm(ball_to_goal)
                if alignment > 0.5:  # Ball moving toward goal
                    reward += 0.1
                elif alignment < -0.5:  # Ball moving away from goal
                    reward -= 0.05
        
        # Reward for ball speed control (not just hitting hard)
        if self.last_touch == "player":
            if 2.0 < ball_velocity_magnitude < 15.0:  # Good ball speed range
                reward += 0.05
        
        # Reward for dribbling (keeping ball close while moving)
        if (ball_distance < 5.0 and car_speed > 2.0 and 
            self.last_touch == "player"):
            reward += 0.03  # Dribbling reward
        
        # Reward for ball possession time
        if self.last_touch == "player":
            self.ball_possession_time += 1
            if self.ball_possession_time > 20:  # Sustained possession
                reward += 0.02
        else:
            self.ball_possession_time = 0
        
        # ===== TEMPORAL & CONSISTENCY REWARDS =====
        # Reward for consistent performance over time
        if reward > 0:
            self.consistent_steps += 1
            if self.consistent_steps > 10:
                reward += 0.01  # Bonus for sustained good performance
        else:
            self.consistent_steps = 0
        
        # Reward for quick responses (reaction time)
        if self.last_touch == "player":
            response_time = self.current_step - self.last_ball_touch_time
            if response_time < 5:  # Quick response
                reward += 0.02
            self.last_ball_touch_time = self.current_step
        
        # ===== RISK MANAGEMENT =====
        # Penalty for being out of bounds (safety)
        if (abs(self.car_position[0]) > 50 or 
            abs(self.car_position[1]) > 50 or 
            self.car_position[2] < -10):
            reward -= 2.0  # Increased penalty
        
        # Penalty for excessive actions (encourage efficiency)
        action_magnitude = np.linalg.norm(action)
        if action_magnitude > 1.5:  # Only penalize excessive actions
            reward -= (action_magnitude - 1.5) * 0.002
        
        # Penalty for reckless driving (high speed near walls)
        wall_proximity = min(abs(self.car_position[0]), abs(self.car_position[1]))
        if wall_proximity < 5.0 and car_speed > 10.0:
            reward -= 0.05
        
        # ===== PERFORMANCE MILESTONES =====
        # Reward for achieving certain heights (aerial milestones)
        if self.car_position[2] > 10.0:
            reward += 0.05  # High aerial
        if self.car_position[2] > 20.0:
            reward += 0.1   # Very high aerial
        
        # Reward for maintaining high speed
        if car_speed > 15.0:
            reward += 0.02  # High speed bonus
        if car_speed > 25.0:
            reward += 0.05  # Very high speed bonus
        
        # ===== TEAM PLAY & COORDINATION =====
        # Reward for passing plays (ball moving in good direction after touch)
        ball_movement = self.ball_position - self.last_ball_position
        if np.linalg.norm(ball_movement) > 1.0:  # Ball moved significantly
            if self.last_touch == "player":
                # Check if movement is toward goal
                goal_direction = self.goal_position - self.last_ball_position
                movement_alignment = np.dot(ball_direction, goal_direction)
                if movement_alignment > 0:
                    reward += 0.03  # Good pass toward goal
        self.last_ball_position = self.ball_position.copy()
        
        # ===== DIFFICULTY-BASED REWARDS =====
        # Adjust rewards based on difficulty setting
        if self.difficulty == "hard":
            # Higher rewards for advanced skills in hard mode
            if self.car_position[2] > 5.0:
                reward *= 1.2  # Bonus for aerial play in hard mode
        elif self.difficulty == "easy":
            # Lower penalties in easy mode
            if reward < 0:
                reward *= 0.8  # Reduce penalties in easy mode
        
        # ===== MOMENTUM REWARDS =====
        # Reward for building momentum (increasing speed)
        if car_speed > self.last_car_speed:
            reward += 0.01  # Building momentum
        self.last_car_speed = car_speed
        
        # Reward for skill combos (multiple advanced actions in sequence)
        if (self.car_position[2] > 5.0 or  # Aerial
            wall_distance < 3.0 or          # Wall ride
            car_speed > 15.0):              # High speed
            self.skill_combo_count += 1
            if self.skill_combo_count > 3:
                reward += 0.05  # Combo bonus
        else:
            self.skill_combo_count = 0
        
        # Reward for recovery (getting back on track after mistakes)
        if reward > self.last_reward and self.last_reward < 0:
            reward += 0.02  # Recovery bonus
        self.last_reward = reward
        
        # ===== FINAL REWARD PROCESSING =====
        # Clamp reward to reasonable bounds to prevent extreme values
        reward = np.clip(reward, -5.0, 10.0)
        
        return reward
    
    def _is_episode_done(self):
        """Check if the episode should end"""
        # End if score difference is too large
        if abs(self.score - self.opponent_score) >= 5:
            return True
        
        # End if car is out of bounds
        if (abs(self.car_position[0]) > 100 or 
            abs(self.car_position[1]) > 100 or 
            self.car_position[2] < -20):
            return True
        
        return False
    
    def _get_observation(self, car_id):
        """Generate the observation vector for a specific car"""
        obs = np.zeros(43, dtype=np.float32)

        # Car information (0-8)
        obs[0:3] = self.car_positions[car_id] / 100.0  # Normalize position
        obs[3:6] = self.car_velocities[car_id] / 20.0   # Normalize velocity
        obs[6:9] = self.car_rotations[car_id] / np.pi  # Normalize rotation

        # Ball information (9-17)
        obs[9:12] = self.ball_position / 100.0
        obs[12:15] = self.ball_velocity / 20.0
        obs[15:18] = [self.boost_amounts[car_id] / 100.0, float(self.on_grounds[car_id]), float(self.has_flips[car_id])]

        # Goal information (18-20) - relative to the car's goal
        obs[18:21] = self.goal_positions[car_id] / 100.0

        # Game state (21-23)
        obs[21] = self.scores[car_id] / 10.0 # Own score
        obs[22] = self.scores[1 - car_id] / 10.0 # Opponent score
        obs[23] = self.current_step / self.max_steps

        # Relative positions and velocities (24-47)
        car_to_ball = self.ball_position - self.car_positions[car_id]
        car_to_goal = self.goal_positions[car_id] - self.car_positions[car_id]

        obs[24:27] = car_to_ball / 100.0
        obs[27:30] = car_to_goal / 100.0

        # Distance metrics (30-35)
        obs[30] = np.linalg.norm(car_to_ball) / 100.0
        obs[31] = np.linalg.norm(car_to_goal) / 100.0
        obs[32] = np.linalg.norm(self.car_velocities[car_id]) / 20.0
        obs[33] = np.linalg.norm(self.ball_velocity) / 20.0

        # Angular information (34-40)
        obs[34] = np.sin(self.car_rotations[car_id][0])  # pitch
        obs[35] = np.cos(self.car_rotations[car_id][0])
        obs[36] = np.sin(self.car_rotations[car_id][1])  # yaw
        obs[37] = np.cos(self.car_rotations[car_id][1])
        obs[38] = np.sin(self.car_rotations[car_id][2])  # roll
        obs[39] = np.cos(self.car_rotations[car_id][2])

        # Game mechanics (40-50)
        obs[40] = float(self.last_touches[car_id] == car_id)
        obs[41] = float(self.last_touches[car_id] == (1 - car_id)) # Touched by opponent
        obs[42] = float(self.last_touches[car_id] is None)

        return obs
    
    def render(self, mode='human'):
        """Render the environment (placeholder)"""
        if mode == 'human':
            # In a real scenario, this would render the game state
            # For a simple text-based environment, you might print key info
            pass # No visual rendering for this simple env
        super().render() # Call parent render if it has functionality

    def close(self):
        """Clean up environment resources"""
        super().close() # Call parent close if it has functionality


def make_training_env(max_steps=1000, difficulty="medium"):
    """Create a training environment with specified parameters"""
    env = SimpleRocketLeagueEnv(max_steps=max_steps, difficulty=difficulty)
    return env



def get_env_info(env):
    """Get information about the environment"""
    obs_space = env.observation_space
    action_space = env.action_space
    
    logger.environment_info(obs_space, action_space)
    
    return obs_space, action_space


if __name__ == "__main__":
    # Test the enhanced environment
    logger.info("🧪 Testing Enhanced SimpleRocketLeagueEnv...")
    
    env = make_training_env(max_steps=500, difficulty="medium")
    get_env_info(env)
    
    # Test a few steps
    obs, info = env.reset()
    logger.info(f"Initial observation shape: {obs[0].shape} (Car A), {obs[1].shape} (Car B)")
    logger.info(f"Initial info: {info}")
    
    total_reward_a = 0.0
    total_reward_b = 0.0
    
    for i in range(20):
        action_a = env.single_action_space.sample()
        action_b = env.single_action_space.sample()
        obs, rewards, done, truncated, info = env.step((action_a, action_b))
        total_reward_a += rewards[0]
        total_reward_b += rewards[1]
        
        logger.debug(f"Step {i}: Rewards = {rewards[0]:.3f} (A), {rewards[1]:.3f} (B), Done = {done}, Truncated = {truncated}")
        logger.debug(f"   Car A pos: {info['car_position_a']}, Car B pos: {info['car_position_b']}, Ball pos: {info['ball_position']}")
        logger.debug(f"   Score: {info['score_a']}-{info['score_b']}, Boost A: {info['boost_a']:.1f}, Boost B: {info['boost_b']:.1f}")
        
        if done or truncated:
            logger.info(f"Episode ended after {i+1} steps. Total rewards: {total_reward_a:.3f} (A), {total_reward_b:.3f} (B)")
            obs, info = env.reset()
            total_reward_a = 0.0
            total_reward_b = 0.0
    
    env.close()
    logger.info("✅ Enhanced environment test completed!")
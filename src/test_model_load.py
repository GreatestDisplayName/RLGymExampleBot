from stable_baselines3 import PPO
from rlgym_compat import GameState
from training_env import make_training_env

model_path = "models/PPO/final_model.zip"

try:
    env = make_training_env()
    model = PPO.load(model_path, env=env)
    print(f"Successfully loaded model from {model_path}")
    env.close()
except Exception as e:
    print(f"Error loading model: {e}")

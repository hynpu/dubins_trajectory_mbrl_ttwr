import gymnasium as gym
import ttwrPathFollow
import numpy as np

from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.noise import NormalActionNoise

env = gym.make('ttwrPathFollow/ttwrPathFollow-v0', controlMode='lateralDeviation')

# Create 4 artificial transitions per real transition
n_sampled_goal = 4

# PPO hyperparams:
model = PPO("MlpPolicy", env, device="cuda", n_steps = 1000, verbose=0, tensorboard_log="./ttwr_freeMap/")
# model = RecurrentPPO("MlpLstmPolicy", env, device="cuda", learning_rate=1e-3, verbose=0, n_steps = 350,
#         batch_size=256, tensorboard_log="./ttwr_freeMap/")

model.learn(total_timesteps=5e5)
model.save("ppo_ttwr")

# Load saved model
# Because it needs access to `env.compute_reward()`
# model weights must be loaded with the env
env = gym.make('ttwrPathFollow/ttwrPathFollow-v0', render_mode="human", controlMode='lateralDeviation') # Change the render mode
model = PPO.load("ppo_ttwr", env=env)

obs, info = env.reset() #trailer_state = np.array([25, 15, -np.pi/3, 0]), target_state = np.array([-25, -10, 0, 0])

# Evaluate the agent for N times
episode_reward = 0
for _ in range(1000):
    env.render()
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    episode_reward += reward
    if terminated or truncated or info.get("is_success", False):
        print("Reward:", episode_reward, "Success?", info.get("is_success", False))
        episode_reward = 0.0
        obs, info = env.reset()

# python -m tensorboard.main --logdir=D:\github\ttwr_gym\ppo_ttwr_tensorboard
"""Train a PPO agent to play Flappy Bird.

This script trains a PPO (Proximal Policy Optimization) agent using Stable-Baselines3.
The trained model is saved as 'flappy_bird_ppo.zip'.
"""
import flappy_bird_gymnasium
import gymnasium
import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# Silencing pygame
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"

print("🐦 Flappy Bird PPO Training")
print("Setting up environment...")

# Create environment
env = make_vec_env("FlappyBird-v0", n_envs=1, env_kwargs={"use_lidar": False})

print("Creating PPO model...")
# Create PPO model
model = PPO(
    "MlpPolicy", 
    env, 
    verbose=1,
    learning_rate=0.0003,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99
)

print("Training for 1,000,000 steps...")
# Train the model
model.learn(total_timesteps=1000000)

# Save the model
model.save("flappy_bird_ppo")
print("Model saved to flappy_bird_ppo.zip!")

# Test the trained model
print("\nTesting the trained AI...")
test_env = gymnasium.make("FlappyBird-v0", render_mode="human", use_lidar=False)

for test_episode in range(5):
    obs, _ = test_env.reset()
    total_score = 0
    
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = test_env.step(action)
        total_score = info.get('score', 0)
        
        if done:
            break
    
    print(f"Test {test_episode + 1}: Score = {total_score}")

test_env.close()
print("Done! PPO learned to play Flappy Bird.")

import gym

env = gym.make("CarRacing-v2", render_mode="human")
obs, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()  # 随机动作
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()

env.close()

input("Press Enter to exit...")

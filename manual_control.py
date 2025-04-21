import gym
import pygame
from pygame.locals import *
import time

# ⬇️ 你要在这里加入创建环境的这行代码！
env = gym.make("CarRacing-v2", render_mode="human")

# 初始化 Pygame 用于键盘监听
pygame.init()
screen = pygame.display.set_mode((100, 100))
pygame.display.set_caption("CarRacing Manual Control")

obs, info = env.reset()

running = True
clock = pygame.time.Clock()

# 初始化动作：⬇️ 你可以在这里加入 action 的注释和格式说明
action = [0.0, 0.0, 0.0]  # [steering, gas, brake]
# steering: -1 ~ +1，gas: 0 ~ 1，brake: 0 ~ 1

while running:
    for event in pygame.event.get():
        if event.type == QUIT:
            running = False

    # 获取键盘状态
    keys = pygame.key.get_pressed()

    # 键盘控制逻辑更新动作
    action[0] = 0.0  # steering
    if keys[K_LEFT]:
        action[0] = -1.0
    if keys[K_RIGHT]:
        action[0] = 1.0

    action[1] = 0.0  # gas
    if keys[K_UP]:
        action[1] = 1.0

    action[2] = 0.0  # brake
    if keys[K_DOWN]:
        action[2] = 0.8

    # 执行动作
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()

    # 控制帧率
    clock.tick(50)

env.close()
pygame.quit()

#用tilt矩形方块生成赛道，解决了赛道交叉
import gym
from gym import spaces
import numpy as np
import Box2D
from Box2D.b2 import (world, polygonShape, staticBody, dynamicBody, vec2)
import pygame
import math
from shapely.geometry import Polygon

# =================== 全局设置 =====================
SCALE = 0.3 # 数值越小，画面比例尺越小。单位好像是像素，我忘了。
FPS = 50
VIEWPORT_W = 800
VIEWPORT_H = 600

class CustomRacingEnv(gym.Env):
    def __init__(self):
        super(CustomRacingEnv, self).__init__()

        # 初始化物理世界
        self.world = world(gravity=(0, 0), doSleep=True)
        self.car = None
        self.track = []

        # 状态空间（速度、方向角cos/sin、是否倒退）
        high = np.array([100.0, 1.0, 1.0, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        # 动作空间（转向Δ，油门Δ）
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32
        )

        self.viewer = None
        self.reset()

    def create_track(self):
        self.track = []
        self.road_polys = []

        length = 100  # tile数量，即赛道分段数量，总长度是100*10=1000
        tile_size = 10  # tile尺寸，单位可理解为米
        road_width = 15  # 赛道宽度，单位可理解为米
        angle = 0.0 # 我猜是起点指向
        x, y = 0.0, 0.0 # 起点坐标
        start_x, start_y = x, y  # 记录起点位置

        # 存储所有中心点用于生成平滑赛道
        center_points = [(x, y)]

        for i in range(length):
            # 调整转向概率和幅度
            turn = np.random.choice([-1, 0, 1], p=[0.25, 0.5, 0.25]) # 直行概率50%，左右转各25%
            angle += turn * (np.pi / 8)  # 转向角度+
        
            dx = tile_size * np.cos(angle)
            dy = tile_size * np.sin(angle)
            nx = x + dx
            ny = y + dy
            center_points.append((nx, ny))

            # 计算法向量（垂直于前进方向）
            perp = np.array([-dy, dx])
            perp = perp / np.linalg.norm(perp) * road_width / 2

            # 计算赛道边界
            p1 = (x + perp[0], y + perp[1])
            p2 = (nx + perp[0], ny + perp[1])
            p3 = (nx - perp[0], ny - perp[1])
            p4 = (x - perp[0], y - perp[1])

            # 新增：检测是否交叉（shapely）
            new_poly = Polygon([p1, p2, p3, p4])
            # for existing in self.road_polys:
            #     if new_poly.intersects(Polygon(existing)):
            #         raise ValueError("赛道 tile 发生交叉，重新生成")
            for j, existing in enumerate(self.road_polys):
                if abs(i - j) <= 1:
                    continue  # 跳过前一个/自己/下一块
                if new_poly.overlaps(Polygon(existing)):
                    print(f"Tile {i} overlaps with existing tile {j}")
                    raise ValueError("赛道 tile 发生交叉，重新生成")
            
            # 创建物理碰撞体
            tile = self.world.CreateStaticBody(
                shapes=polygonShape(vertices=[p1, p2, p3, p4])
            )
            self.track.append(tile)
            self.road_polys.append([p1, p2, p3, p4])

            x, y = nx, ny  # 更新位置

        # 添加连接起点和终点的赛道段，形成闭环
        if length > 0:
            dx = start_x - x
            dy = start_y - y
            perp = np.array([-dy, dx])
            perp = perp / np.linalg.norm(perp) * road_width / 2
        
            p1 = (x + perp[0], y + perp[1])
            p2 = (start_x + perp[0], start_y + perp[1])
            p3 = (start_x - perp[0], start_y - perp[1])
            p4 = (x - perp[0], y - perp[1])
            
            # 检查闭环连线是否与之前随机转向生成的赛道冲突
            new_poly = Polygon([p1, p2, p3, p4])
            for j, existing in enumerate(self.road_polys):
                # 注意：这里 tile 之间距离差比较大，可以全检查
                if j in range(0, 4) or j in range(length - 4, length):
                    continue  # tilt100与tile 0，1，2，3，96，97，98，99的交叉都不算作赛道交叉
                if new_poly.overlaps(Polygon(existing)):
                    print(f"Tile {length} overlaps with tile {j}")
                    raise ValueError("闭环 tile 与之前赛道交叉，重新生成")
    
            closing_tile = self.world.CreateStaticBody(
                shapes=polygonShape(vertices=[p1, p2, p3, p4])
            )
            self.track.append(closing_tile)
            self.road_polys.append([p1, p2, p3, p4])

        return center_points  # 返回中心点可用于调试
	

    def create_car(self):
        return self.world.CreateDynamicBody(
            position=(0, 0),  # 从赛道起点开始
            angle=0.0,
            fixtures=Box2D.b2FixtureDef(
                shape=polygonShape(box=(2.5, 1.5)),
                density=1.0,
                friction=0.3
            )
        )
        return car_body

    def reset(self):
        max_attempts = 20
        for attempt in range(max_attempts):
            try:
                self.world = world(gravity=(0, 0), doSleep=True)
                self.create_track()
                self.car = self.create_car()
                self.time = 0
                return self._get_obs()
            except ValueError as e:
                print(f"[RESET] 第 {attempt+1} 次尝试失败：{e}")
        raise RuntimeError("多次尝试仍无法生成有效、不交叉的赛道")

    # 原来的reset，没加多次尝试
    # def reset(self):
    #    self.world = world(gravity=(0, 0), doSleep=True)
    #    self.create_track()
    #    self.car = self.create_car()
    #    self.time = 0
    #    return self._get_obs()

    def _get_obs(self):
        lin_vel = self.car.linearVelocity
        speed = np.linalg.norm([lin_vel.x, lin_vel.y])
        angle = self.car.angle
        backward = 1 if np.dot([lin_vel.x, lin_vel.y], [np.cos(angle), np.sin(angle)]) < 0 else 0
        return np.array([speed, np.cos(angle), np.sin(angle), backward], dtype=np.float32)

    def step(self, action):
        Δsteer, Δacc = float(action[0]), float(action[1])
        forward = vec2(math.cos(self.car.angle), math.sin(self.car.angle))
        self.car.ApplyForceToCenter(100 * Δacc * forward, True)
        self.car.ApplyTorque(15 * Δsteer, True)

        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)
        self.time += 1

        obs = self._get_obs()
        reward = -1
        done = False
        info = {}

        if self.time > 1000:
            done = True

        return obs, reward, done, info

    def render(self, mode="human"):
        if self.viewer is None:
            pygame.init()
            self.screen = pygame.display.set_mode((VIEWPORT_W, VIEWPORT_H))
            pygame.display.set_caption("Custom Racing RL")
            self.clock = pygame.time.Clock()

        # 背景草地颜色
        self.screen.fill((144, 238, 144))  # 浅绿色背景

        def transform(pos):
            return int(VIEWPORT_W/2 + pos[0]*SCALE), int(VIEWPORT_H/2 - pos[1]*SCALE)

        # 🚧 绘制赛道 tile（用 road_polys）
        for poly in self.road_polys:
            pygame.draw.polygon(self.screen, (100, 100, 100), [transform(v) for v in poly])

        # 🚗 绘制小车（红色圆点）
        pos = transform(self.car.position)
        pygame.draw.circle(self.screen, (255, 0, 0), pos, int(4))

        pygame.display.flip()
        self.clock.tick(FPS)

    def close(self):
        if self.viewer:
            pygame.quit()
            self.viewer = None

# 🚗 调试用入口
if __name__ == "__main__":
    env = CustomRacingEnv()
    obs = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, done, _ = env.step(action)
        env.render()
    env.close()

#用tilt矩形方块生成赛道，解决了赛道交叉。加了键盘控制，车画成三角形等功能。
import gym
from gym import spaces
import numpy as np
import Box2D
from Box2D.b2 import (world, polygonShape, staticBody, dynamicBody, vec2)
import pygame
import math
from shapely.geometry import Polygon

# =================== 全局设置 =====================
SCALE = 1.3 # 数值越小，画面比例尺越小。单位好像是像素，我忘了。
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
        # 记录窗口和时钟，初始时设为 None
        self.viewer = None
        self.clock = None

        # 状态空间（速度、方向角cos/sin、是否倒退）
        high = np.array([100.0, 1.0, 1.0, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        # 动作空间（转向Δ，油门Δ）
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32,
        )

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
            angle += turn * (np.pi / 8)  # 转向角度
        
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
            # 将该 tile 的所有 fixture 都设为 sensor，否则赛道是碰撞体
            for fixture in tile.fixtures:
                fixture.sensor = True
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
                if j in range(0, 5) or j in range(length - 5, length):
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
        car = self.world.CreateDynamicBody(
            position=(0, 0),
            angle=0.0,
            fixtures=Box2D.b2FixtureDef(
                shape=polygonShape(box=(2.5, 1.5)),
                density=1.0, # 刚体质量，密度，惯性
                friction=300,
            ),
        )
        #以下2行新加的
        car.linearDamping = 0.01
        car.angularDamping = 0.01
        return car

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
        self.car.ApplyForceToCenter(1000 * Δacc * forward, True)
        self.car.ApplyTorque(1000 * Δsteer, True)

        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)
        self.time += 1

        obs = self._get_obs()
        reward = -1
        done = False
        info = {}

        if self.time > 1e5: #赋值1000，大概40秒自动刷新赛道，1000单位是帧吧
            done = True

        if self.time % FPS == 0:
            print("车的位置:", self.car.position, "速度:", self.car.linearVelocity)
    
        return obs, reward, done, info

    def render(self, mode="human"):
        # 不再在这里初始化窗口，而直接使用已经存在的 self.viewer 和 self.clock
        # 确保调用 render() 前，窗口已在 main 中初始化并赋值给 env.viewer

        # 背景草地颜色
        self.viewer.fill((144, 238, 144))  # 浅绿色背景

        def transform(pos):
            return int(VIEWPORT_W/2 + pos[0]*SCALE), int(VIEWPORT_H/2 - pos[1]*SCALE)

        # 🚧 绘制赛道 tile（用 road_polys）
        for poly in self.road_polys:
            pygame.draw.polygon(self.viewer, (100, 100, 100), [transform(v) for v in poly])

        # 🚗 绘制小车（三角形）
        # 计算车体三角形的顶点（世界坐标）
        x, y = self.car.position
        angle = self.car.angle
        # L: 车的前部长度，W: 车的宽度的一半。你可以根据实际情况调整这些值（单位与赛道相同）。
        L = 20   # 前部长度
        W = 10   # 半宽

        # 车的前部点（朝向正前方）
        front = (x + L * math.cos(angle), y + L * math.sin(angle))
        # 车的后部中心点（稍微向后移半个L）
        rear_center = (x - (L * 0.5) * math.cos(angle), y - (L * 0.5) * math.sin(angle))
        # 后部左侧点
        rear_left = (rear_center[0] + W * math.cos(angle + math.pi/2),
                    rear_center[1] + W * math.sin(angle + math.pi/2))
        # 后部右侧点
        rear_right = (rear_center[0] + W * math.cos(angle - math.pi/2),
                    rear_center[1] + W * math.sin(angle - math.pi/2))

        # 将世界坐标转换为屏幕坐标
        pt_front = transform(front)
        pt_left = transform(rear_left)
        pt_right = transform(rear_right)

        # 绘制三角形（箭头形状），显示车的位置和朝向
        pygame.draw.polygon(self.viewer, (255, 0, 0), [pt_front, pt_left, pt_right])

        pygame.display.flip()
        self.clock.tick(FPS)

    def close(self):
        pygame.quit()
        # if self.viewer:
        #     pygame.quit()
        #     self.viewer = None

# 下面是键盘控制逻辑的主入口 #
if __name__ == "__main__":  
    # 初始化 Pygame 单一窗口（这里 env.render() 会创建窗口）
    pygame.init()
    screen = pygame.display.set_mode((VIEWPORT_W, VIEWPORT_H))
    pygame.display.set_caption("Custom Racing RL - 控制窗口")
    clock = pygame.time.Clock()
    
    # 创建环境
    env = CustomRacingEnv()
    # 将创建好的窗口和时钟赋值给环境
    env.viewer = screen
    env.clock = clock

    obs = env.reset()

    running = True
    action = [0.0, 0.0]  # [转向, 油门]
    
    while running:
        # 处理所有事件，确保窗口响应
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # 获取键盘状态（请确保窗口处于活动状态，即被点击过）
        keys = pygame.key.get_pressed()
        
        # 重置动作
        action[0] = 0.0  # 转向
        action[1] = 0.0  # 油门
        
        if keys[pygame.K_LEFT]:
            action[0] = 1.0
            print("左")
        elif keys[pygame.K_RIGHT]:
            action[0] = -1.0
            print("右")
        if keys[pygame.K_UP]:
            action[1] = 1.0
            print("上")
        if keys[pygame.K_DOWN]:
            action[1] = -1.0
            print("下")
        
        # 执行动作和更新环境
        obs, reward, done, _ = env.step(action)
        env.render()  # render() 内部使用了 pygame.display.flip()
        clock.tick(FPS)
        
        if done:
            obs = env.reset()
    
    env.close()
    pygame.quit()

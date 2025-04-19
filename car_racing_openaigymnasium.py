__credits__ = ["Andrea PIERRÉ"]  # 作者信息
import math  # 数学函数模块
from typing import Optional, Union  # 用于类型注解
import numpy as np  # 数组与数值计算库
import gymnasium as gym  # 导入Gymnasium环境包
from gymnasium import spaces  # 导入动作和观察空间的定义
from gymnasium.envs.box2d.car_dynamics import Car  # 导入汽车动力学模型
from gymnasium.error import DependencyNotInstalled, InvalidAction  # 导入错误处理类
from gymnasium.utils import EzPickle  # 导入便捷的序列化工具
import matplotlib.pyplot as plt
import imageio

# 尝试导入Box2D相关库，如果导入失败则抛出依赖未安装异常
try:
    import Box2D
    from Box2D.b2 import contactListener, fixtureDef, polygonShape
except ImportError as e:
    raise DependencyNotInstalled(
        'Box2D is not installed, you can install it by run pip install swig followed by pip install "gymnasium[box2d]"'
    ) from e

# 尝试导入pygame库，作为环境重置和步进中必要的显示组件
try:
    # As pygame is necessary for using the environment (reset and step) even without a render mode
    #   therefore, pygame is a necessary import for the environment.
    import pygame
    from pygame import gfxdraw
except ImportError as e:
    raise DependencyNotInstalled(
        'pygame is not installed, run `pip install "gymnasium[box2d]"`'
    ) from e


STATE_W = 128 # 状态图像的宽。less than Atari 160x192
STATE_H = 128 
VIDEO_W = 600 # 视频输出宽度
VIDEO_H = 400
WINDOW_W = 256#1000 # 渲染窗口宽度
WINDOW_H = 256#800

SCALE = 6.0  # Track scale。赛道缩放比例
TRACK_RAD = 888 / SCALE  # 圆形赛道基础半径。好像是先生成圆，再使其变形产生弯角。半径就是888/6米。
PLAYFIELD = 2000 / SCALE  # Game over boundary。总图范围。
FPS = 50  # Frames per second
ZOOM = 1.6  # Camera zoom。相机放大比例
ZOOM_FOLLOW = True  # Set to False for fixed view (no zoom)

TRACK_DETAIL_STEP = 20 / SCALE  # 赛道分段步长。值越大，如40，长弯多
TRACK_TURN_RATE = 0.3  # 赛道转弯率，控制赛道弯曲程度。越小倾向长弯，高速弯；越大倾向锐利弯，短弯。
TRACK_WIDTH = 32 / SCALE # 赛道宽度
BORDER = 8 / SCALE # 赛道边界（路肩）宽度
BORDER_MIN_COUNT = 4 # 最少的边界块数量，用于判断赛道边界的生成
GRASS_DIM = PLAYFIELD / 20.0 # 草地块尺寸
MAX_SHAPE_DIM = (
    max(GRASS_DIM, TRACK_WIDTH, TRACK_DETAIL_STEP) * math.sqrt(2) * ZOOM * SCALE
) # 最大形状尺寸，可能用于渲染或碰撞检测时的大小计算

# 定义处理汽车与赛道接触的检测器，继承自Box2D的contactListener
class FrictionDetector(contactListener):
    #处理车辆与赛道接触的物理检测
    def __init__(self, env, lap_complete_percent):
        contactListener.__init__(self)
        self.env = env
        self.lap_complete_percent = lap_complete_percent  # 完成一圈所需百分比

    # Box2D调用，当两个物体开始接触时触发
    def BeginContact(self, contact):
        self._contact(contact, True)

    # Box2D调用，当两个物体结束接触时触发
    def EndContact(self, contact):
        self._contact(contact, False)

    # 处理接触事件
    def _contact(self, contact, begin):
        tile = None  # 用于保存检测到的赛道块
        obj = None  # 用于保存与赛道块接触的其他物体（通常为车辆）
        # 获取两个接触物体的用户数据，这里用户数据用于存储对象的相关属性
        u1 = contact.fixtureA.body.userData
        u2 = contact.fixtureB.body.userData
        # 如果u1中有“road_friction”属性，说明该物体是赛道块
        if u1 and "road_friction" in u1.__dict__:
            tile = u1
            obj = u2
        # 同样检查u2是否为赛道块
        if u2 and "road_friction" in u2.__dict__:
            tile = u2
            obj = u1
        if not tile:
            return  # 如果都不是赛道块，则直接返回

        # inherit tile color from env。更新赛道块颜色
        tile.color[:] = self.env.road_color
        if not obj or "tiles" not in obj.__dict__:
            return
        if begin:  # 开始接触
            obj.tiles.add(tile)  # 将当前赛道块加入车辆已访问的集合中
            if not tile.road_visited: # 首次访问该赛道块
                tile.road_visited = True  # 标记为已访问
                self.env.reward += 1000.0 / len(self.env.track) # 计算reward。根据赛道总块数给予奖励，奖励值为1000除以赛道块总数
                self.env.tile_visited_count += 1 # 更新已访问的块计数

                # Lap is considered completed if enough % of the track was covered。检查是否完成一圈
                if (
                    tile.idx == 0  # 判断是否为起点块
                    and self.env.tile_visited_count / len(self.env.track)
                    > self.lap_complete_percent
                ):
                    self.env.new_lap = True # 标记完成新一圈
        else: # 结束接触，将赛道块从车辆已接触集合中移除
            obj.tiles.remove(tile)

# 赛车环境主类
class CarRacing(gym.Env, EzPickle):
    """
    ## Description
    The easiest control task to learn from pixels - a top-down
    racing environment. The generated track is random every episode.

    Some indicators are shown at the bottom of the window along with the
    state RGB buffer. From left to right: true speed, four ABS sensors,
    steering wheel position, and gyroscope.
    To play yourself (it's rather fast for humans), type:
    ```shell
    python gymnasium/envs/box2d/car_racing.py
    ```
    Remember: it's a powerful rear-wheel drive car - don't press the accelerator
    and turn at the same time.

    ## Action Space
    If continuous there are 3 actions :
    - 0: steering, -1 is full left, +1 is full right
    - 1: gas
    - 2: braking

    If discrete there are 5 actions:
    - 0: do nothing
    - 1: steer left
    - 2: steer right
    - 3: gas
    - 4: brake

    ## Observation Space

    A top-down 96x96 RGB image of the car and race track.

    ## Rewards
    The reward is -0.1 every frame and +1000/N for every track tile visited, where N is the total number of tiles
     visited in the track. For example, if you have finished in 732 frames, your reward is 1000 - 0.1*732 = 926.8 points.

    ## Starting State
    The car starts at rest in the center of the road.

    ## Episode Termination
    The episode finishes when all the tiles are visited. The car can also go outside the playfield -
     that is, far off the track, in which case it will receive -100 reward and die.

    ## Arguments

    ```python
    >>> import gymnasium as gym
    >>> env = gym.make("CarRacing-v3", render_mode="rgb_array", lap_complete_percent=0.95, domain_randomize=False, continuous=False)
    >>> env
    <TimeLimit<OrderEnforcing<PassiveEnvChecker<CarRacing<CarRacing-v3>>>>>

    ```

    * `lap_complete_percent=0.95` dictates the percentage of tiles that must be visited by
     the agent before a lap is considered complete.

    * `domain_randomize=False` enables the domain randomized variant of the environment.
     In this scenario, the background and track colours are different on every reset.

    * `continuous=True` specifies if the agent has continuous (true) or discrete (false) actions.
     See action space section for a description of each.

    ## Reset Arguments

    Passing the option `options["randomize"] = True` will change the current colour of the environment on demand.
    Correspondingly, passing the option `options["randomize"] = False` will not change the current colour of the environment.
    `domain_randomize` must be `True` on init for this argument to work.

    ```python
    >>> import gymnasium as gym
    >>> env = gym.make("CarRacing-v3", domain_randomize=True)

    # normal reset, this changes the colour scheme by default
    >>> obs, _ = env.reset()

    # reset with colour scheme change
    >>> randomize_obs, _ = env.reset(options={"randomize": True})

    # reset with no colour scheme change
    >>> non_random_obs, _ = env.reset(options={"randomize": False})

    ```

    ## Version History
    - v2: Change truncation to termination when finishing the lap (1.0.0)
    - v1: Change track completion logic and add domain randomization (0.24.0)
    - v0: Original version

    ## References
    - Chris Campbell (2014), http://www.iforce2d.net/b2dtut/top-down-car.

    ## Credits
    Created by Oleg Klimov
    """

    metadata = { # 定义环境元数据，包括支持的渲染模式和帧率
        "render_modes": [
            "human",
            "rgb_array",
            "state_pixels",
        ],
        "render_fps": FPS,
    }

    def __init__(
        self,
        render_mode: Optional[str] = None, # 渲染模式
        verbose: bool = False, # 是否打印调试信息
        lap_complete_percent: float = 0.95, # 完成一圈所需赛道百分比
        domain_randomize: bool = False, # 是否随机化颜色
        continuous: bool = True, # 渲是否使用连续动作空间
    ):
        EzPickle.__init__( # 使用EzPickle进行对象序列化（便于环境保存和重启）
            self,
            render_mode,
            verbose,
            lap_complete_percent,
            domain_randomize,
            continuous,
        )
        self.continuous = continuous
        self.domain_randomize = domain_randomize
        self.lap_complete_percent = lap_complete_percent
        self._init_colors()

        # 创建一个FrictionDetector实例，用于处理碰撞和接触逻辑
        self.contactListener_keepref = FrictionDetector(self, self.lap_complete_percent)
        # 创建Box2D物理世界，并指定重力为0（二维平面上无需重力）以及接触监听器
        self.world = Box2D.b2World((0, 0), contactListener=self.contactListener_keepref)
        self.screen: Optional[pygame.Surface] = None  # 渲染窗口Surface对象
        self.surf = None  # 用于离屏绘制的Surface对象
        self.clock = None  # pygame时钟对象，用于控制刷新率
        self.isopen = True  # 标识渲染窗口是否打开
        self.invisible_state_window = None  # 用于状态图像（可能不显示的窗口）
        self.invisible_video_window = None  # 用于视频输出（可能不显示的窗口）
        self.road = None  # 存储赛道各部分的集合
        self.car: Optional[Car] = None  # 存储车辆对象
        self.reward = 0.0  # 当前回合累计奖励
        self.prev_reward = 0.0  # 前一帧奖励（可能用于计算奖励变化）
        self.verbose = verbose  # 是否启用详细调试输出
        self.new_lap = False  # 是否开始新的一圈
        # 定义Box2D中用于创建赛道块的fixture定义，包含形状信息
        self.fd_tile = fixtureDef(
            shape=polygonShape(vertices=[(0, 0), (1, 0), (1, -1), (0, -1)])
        )

        # This will throw a warning in tests/envs/test_envs in utils/env_checker.py as the space is not symmetric
        #   or normalised however this is not possible here so ignore
        
        # if self.continuous:
        #     self.action_space = spaces.Box( # 连续动作: [转向,油门,刹车]
        #         np.array([-1, 0, 0]).astype(np.float32), # 动作下界：转向最左，油门和刹车为0
        #         np.array([+1, +1, +1]).astype(np.float32), # 动作上界：转向最右，油门和刹车最大值为1
        #     )  
        if self.continuous: # 把油门刹车合并成1个值
            self.action_space = spaces.Box(
                np.array([-1, -1]).astype(np.float32),  # 转向范围[-1,1]，油门/刹车范围[-1,1]
                np.array([+1, +1]).astype(np.float32),  # 负值刹车，正值油门
            )
        else:
            self.action_space = spaces.Discrete(5) # 离散动作: 无操作/左转/右转/油门/刹车

        # 初始化观察空间(96x96 RGB图像)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(STATE_H, STATE_W, 3), dtype=np.uint8
        )
        self.render_mode = render_mode # 保存渲染模式

        # 偏离赛道检测相关变量
        self.off_road_steps = 0      # 目前连续“离开赛道”的帧数
        self.off_road_threshold = 10 # 连续 N 帧未踩到赛道就算偏离
        self.off_road_penalty = -0.2 # 一旦判定偏离，就扣这么多分

    # 内部方法，用于清理和销毁环境中创建的物理对象
    def _destroy(self):
        if not self.road:
            return
        for t in self.road:
            self.world.DestroyBody(t) # 销毁每个赛道块
        self.road = []
        assert self.car is not None
        self.car.destroy() # 销毁车辆对象

    # 初始化颜色配置，根据是否启用随机化选择不同的颜色方案
    def _init_colors(self):
        if self.domain_randomize:
            #启用域随机化时，随机生成赛道、背景和草地颜色
            self.road_color = self.np_random.uniform(0, 210, size=3)

            self.bg_color = self.np_random.uniform(0, 210, size=3)

            self.grass_color = np.copy(self.bg_color)
            idx = self.np_random.integers(3) # 随机选取一个通道增加亮度
            self.grass_color[idx] += 20
        else:
            # default colours
            self.road_color = np.array([0, 0, 0])#赛道颜色，原来是[102, 102, 102]
            self.bg_color = np.array([255, 255, 255])#草地颜色，原来是[102, 204, 102]
            self.grass_color = np.array([255, 255, 255])#草地颜色，原来是[102, 230, 102]

    # 重新初始化颜色，当环境重置时可以通过选项指定是否随机化颜色
    def _reinit_colors(self, randomize):
        # 仅当启用域随机化时，才可以重新随机化颜色
        assert (
            self.domain_randomize
        ), "domain_randomize must be True to use this function."

        if randomize:
            # domain randomize the bg and grass colour
            self.road_color = self.np_random.uniform(0, 210, size=3)

            self.bg_color = self.np_random.uniform(0, 210, size=3)

            self.grass_color = np.copy(self.bg_color)
            idx = self.np_random.integers(3)
            self.grass_color[idx] += 20

    def _create_track(self):
        # 设置检查点的数量，用于构造赛道骨架
        CHECKPOINTS = 12  # 检查点数量

        # -----------------------
        # 生成检查点，每个检查点由一个角度alpha和一个半径rad确定
        checkpoints = []
        for c in range(CHECKPOINTS):
            # 为每个检查点添加随机噪声，扰动角度，增加赛道的随机性
            noise = self.np_random.uniform(0, 2 * math.pi * 1 / CHECKPOINTS)  # 随机噪声            
            # 均匀分布的角度，叠加噪声
            alpha = 2 * math.pi * c / CHECKPOINTS + noise
            # 随机选择半径，保证检查点不都在固定圆周上，值介于TRACK_RAD/3和TRACK_RAD之间。TRACK_RAD/0.1会导致弯角之间的直道很长。
            rad = self.np_random.uniform(TRACK_RAD / 3, TRACK_RAD)

            # 对于第一个检查点，固定参数（作为起点）
            if c == 0:
                alpha = 0
                rad = 1.5 * TRACK_RAD
            # 对于最后一个检查点，也做固定处理，并设置起始角度参数（用于后续判定赛道闭合）
            if c == CHECKPOINTS - 1:
                alpha = 2 * math.pi * c / CHECKPOINTS
                self.start_alpha = 2 * math.pi * (-0.5) / CHECKPOINTS
                rad = 1.5 * TRACK_RAD

            # 将检查点存入列表，检查点结构：(角度, x坐标, y坐标)
            checkpoints.append((alpha, rad * math.cos(alpha), rad * math.sin(alpha)))
        self.road = []  # 初始化道路列表，后续将存储所有赛道块

        # -----------------------
        # 从一个检查点到另一个检查点生成赛道轨迹
        # 初始点设定在(1.5 * TRACK_RAD, 0)，beta表示初始方向角（车头朝向）
        x, y, beta = 1.5 * TRACK_RAD, 0, 0
        dest_i = 0  # 当前目标检查点索引
        laps = 0    # 记录已经绕过的圈数（用于闭合赛道检测）
        track = []  # 用于保存轨迹数据，每个点包含(alpha, beta, x, y)
        no_freeze = 2500  # 计数器，防止死循环（如果生成赛道过程陷入无限循环则中断）
        visited_other_side = False  # 标记是否已经越过半圆，用于圈数统计

        # 主循环：不断生成轨迹点直到满足闭合条件或者达到最大计数
        while True:
            # 根据当前位置计算角度（从原点到当前位置的夹角）
            alpha = math.atan2(y, x)
            # 判断是否已经穿过正半轴，若是，则可能完成了一圈
            if visited_other_side and alpha > 0:
                laps += 1
                visited_other_side = False
            if alpha < 0:
                visited_other_side = True
                alpha += 2 * math.pi  # 将负角转换为正角

            # -----------------------
            # 根据检查点找到下一个目标检查点
            while True:  # 循环直到找到合适的目标检查点
                failed = True
                # 内部循环查找目标检查点，使得当前角度alpha小于目标检查点的角度
                while True:
                    dest_alpha, dest_x, dest_y = checkpoints[dest_i % len(checkpoints)]
                    if alpha <= dest_alpha:
                        failed = False
                        break
                    dest_i += 1
                    # 如果遍历了一圈检查点，退出循环
                    if dest_i % len(checkpoints) == 0:
                        break

                if not failed:
                    break

                # 如果未找到合适的检查点，则将alpha减去2π，再次尝试
                alpha -= 2 * math.pi
                continue

            # -----------------------
            # 计算当前位置的单位方向向量：r1表示前进方向，p1为垂直方向（向左或向右）
            r1x = math.cos(beta)
            r1y = math.sin(beta)
            # 计算垂直方向：左手法则得到垂直向量
            p1x = -r1y
            p1y = r1x
            # 计算从当前位置到目标检查点的向量差
            dest_dx = dest_x - x  # 指向目标的x分量
            dest_dy = dest_y - y  # 指向目标的y分量
            # 计算目标向量在前进方向上的投影（用于判断需要转向多少）
            proj = r1x * dest_dx + r1y * dest_dy

            # 调整beta角度，确保beta与alpha之间的差值在合理范围内（处理角度环绕问题）
            while beta - alpha > 1.5 * math.pi:
                beta -= 2 * math.pi
            while beta - alpha < -1.5 * math.pi:
                beta += 2 * math.pi

            prev_beta = beta  # 保存上一次的beta值
            proj *= SCALE  # 根据比例尺放大投影长度

            # 根据投影值调整车辆方向beta（目标：使得车头更靠近目标方向）
            if proj > 0.3: # 目标在右侧
                beta -= min(TRACK_TURN_RATE, abs(0.001 * proj)) # 向右
            if proj < -0.3: # 目标在左侧
                beta += min(TRACK_TURN_RATE, abs(0.001 * proj)) # 向左
            # 沿着垂直方向更新位置，形成赛道曲线
            x += p1x * TRACK_DETAIL_STEP
            y += p1y * TRACK_DETAIL_STEP

            # 将当前生成的轨迹点保存到track列表中，
            # 每个轨迹点包含：当前角度alpha，平滑后的方向beta（取前后平均），以及位置(x, y)
            track.append((alpha, prev_beta * 0.5 + beta * 0.5, x, y))

            # 当完成圈数超过4时退出循环，避免轨道过长
            if laps > 4:
                break
            no_freeze -= 1
            if no_freeze == 0:
                break  # 如果生成点数达到上限，防止无限循环

        # -----------------------
        # 找到闭合轨道的起止范围，确保生成的赛道形成闭环（忽略第一个闭环，只用第二个闭环）
        i1, i2 = -1, -1
        i = len(track)
        while True:
            i -= 1
            if i == 0:
                return False  # 若遍历完所有轨迹仍未闭合，则返回失败
            # 判断是否穿过起始角度（start_alpha），用于标记赛道闭合
            pass_through_start = (
                track[i][0] > self.start_alpha and track[i - 1][0] <= self.start_alpha
            )
            if pass_through_start and i2 == -1:
                i2 = i  # 记录第一次穿过起点的轨迹索引
            elif pass_through_start and i1 == -1:
                i1 = i  # 记录第二次穿过起点的轨迹索引，然后退出循环
                break
        if self.verbose:
            print("Track generation: %i..%i -> %i-tiles track" % (i1, i2, i2 - i1))
        assert i1 != -1
        assert i2 != -1

        # 取出闭环区间内的轨迹作为最终赛道
        track = track[i1 : i2 - 1]

        # -----------------------
        # 检查轨道起点与终点是否“粘合”得足够好，即两端相连平滑
        first_beta = track[0][1]
        first_perp_x = math.cos(first_beta)
        first_perp_y = math.sin(first_beta)
        well_glued_together = np.sqrt(
            np.square(first_perp_x * (track[0][2] - track[-1][2]))
            + np.square(first_perp_y * (track[0][3] - track[-1][3]))
        )
        if well_glued_together > TRACK_DETAIL_STEP:
            return False  # 如果两端距离过远，认为赛道生成失败

        # -----------------------
        # 处理赛道边界的装饰：在急转弯处添加红白边框
        border = [False] * len(track)
        for i in range(len(track)):
            good = True
            oneside = 0
            # 检查前面若干个轨迹点的方向变化是否足够大，判断是否为急转弯
            for neg in range(BORDER_MIN_COUNT):
                beta1 = track[i - neg - 0][1]
                beta2 = track[i - neg - 1][1]
                good &= abs(beta1 - beta2) > TRACK_TURN_RATE * 0.2
                oneside += np.sign(beta1 - beta2)
            good &= abs(oneside) == BORDER_MIN_COUNT
            border[i] = good
        # 将边界标记向前传播，确保边界块连续
        for i in range(len(track)):
            for neg in range(BORDER_MIN_COUNT):
                border[i - neg] |= border[i]

        # -----------------------
        # 根据生成的轨迹创建实际的赛道“瓷砖”（每块为一个静态刚体）
        for i in range(len(track)):
            alpha1, beta1, x1, y1 = track[i]
            # 上一个轨迹点（用于构建两个轨迹点之间的四边形）
            alpha2, beta2, x2, y2 = track[i - 1]
            # 计算当前轨迹点左右两侧的边界点，依据赛道宽度和轨迹方向
            road1_l = (
                x1 - TRACK_WIDTH * math.cos(beta1),
                y1 - TRACK_WIDTH * math.sin(beta1),
            )
            road1_r = (
                x1 + TRACK_WIDTH * math.cos(beta1),
                y1 + TRACK_WIDTH * math.sin(beta1),
            )
            # 上一个轨迹点对应的左右边界点
            road2_l = (
                x2 - TRACK_WIDTH * math.cos(beta2),
                y2 - TRACK_WIDTH * math.sin(beta2),
            )
            road2_r = (
                x2 + TRACK_WIDTH * math.cos(beta2),
                y2 + TRACK_WIDTH * math.sin(beta2),
            )
            # 四边形顶点顺序构成赛道“瓷砖”
            vertices = [road1_l, road1_r, road2_r, road2_l]
            # 设置fixture定义中的顶点为当前四边形
            self.fd_tile.shape.vertices = vertices
            # 在Box2D物理世界中创建一个静态刚体表示这块瓷砖
            t = self.world.CreateStaticBody(fixtures=self.fd_tile)
            t.userData = t  # 将自身设置为用户数据，便于后续碰撞检测中识别
            # 根据索引为瓷砖附加轻微色彩偏移，增加视觉变化
            c = 0.01 * (i % 3) * 255
            t.color = self.road_color + c
            t.road_visited = False  # 标记该块是否被访问过
            t.road_friction = 1.0  # 设置摩擦系数
            t.idx = i  # 记录该瓷砖的索引
            # 将该瓷砖的fixture设置为传感器，不影响物理碰撞
            t.fixtures[0].sensor = True
            # 存储瓷砖及其颜色，用于渲染
            self.road_poly.append(([road1_l, road1_r, road2_r, road2_l], t.color))
            self.road.append(t)
            # 如果当前轨迹点被标记为边界，则创建边缘装饰（红或白色边框）。不要路肩
            # if border[i]:
            #     side = np.sign(beta2 - beta1)
            #     b1_l = (
            #         x1 + side * TRACK_WIDTH * math.cos(beta1),
            #         y1 + side * TRACK_WIDTH * math.sin(beta1),
            #     )
            #     b1_r = (
            #         x1 + side * (TRACK_WIDTH + BORDER) * math.cos(beta1),
            #         y1 + side * (TRACK_WIDTH + BORDER) * math.sin(beta1),
            #     )
            #     b2_l = (
            #         x2 + side * TRACK_WIDTH * math.cos(beta2),
            #         y2 + side * TRACK_WIDTH * math.sin(beta2),
            #     )
            #     b2_r = (
            #         x2 + side * (TRACK_WIDTH + BORDER) * math.cos(beta2),
            #         y2 + side * (TRACK_WIDTH + BORDER) * math.sin(beta2),
            #     )
            #     self.road_poly.append(
            #         (
            #             [b1_l, b1_r, b2_r, b2_l],
            #             (255, 255, 255) if i % 2 == 0 else (255, 0, 0),
            #         )
            #     )
        
        # 保存最终生成的轨迹数据供后续使用（如车辆初始位置等）
        self.track = track
        # 打印轨迹点数量
        print(f"Generated track length: {len(track)} segments")  

        return True  # 返回True表示赛道生成成功

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):  # 重置环境状态
        # 调用父类reset方法，初始化随机种子等
        super().reset(seed=seed)
        self._destroy()  # 清理上一次生成的物理对象（赛道、车辆等）
        # 重新创建一个FrictionDetector实例用于碰撞检测
        self.world.contactListener_bug_workaround = FrictionDetector(
            self, self.lap_complete_percent
        )
        self.world.contactListener = self.world.contactListener_bug_workaround  # 设定物理世界的接触监听器

        # 重置环境的各项状态变量
        self.reward = 0.0
        self.prev_reward = 0.0
        self.tile_visited_count = 0
        self.t = 0.0  # 重置时间计数器
        self.new_lap = False  # 标记新一圈未开始
        self.road_poly = []  # 清空赛道多边形数据

        # 如果启用了颜色随机化，则根据options参数决定是否重新随机化颜色
        if self.domain_randomize:
            randomize = True
            if isinstance(options, dict):
                if "randomize" in options:
                    randomize = options["randomize"]

            self._reinit_colors(randomize)

        # 循环尝试生成赛道，直到成功为止
        while True:
            success = self._create_track()
            if success:
                break
            if self.verbose:
                print(
                    "retry to generate track (normal if there are not many"
                    "instances of this message)"
                )
        # 创建车辆对象，初始位置根据生成的轨迹第一段设置（取track[0]中保存的位置和角度）
        self.car = Car(self.world, *self.track[0][1:4])
        
        # 添加 tiles 属性用于偏离赛道检测
        self.car.tiles = set()

        # 如果渲染模式为"human"，则调用render方法
        if self.render_mode == "human":
            self.render()
        # reset()的返回值包含初始观察值和一个空字典
        return self.step(None)[0], {}

    def step(self, action: Union[np.ndarray, int]):
        # 确保车辆对象已经初始化
        assert self.car is not None
        if action is not None:
            # -----------------------
            # 根据动作类型处理：连续动作或离散动作
            if self.continuous:  # 连续动作模式
                action = action.astype(np.float64)
                self.car.steer(-action[0])  # 转向操作，注意方向取反
                # self.car.gas(action[1])      # 油门操作
                # self.car.brake(action[2])    # 刹车操作
                
                # 油门刹车合并成1个值
                if action[1] > 0:
                    self.car.gas(action[1])   # 正值：油门
                    self.car.brake(0)         # 刹车置零
                else:
                    self.car.gas(0)           # 油门置零
                    self.car.brake(-action[1]) # 负值：刹车（取绝对值）
            else:
                # 检查传入的动作是否合法
                if not self.action_space.contains(action):
                    raise InvalidAction(
                        f"you passed the invalid action `{action}`. "
                        f"The supported action_space is `{self.action_space}`"
                    )
                # 离散动作：分别处理向左转、向右转、加速和制动
                self.car.steer(-0.6 * (action == 1) + 0.6 * (action == 2))
                self.car.gas(0.2 * (action == 3))
                self.car.brake(0.8 * (action == 4))

        # 更新车辆状态，调用车辆自身的step方法，传入时间步长（1/FPS秒）
        self.car.step(1.0 / FPS)
        # 让Box2D物理世界进行一步仿真
        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)
        # 更新时间计数器
        self.t += 1.0 / FPS

        # 获取当前状态图像（通过_render方法生成）
        self.state = self._render("state_pixels")

        step_reward = 0
        terminated = False  # 回合是否结束
        truncated = False   # 是否截断（如时间限制）
        info = {}
        if action is not None:  # First step without action, called from reset()
            # 计算reward
            self.reward -= 0.1  # 每步小惩罚
            # We actually don't want to count fuel spent, we want car to be faster.
            # self.reward -=  10 * self.car.fuel_spent / ENGINE_POWER
            self.car.fuel_spent = 0.0
            # 本步奖励为当前累计奖励与上一帧奖励的差值
            step_reward = self.reward - self.prev_reward
            self.prev_reward = self.reward
            # 如果所有赛道块均已访问或完成一圈，则回合结束，且设置info标记
            if self.tile_visited_count == len(self.track) or self.new_lap:
                terminated = True
                info["lap_finished"] = True
            # 检查车辆是否超出边界（超出PLAYFIELD），若超出则回合失败，给予较大惩罚
            x, y = self.car.hull.position
            if abs(x) > PLAYFIELD or abs(y) > PLAYFIELD:
                terminated = True
                info["lap_finished"] = False
                step_reward = -100

        # ===== 新增偏离赛道检测 =====
        if len(self.car.tiles) == 0:
            # 这一帧没有与任何 road tile 接触
            self.off_road_steps += 1
        else:
            # 只要重新“踩”到路面，就重置计数
            self.off_road_steps = 0

        # —— 偏离赛检测逻辑 —— 
        speed = np.linalg.norm(self.car.hull.linearVelocity)
        info["speed"] = speed #speed本身时局部变量，这样能实现输出全局变量
        if speed > 0.01: # 静止时不判断偏离赛道
            if len(self.car.tiles) == 0:
                self.off_road_steps += 1
            else:
                self.off_road_steps = 0
        else:
            self.off_road_steps = 0

        if self.off_road_steps >= self.off_road_threshold:
            step_reward += self.off_road_penalty
            self.off_road_steps = 0

        # 如果渲染模式为human，则调用render进行界面更新
        if self.render_mode == "human":
            self.render()
        # 返回当前状态、奖励、终止标志、截断标志和附加信息
        return self.state, step_reward, terminated, truncated, info

    def render(self):
        # 如果render_mode未设置，则发出警告提示用户需要在初始化时指定
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return
        else:
            # 根据render_mode调用内部的_render方法进行实际渲染（可能为图像数组或直接显示）
            return self._render(self.render_mode)
        

    # 该函数根据指定的渲染模式，将当前环境状态绘制到屏幕或转换为图像数组返回
    def _render(self, mode: str): 
        # 确保传入的mode在metadata["render_modes"]中（human、rgb_array、state_pixels等）
        assert mode in self.metadata["render_modes"]

        # 初始化pygame字体模块
        pygame.font.init()
        # 如果还没有创建窗口，并且渲染模式为human，则初始化窗口
        if self.screen is None and mode == "human":
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
        # 初始化时钟用于控制帧率
        if self.clock is None:
            self.clock = pygame.time.Clock()

        # 如果没有t属性，则说明还没有调用reset()，直接返回
        if "t" not in self.__dict__:
            return  # reset() not called yet

        # 创建一个新的Surface作为绘图区域，尺寸为窗口尺寸
        self.surf = pygame.Surface((WINDOW_W, WINDOW_H))

        # 确保车辆对象存在，用于获取车辆状态信息
        assert self.car is not None
        # -----------------------------
        # 计算变换参数（缩放、平移、旋转），以便从物理坐标转换到屏幕像素坐标
        angle = -self.car.hull.angle  # 车辆角度取反（因为渲染时坐标系转换）
        # 渐变式缩放：刚开始时较小，随着时间t增加，逐渐达到预设的ZOOM比例
        # zoom = 0.1 * SCALE * max(1 - self.t, 0) + ZOOM * SCALE * min(self.t, 1)
        zoom = ZOOM * SCALE # 不要渐变缩放：
        # 平移量：将车辆位置平移到屏幕中心附近（根据缩放比例计算）
        scroll_x = -(self.car.hull.position[0]) * zoom
        scroll_y = -(self.car.hull.position[1]) * zoom
        # 利用pygame的Vector2进行旋转变换，考虑车辆旋转角度
        trans = pygame.math.Vector2((scroll_x, scroll_y)).rotate_rad(angle)
        # 将变换后的结果调整到屏幕坐标（屏幕中心、上偏一点）
        # trans = (WINDOW_W / 2 + trans[0], WINDOW_H / 4 + trans[1])
        trans = (WINDOW_W / 2 + trans[0], WINDOW_H / 4 + trans[1]-90) #平移渲染窗口的视角

        # 绘制赛道（包括背景、草地和道路），传入当前缩放、平移和旋转角度
        self._render_road(zoom, trans, angle)
        # 绘制车辆，将车辆绘制到self.surf上，最后一个参数决定是否显示车辆轮廓
        self.car.draw(
            self.surf,
            zoom,
            trans,
            angle,
            mode not in ["state_pixels_list", "state_pixels"],
        )

        # 将Surface垂直翻转（因为pygame的坐标系与常规图像坐标系方向相反）
        self.surf = pygame.transform.flip(self.surf, False, True)

        # 在屏幕上绘制状态指标（例如速度、ABS传感器、转向角、陀螺仪等）
        # self._render_indicators(WINDOW_W, WINDOW_H) #渲染窗口不显示底部信息，注释掉

        # 绘制奖励数值（最左边显示累计奖励）
        font = pygame.font.Font(pygame.font.get_default_font(), 42)
        text = font.render("%04i" % self.reward, True, (255, 255, 255), (0, 0, 0))
        text_rect = text.get_rect()
        text_rect.center = (60, WINDOW_H - WINDOW_H * 2.5 / 40.0)
        # self.surf.blit(text, text_rect) #渲染窗口不显示奖励值，注释掉

        # -----------------------------
        # 根据不同的mode做不同处理：
        if mode == "human":
            # 人类模式：处理事件、更新时钟、将绘制结果显示到窗口
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            assert self.screen is not None
            self.screen.fill(0)
            self.screen.blit(self.surf, (0, 0))
            pygame.display.flip()
        elif mode == "rgb_array":
            # 返回视频输出尺寸的图像数组
            return self._create_image_array(self.surf, (VIDEO_W, VIDEO_H))
        elif mode == "state_pixels":
            # 返回状态图像尺寸的图像数组（用于RL算法）
            return self._create_image_array(self.surf, (STATE_W, STATE_H))
        else:
            # 其他模式则返回是否仍在打开状态
            return self.isopen
        
        # 截取中央256*256区域
        # 获取当前512×512的渲染表面
        surf = pygame.display.get_surface()        
        # 计算中央256×256区域的坐标（从(128,128)开始截取）
        center_x, center_y = WINDOW_W//2 - 128, WINDOW_H//2 - 128
        cropped_surf = surf.subsurface((center_x, center_y, 256, 256))
        # 返回截取后的图像（或直接显示）
        # if mode == 'human':
        #     pygame.init()  # 确保Pygame已初始化
        #     screen = pygame.display.set_mode((256, 256))  # 创建新窗口
        #     screen.blit(cropped_surf, (0, 0))  # 绘制裁剪后的图像
        #     pygame.display.flip()  # 更新显示
        # if mode == 'rgb_array': #此时返回numpy数组
        #     return pygame.surfarray.array3d(cropped_surf)
        return cropped_surf

    # 该函数负责绘制背景、草地区域和赛道瓷砖。
    def _render_road(self, zoom, translation, angle): 
        bounds = PLAYFIELD
        # 构造一个大矩形，作为背景绘制的区域（四个角点）
        field = [
            (bounds, bounds),
            (bounds, -bounds),
            (-bounds, -bounds),
            (-bounds, bounds),
        ]

        # 绘制背景：使用背景颜色填充整个大矩形区域
        self._draw_colored_polygon(
            self.surf, field, self.bg_color, zoom, translation, angle, clip=False
        )

        # -----------------------------
        # 绘制草地区域（利用多个小方块拼接成草地）
        grass = []
        for x in range(-20, 20, 2):
            for y in range(-20, 20, 2):
                # 每个草地区块为一个小四边形
                grass.append(
                    [
                        (GRASS_DIM * x + GRASS_DIM, GRASS_DIM * y + 0),
                        (GRASS_DIM * x + 0, GRASS_DIM * y + 0),
                        (GRASS_DIM * x + 0, GRASS_DIM * y + GRASS_DIM),
                        (GRASS_DIM * x + GRASS_DIM, GRASS_DIM * y + GRASS_DIM),
                    ]
                )
        for poly in grass:
            self._draw_colored_polygon(
                self.surf, poly, self.grass_color, zoom, translation, angle
            )

        # -----------------------------
        # 绘制赛道：遍历之前创建的road_poly列表（存储每块赛道瓷砖的顶点和颜色）
        for poly, color in self.road_poly:
            # 将每个顶点的坐标转化为像素坐标
            poly = [(p[0], p[1]) for p in poly]
            color = [int(c) for c in color]
            self._draw_colored_polygon(self.surf, poly, color, zoom, translation, angle)

        # 绘制顺逆方向标志，箭头/花纹
        for i, (poly, color) in enumerate(self.road_poly):
            if i % 3 == 0:  # 每3个赛道块显示一个箭头
                poly = [(p[0], p[1]) for p in poly]
                color = [int(c) for c in color]
                self._draw_colored_polygon(self.surf, poly, color, zoom, translation, angle)

                # 计算赛道块中心点
                center_x = sum(p[0] for p in poly) / len(poly)
                center_y = sum(p[1] for p in poly) / len(poly)

                # 获取当前赛道块的切向方向（beta角度）
                # 注意：这里假设self.track存储了每个赛道段的beta角度
                if hasattr(self, 'track') and i < len(self.track):
                    _, beta, _, _ = self.track[i]
                else:
                    beta = 0  # 默认方向

                # 绘制箭头（指向切向方向）
                arrow_size = 1  # 箭头大小
                # 箭头的初始方向是向上（0度），需要根据beta旋转
                arrow_points = [
                    (0, arrow_size),  # 顶点
                    (arrow_size, -arrow_size),  # 左下
                    (-arrow_size, -arrow_size),  # 右下
                ]
                # 旋转箭头
                rotated_arrow = []
                for (x, y) in arrow_points:
                    # 旋转beta角度
                    rx = x * math.cos(beta) - y * math.sin(beta)
                    ry = x * math.sin(beta) + y * math.cos(beta)
                    # 平移到中心点
                    rotated_arrow.append((center_x + rx, center_y + ry))
                
                arrow_color = (127, 127, 127)  # 灰色箭头
                self._draw_colored_polygon(
                    self.surf, rotated_arrow, arrow_color, zoom, translation, angle
                )

    # 该函数绘制屏幕底部信息,从左到右分别是：数字，累积奖励；白色，车速；4个蓝色块，4个轮子的角速度；绿色，方向盘的转向角度；红色，车身的横摆角变化率
    def _render_indicators(self, W, H):
        s = W / 40.0  # 指示器宽度单位
        h = H / 40.0  # 指示器高度单位
        color = (0, 0, 0)
        # 绘制底部背景条（一个长矩形）
        polygon = [(W, H), (W, H - 5 * h), (0, H - 5 * h), (0, H)]
        pygame.draw.polygon(self.surf, color=color, points=polygon)

        # 内部方法：根据位置和数值生成一个竖直的指示器形状（多边形）
        def vertical_ind(place, val):
            return [
                (place * s, H - (h + h * val)),
                ((place + 1) * s, H - (h + h * val)),
                ((place + 1) * s, H - h),
                ((place + 0) * s, H - h),
            ]

        # 内部方法：生成水平方向的指示器形状
        def horiz_ind(place, val):
            return [
                ((place + 0) * s, H - 4 * h),
                ((place + val) * s, H - 4 * h),
                ((place + val) * s, H - 2 * h),
                ((place + 0) * s, H - 2 * h),
            ]

        # 计算车速（车辆线速度的模长）
        assert self.car is not None
        true_speed = np.sqrt(
            np.square(self.car.hull.linearVelocity[0])
            + np.square(self.car.hull.linearVelocity[1])
        )

        # 内部方法：若指标值大于微小阈值，则绘制多边形表示指标
        def render_if_min(value, points, color):
            if abs(value) > 1e-4:
                pygame.draw.polygon(self.surf, points=points, color=color)

        # 绘制车速指标（使用竖直指示器）
        render_if_min(true_speed, vertical_ind(5, 0.02 * true_speed), (255, 255, 255))
        # 绘制ABS传感器状态，分别为前后轮，采用不同的位置和颜色
        render_if_min(
            self.car.wheels[0].omega,
            vertical_ind(7, 0.01 * self.car.wheels[0].omega),
            (0, 0, 255),
        )
        render_if_min(
            self.car.wheels[1].omega,
            vertical_ind(8, 0.01 * self.car.wheels[1].omega),
            (0, 0, 255),
        )
        render_if_min(
            self.car.wheels[2].omega,
            vertical_ind(9, 0.01 * self.car.wheels[2].omega),
            (51, 0, 255),
        )
        render_if_min(
            self.car.wheels[3].omega,
            vertical_ind(10, 0.01 * self.car.wheels[3].omega),
            (51, 0, 255),
        )

        # 绘制转向角和车体角速度指标，分别使用水平指示器显示
        render_if_min(
            self.car.wheels[0].joint.angle,
            horiz_ind(20, -10.0 * self.car.wheels[0].joint.angle),
            (0, 255, 0),
        )
        render_if_min(
            self.car.hull.angularVelocity,
            horiz_ind(30, -0.8 * self.car.hull.angularVelocity),
            (255, 0, 0),
        )

    # 该函数实现了多边形绘制，并进行坐标变换、旋转及剪裁检测
    def _draw_colored_polygon(
        self, surface, poly, color, zoom, translation, angle, clip=True
    ):
        # 将传入的多边形每个顶点先旋转指定角度
        poly = [pygame.math.Vector2(c).rotate_rad(angle) for c in poly]
        # 将旋转后的顶点进行缩放和平移，转化为屏幕像素坐标
        poly = [
            (c[0] * zoom + translation[0], c[1] * zoom + translation[1]) for c in poly
        ]
        # This checks if the polygon is out of bounds of the screen, and we skip drawing if so.
        # Instead of calculating exactly if the polygon and screen overlap,
        # we simply check if the polygon is in a larger bounding box whose dimension
        # is greater than the screen by MAX_SHAPE_DIM, which is the maximum
        # diagonal length of an environment object
        
        # clip参数控制是否需要检测多边形是否在屏幕范围内
        if not clip or any(
            (-MAX_SHAPE_DIM <= coord[0] <= WINDOW_W + MAX_SHAPE_DIM)
            and (-MAX_SHAPE_DIM <= coord[1] <= WINDOW_H + MAX_SHAPE_DIM)
            for coord in poly
        ):
            gfxdraw.aapolygon(self.surf, poly, color) # 使用抗锯齿方式绘制多边形边缘
            gfxdraw.filled_polygon(self.surf, poly, color) # 绘制填充的多边形

    # 该函数将Surface转换为图像数组（NumPy数组），并调整数组的维度顺序
    def _create_image_array(self, screen, size):
        # 平滑缩放到指定尺寸（VIDEO_W, VIDEO_H或STATE_W, STATE_H）
        scaled_screen = pygame.transform.smoothscale(screen, size)
        # 利用pygame.surfarray获取像素数据，并转换为NumPy数组，注意需要转置轴顺序（行列互换）
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(scaled_screen)), axes=(1, 0, 2)
        )

    def close(self): # 该函数用于关闭渲染窗口并释放pygame资源
        if self.screen is not None:
            pygame.display.quit()
            self.isopen = False
            pygame.quit()


if __name__ == "__main__":
    # a = np.array([0.0, 0.0, 0.0])  # 用于存储控制动作
    a = np.array([0.0, 0.0])  # 油门刹车合并成1个值

    def register_input():
        # 全局变量quit和restart用于控制退出或重启
        global quit, restart
        # 遍历处理pygame事件
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN: # 按键按下事件
                if event.key == pygame.K_LEFT:
                    a[0] = -1.0
                if event.key == pygame.K_RIGHT:
                    a[0] = +1.0
                if event.key == pygame.K_UP:
                    a[1] = +1.0
                if event.key == pygame.K_DOWN:
                    # a[2] = +0.8  # 设置较低的刹车值，使车轮减转速
                    a[1] = -0.2  # 油门刹车合并成1个值
                if event.key == pygame.K_RETURN:
                    restart = True
                if event.key == pygame.K_ESCAPE:
                    quit = True

            if event.type == pygame.KEYUP: # 按键释放事件
                # 松开按键后将对应动作归零
                if event.key == pygame.K_LEFT:
                    a[0] = 0
                if event.key == pygame.K_RIGHT:
                    a[0] = 0
                if event.key == pygame.K_UP:
                    a[1] = 0
                if event.key == pygame.K_DOWN:
                    # a[2] = 0
                    a[1] = 0 # 油门刹车合并成1个值

            if event.type == pygame.QUIT:
                quit = True

    # 创建CarRacing环境，并设置渲染模式为 human/rgb_array/state_pixels
    env = CarRacing(render_mode="human")

    quit = False
    while not quit:
        env.reset()
        total_reward = 0.0
        steps = 0
        restart = False
        while True:
            # 只有human模式下才去处理键盘输入,state_pixels模式不需要键盘输入。没有这行会报错。
            if env.render_mode == "human":
               register_input()
            # 执行动作，获取下一个状态、奖励、是否终止等信息
            img, reward, terminated, truncated, info = env.step(a)
            # # 用Matplotlib显示96*96observation
            # plt.imshow(img)
            # plt.axis('off')
            # plt.show()
            # # 用imageio将当前帧保存成文件
            # imageio.imwrite('/home/user/Desktop/frame.png', img)
            # 查看 observation img 的类型和维度
            print(f"img 类型: {type(img)}, dtype: {img.dtype}, shape: {img.shape}")
            # （可选）查看维度数量
            print(f"img 维度数 ndim: {img.ndim}")
            total_reward += reward
            if steps % 50 == 0 or terminated or truncated: # 每50帧，即1秒
                print("\naction " + str([f"{x:+0.2f}" for x in a]))
                print(f"step {steps} total_reward {total_reward:+0.2f}")
                # print(f"step {steps} env.reward {env.reward:+0.2f}")                
            steps += 1
            # 如果回合结束、重启或者退出，则跳出当前循环
            if terminated or truncated or restart or quit:
                break
    env.close()  # 关闭环境并退出pygame

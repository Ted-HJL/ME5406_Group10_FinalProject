#基于test_rl.py改的
import numpy as np
from car_racing_openaigymnasium import CarRacing
global_terminated   = None
steps        = 0
reward       = 0.0       
total_reward = 0.0       
speed        = None
img          = None  

action = np.array([0.0, 0.0], dtype=np.float32)

def simulator():
    global global_terminated, steps, reward, total_reward, speed, img, action
    # 1) 初始化环境，render_mode="human/rgb_array/state_pixels" 会打开一个窗口显示真实画面
    env = CarRacing(render_mode="human", continuous=True)

    # 2) 重置，拿到初始 observation 
    obs, _ = env.reset()
    total_reward = 0.0
    steps        = 0
    done = False

    # 3) 主循环：不断输出 action 给环境，action 形状和范围要匹配 env.action_space
    while not done and steps < 1e3:
        img, reward, global_terminated, truncated, info = env.step(action)
        
        total_reward += reward
        steps += 1
        done = bool(global_terminated or truncated)

        speed = info.get("speed", 0.0)# grab the speed out of info。如果 info 中找不到，就返回默认值 0.0
        if steps % 50 == 0 or global_terminated or truncated: # 每50帧，即1秒
            print(f"global_terminated {global_terminated} steps {steps:+0.2f} reward {reward:+0.2f} speed {speed:+0.2f}") 

    print(f"Episode 完成，总奖励 {total_reward:+.2f}")
    env.close()

if __name__ == "__main__":
    simulator()


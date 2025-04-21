import numpy as np
from car_racing_openaigymnasium import CarRacing

def main():
    # 1) 初始化环境，render_mode="human/rgb_array/state_pixels" 会打开一个窗口显示真实画面
    env = CarRacing(render_mode="human", continuous=True)

    # 2) 重置，拿到初始 observation (128×128×3 的 NumPy 数组)
    obs, _ = env.reset()

    done = False
    total_reward = 0.0
    step = 0

    # 3) 主循环：不断输出 action 给环境，action 形状和范围要匹配 env.action_space
    while not done and step < 1e3:
        # 动作示例：左转打死，地板油 
        action = np.array([-1.0, 1.0], dtype=np.float32)
        # (可选)随机测试：
        # action = np.random.uniform(low=-1.0, high=1.0, size=(2,)).astype(np.float32)

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step += 1
        done = terminated or truncated

        print(f"Step {step:03d} — action={action}, reward={reward:+.2f}")

    print(f"Episode 完成，总奖励 {total_reward:+.2f}")
    env.close()

if __name__ == "__main__":
    main()

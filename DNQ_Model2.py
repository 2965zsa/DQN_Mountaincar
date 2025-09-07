import os, gymnasium as gym, numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor

# 1.1 创建日志目录，用于保存 monitor.csv 与后续 TensorBoard 数据
log_dir = "./log/model3"
os.makedirs(log_dir, exist_ok=True)

# 1.2 创建环境并包装 Monitor，自动记录每条 episode 的 return 与长度
env = gym.make("MountainCar-v0", render_mode="rgb_array")  # rgb_array 方便后续离线渲染
env = Monitor(env, log_dir)

# ==========================================
# 2. 训练阶段
#    - 使用 DQN + MlpPolicy（纯全连接网络）
#    - 总步数 750k，约几分钟～十几分钟（CPU 下）
# ==========================================
model = DQN("MlpPolicy", env, verbose=1, learning_rate=5e-4,device="cuda")
model.learn(total_timesteps=int(7.5e5), progress_bar=True)
model.save("DQN_MountainCar2")

# ==========================================
# 3. 训练后分析
#    - 读取 monitor.csv，计算成功率、平均回合长度等
# ==========================================
log_file = os.path.join(log_dir, "monitor.csv")
rew, lng = [], []
with open(log_file) as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith('#') or line.startswith('r,'):
            continue          # 去掉注释行和表头
        parts = line.split(',')
        rew.append(float(parts[0]))   # reward
        lng.append(int(parts[1]))     # length
rew, lng = np.array(rew), np.array(lng)
success = lng < 200

print("\n=====  Training Finished  =====")
print(f"Episodes: {len(rew)}")
print(f"Average reward: {rew.mean():.2f}")
print(f"Success episodes: {success.sum()}")
print(f"Success rate: {success.mean()*100:.1f}%")
if success.any():
    print(f"Mean steps in success: {lng[success].mean():.1f}")
print("===============================")

# ==========================================
# 4. 可视化
#    - 对 reward 做 100 滑动平均，观察收敛趋势
# ==========================================
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 4))
plt.plot(np.convolve(rew, np.ones(100)/100, mode='valid'))
plt.xlabel("Episode")
plt.ylabel("Reward (100-ep moving average)")
plt.grid(True)
plt.title(f"MountainCar-v0 DQN — success={success.mean():.1%}  mean={rew.mean():.1f}")
plt.tight_layout()
plt.show()
import gym
import d4rl
import numpy as np
import matplotlib.pyplot as plt

# 创建环境
env = gym.make('antmaze-large-diverse-v0')

# 重置环境
env.reset()

# 获取当前环境的模拟器对象
sim = env.unwrapped.sim

# 初始化渲染器（使用无头模式）
viewer = env.unwrapped._get_viewer(mode='rgb_array')

focus = {
    "antmaze-umaze-v0": [3.9,3.9,1],
    "antmaze-medium-diverse-v0": [10,10,1],
    "antmaze-large-diverse-v0": [18,12,1]
}

distance = {
    "antmaze-umaze-v0": 28.0,
    "antmaze-medium-diverse-v0": 45.0,
    "antmaze-large-diverse-v0": 65.0
}

# 设置相机参数
viewer.cam.distance = 65.0  # 增加相机距离以覆盖整个迷宫
viewer.cam.elevation = -90  # 俯视角度
viewer.cam.azimuth = 90     # 水平旋转角度

viewer.cam.lookat[:] = [18,12,1]  # 设置相机焦点

# 使用无头模式获取 RGB 图像数据
frame = env.render(mode='rgb_array')

# 显示图像（可选）
plt.imshow(frame)
plt.axis('off')  # 关闭坐标轴
plt.show()

# 保存图像为文件
plt.imsave('./imgs/antmaze_large.png', frame, dpi=300)

# 关闭环境
env.close()
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scienceplots  # noqa: F401

# 1. 提取 Claude Opus 4.6 的真实数据 (源自 P1 和 P5)
species = ['Humpback', 'Bird', 'WhaleShark', 'Lion']

# Clean (P1) [1]
clean_acc = [89.13, 54.21, 60.27, 50.37]
# Grayscale (P5) [2]
gray_acc = [89.17, 45.74, 59.34, 48.62]
# Occlusion (P5) [2]
occ_acc = [61.43, 34.18, 52.81, 31.84]
# Resolution (P5) [2]
res_acc = [81.26, 30.12, 32.18, 33.17]

x = np.arange(len(species))
width = 0.2  # 柱子宽度

# 2. 设置学术风格主题 (与 3a, 3b, 3c 统一)
plt.style.use(['science', 'no-latex'])
fig, ax = plt.subplots(figsize=(2.5, 2.5))

# 3. 绘制分组柱状图
# 使用更高级、柔和的现代学术配色 (Nature/Science 偏好极简风)
c_clean = '#00718B'  # 深青色/Deep Teal
c_gray  = '#8E9CA3'  # 质感灰/Cool Gray
c_occ   = '#D26466'  # 玫瑰红/Rose Red
c_res   = '#ECA638'  # 暖姜黄/Warm Gold

# 使用纯色搭配白色描边，去除斜线和点点，画面更干净通透
bars1 = ax.bar(x - 1.5*width, clean_acc, width, label='Clean', color=c_clean, edgecolor='white', linewidth=0.6)
bars2 = ax.bar(x - 0.5*width, gray_acc, width, label='Grayscale', color=c_gray, edgecolor='white', linewidth=0.6)
bars3 = ax.bar(x + 0.5*width, occ_acc, width, label='Occlusion', color=c_occ, edgecolor='white', linewidth=0.6)
bars4 = ax.bar(x + 1.5*width, res_acc, width, label='Low-Res', color=c_res, edgecolor='white', linewidth=0.6)

# 4. 图表美化
ax.set_ylabel('Accuracy (%)', fontsize=9)
ax.set_xlabel('Species', fontsize=9)
ax.set_title('(d) Impact of Corruptions', fontsize=10, pad=8)
ax.set_xticks(x)
# 缩小X轴标签字体并水平居中放置，防止在 2.5 英寸画布下相互重叠
ax.set_xticklabels(species, fontsize=6.5)
ax.set_ylim(20, 100)
ax.tick_params(axis='y', which='major', labelsize=8)

# 图例与网格对齐
ax.legend(loc='upper right', fontsize=6.5, frameon=False, ncol=2, 
          handlelength=1.0, handletextpad=0.3, columnspacing=0.6)
ax.grid(True, axis='y', linestyle='--', alpha=0.4, color='#B0B0B0')
ax.set_axisbelow(True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(top=False, right=False)

# 5. 展示与保存
plt.tight_layout()
plt.savefig('figure_3d.png', dpi=300, bbox_inches='tight')
# plt.show()
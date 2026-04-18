import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401

# 1. 准备数据
# 精简模型名称，防止在 2.5 英寸紧凑画布下相互重叠
models = ['Claude', 'Qwen-122B', 'Qwen-35B']

# w/o Distractor 数据 (纯净环境，性能上升)
wo_n2 = [60.95, 58.78, 51.46]
wo_n3 = [63.24, 60.91, 53.04]
wo_n4 = [64.16, 61.64, 53.44]

# w/ Distractor 数据 (有干扰图，性能暴跌且停滞)
w_n2 = [48.37, 38.61, 29.74]
w_n3 = [49.12, 39.42, 30.16]
w_n4 = [49.84, 38.97, 29.48]

x = np.arange(len(models))
width = 0.12  # 画幅缩小后，柱子需要进一步收窄以容纳6根

# 2. 设置学术风格主题和紧凑画布大小 (与 3系列, 4a-4c 统一对齐的 2.5x2.5)
plt.style.use(['science', 'no-latex'])
fig, ax = plt.subplots(figsize=(2.5, 2.5))

# 3. 配置颜色 (同色系代表相同N，深浅代表有无Distractor)
# 使用与 3d, 4b 统一的高级学术配色，纯色搭配白色描边，去除刺眼的黑线与斜线
colors = {
    'N2_wo': '#08519C', 'N2_w': '#6BAED6',  # 蓝色系 (Qwen主色调)
    'N3_wo': '#ECA638', 'N3_w': '#F4CD89',  # 暖姜黄系 (对齐3d/4b)
    'N4_wo': '#D26466', 'N4_w': '#E8A2A3'   # 玫瑰红系 (对齐3d/4b)
}

# 4. 绘制6根柱子
ax.bar(x - 2.5*width, wo_n2, width, label='N=2 (w/o)', color=colors['N2_wo'], edgecolor='white', linewidth=0.6, zorder=3)
ax.bar(x - 1.5*width, w_n2, width, label='N=2 (w/)', color=colors['N2_w'], edgecolor='white', linewidth=0.6, zorder=3)

ax.bar(x - 0.5*width, wo_n3, width, label='N=3 (w/o)', color=colors['N3_wo'], edgecolor='white', linewidth=0.6, zorder=3)
ax.bar(x + 0.5*width, w_n3, width, label='N=3 (w/)', color=colors['N3_w'], edgecolor='white', linewidth=0.6, zorder=3)

ax.bar(x + 1.5*width, wo_n4, width, label='N=4 (w/o)', color=colors['N4_wo'], edgecolor='white', linewidth=0.6, zorder=3)
ax.bar(x + 2.5*width, w_n4, width, label='N=4 (w/)', color=colors['N4_w'], edgecolor='white', linewidth=0.6, zorder=3)

# 5. 图表细节美化
ax.set_title('(d) Robustness to Distraction', fontsize=10, pad=8)
ax.set_xlabel('Evaluated MLLMs', fontsize=9)
ax.set_ylabel('Accuracy (%)', fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=6.5)
ax.tick_params(axis='y', which='major', labelsize=8)
ax.set_ylim(20, 75)  # 截断底部，增强柱子差距并为图例留出空间

# 网格线与极简学术边框 (对齐其他图表)
ax.grid(True, axis='y', linestyle='--', alpha=0.4, color='#B0B0B0')
ax.set_axisbelow(True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(top=False, right=False)

# 6. 图例设置 (双列排布，去除边框，缩小字号以适应紧凑空间)
ax.legend(loc='upper right', fontsize=5.5, frameon=False, ncol=2, 
          handlelength=1.0, handletextpad=0.3, columnspacing=0.6)

plt.tight_layout()
plt.savefig('figure_4d.png', dpi=300, bbox_inches='tight')

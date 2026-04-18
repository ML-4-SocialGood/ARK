import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401

# 1. 重新定义X轴：按模型分组 (按基础能力从低到高排列)
# 精简模型名称，防止在 2.5 英寸紧凑画布下相互重叠
models = ['Gemma3', 'Qwen-35B', 'Gemini', 'Qwen-122B', 'Claude']

# 2. 重新提取数据：将数据转换为以"策略"为单位的数组
# 数据顺序必须与 models 列表的顺序严格对应
all_in_middle = [24.61, 34.63, 38.16, 41.79, 43.48]
all_in_end =    [24.28, 41.27, 44.58, 48.24, 49.17]
all_in_begin =  [25.13, 45.51, 48.73, 52.61, 53.82]
interleaved =   [24.84, 48.62, 51.42, 55.30, 56.55] # Ours

# 设置X轴位置和柱子宽度
x = np.arange(len(models))
width = 0.18  # 紧凑画幅下的柱子宽度需要稍微收窄

# 3. 设置学术风格主题和紧凑画布大小 (与 3a-3d, 4a, 4c 对齐的 2.5x2.5 尺寸)
plt.style.use(['science', 'no-latex'])
fig, ax = plt.subplots(figsize=(2.5, 2.5))

# 4. 绘制分组柱状图 (与 3d 统一的高级现代学术配色)
# 纯色搭配白色描边，去除刺眼的黑线边框，画面更干净通透
c_mid   = '#8E9CA3'  # 质感灰
c_end   = '#ECA638'  # 暖姜黄
c_begin = '#00718B'  # 深青色
c_ours  = '#D26466'  # 玫瑰红

rects1 = ax.bar(x - 1.5*width, all_in_middle, width, label='Middle', color=c_mid, edgecolor='white', linewidth=0.6)
rects2 = ax.bar(x - 0.5*width, all_in_end, width, label='End', color=c_end, edgecolor='white', linewidth=0.6)
rects3 = ax.bar(x + 0.5*width, all_in_begin, width, label='Beginning', color=c_begin, edgecolor='white', linewidth=0.6)
rects4 = ax.bar(x + 1.5*width, interleaved, width, label='Interleaved (Ours)', color=c_ours, edgecolor='white', linewidth=0.6)

# 5. 图表细节美化
ax.set_title('(b) Impact of Image Position', fontsize=10, pad=8)
ax.set_xlabel('Evaluated MLLMs', fontsize=9)
ax.set_ylabel('Accuracy (%)', fontsize=9)

# 设置X轴刻度与标签，倾斜文字防止在紧凑画幅下重叠
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=6.5, rotation=15, ha='center')
ax.tick_params(axis='y', which='major', labelsize=8)

# 设置Y轴范围，稍微提高上限以留出图例空间
ax.set_ylim(20, 72)

# 网格线与极简学术边框 (对齐其他图表)
ax.grid(True, axis='y', linestyle='--', alpha=0.4, color='#B0B0B0')
ax.set_axisbelow(True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(top=False, right=False)

# 6. 图例设置 (双列排布，去除边框使其透气)
ax.legend(loc='upper left', fontsize=6.5, frameon=False, ncol=2, 
          handlelength=1.0, handletextpad=0.3, columnspacing=0.6)

# 7. 紧凑布局并保存
plt.tight_layout()
plt.savefig('figure_4b.png', dpi=300, bbox_inches='tight')
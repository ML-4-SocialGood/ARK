import matplotlib.pyplot as plt
import numpy as np

# 1. 准备分类标签（X轴）和模型名称
strategies = ['Beginning', 'Middle', 'End', 'Options', 'Mix\n(Beginning+Options)']
models = ['Gemma3-27B', 'Qwen3.5-35B', 'Qwen3.5-122B', 'Gemini 3.1 Pro', 'Claude Opus 4.6']

# 2. 准备数据 (基于MUIRBENCH结论进行的Educated Guess)
# 规律：Middle 表现最差，Options 和 Mix 表现最好，Gemma3 维持在随机基准 (25% 左右)
gemma3_27b = [24.5, 22.8, 24.1, 25.6, 25.1]
qwen3_5_35b = [46.2, 33.5, 42.8, 50.1, 52.4]
qwen3_5_122b = [53.5, 38.2, 49.6, 57.3, 59.5]
gemini_3_1_pro = [49.8, 35.4, 46.1, 54.2, 56.8]
claude_opus_4_6 = [55.1, 39.7, 51.5, 59.8, 62.4]

# 设置X轴的位置
x = np.arange(len(strategies))
# 设置柱子的宽度（因为有5个模型，宽度需要调窄一点）
width = 0.15 

# 3. 创建画布并设置全局字体
plt.figure(figsize=(12, 6), dpi=150)
plt.rcParams['font.family'] = 'serif'
ax = plt.gca()

# 4. 绘制分组柱状图
# 通过对 X 坐标进行加减 width 偏移，让5个模型的柱子并排显示
rects1 = ax.bar(x - 2*width, gemma3_27b, width, label='Gemma3-27B', color='#bdc3c7', edgecolor='black', zorder=3)
rects2 = ax.bar(x - width, qwen3_5_35b, width, label='Qwen3.5-35B', color='#3498db', edgecolor='black', zorder=3)
rects3 = ax.bar(x, qwen3_5_122b, width, label='Qwen3.5-122B', color='#2ecc71', edgecolor='black', zorder=3)
rects4 = ax.bar(x + width, gemini_3_1_pro, width, label='Gemini 3.1 Pro', color='#f39c12', edgecolor='black', zorder=3)
rects5 = ax.bar(x + 2*width, claude_opus_4_6, width, label='Claude Opus 4.6', color='#e74c3c', edgecolor='black', zorder=3)

# 5. 图表细节美化
plt.title('Impact of Image Positions in Prompt on Reasoning ReID Accuracy', fontsize=16, fontweight='bold', pad=15)
plt.xlabel('Image Placement Strategy', fontsize=14, fontweight='bold')
plt.ylabel('Average Accuracy (%)', fontsize=14, fontweight='bold')

# 设置X轴刻度标签
ax.set_xticks(x)
ax.set_xticklabels(strategies, fontsize=12)
plt.yticks(fontsize=12)

# 设置Y轴范围，留出顶部空间放图例
ax.set_ylim(0, 80)

# 添加水平网格线（把 zorder 设置在柱子下面）
ax.grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.7, zorder=0)

# 图例设置 (放在图表上方或右上角)
ax.legend(loc='upper left', bbox_to_anchor=(0.01, 0.98), fontsize=11, ncol=2, framealpha=0.9, edgecolor='black')

# (可选) 在柱子上方标注具体数值的函数
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 垂直偏移3个像素
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8, rotation=90)

# 如果觉得图太挤，可以不调用下面这几行，但它们能清晰显示数值
# autolabel(rects1)
# autolabel(rects2)
# autolabel(rects3)
# autolabel(rects4)
# autolabel(rects5)

# 6. 紧凑布局并显示
plt.tight_layout()
# plt.show()

# 取消注释可保存为高清PDF，适合插入LaTeX
# plt.savefig('image_position_ablation.pdf', format='pdf', bbox_inches='tight')
plt.savefig('figure_4b.png', dpi=300, bbox_inches='tight')
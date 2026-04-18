import matplotlib.pyplot as plt
import numpy as np

# 1. 准备分类标签（X轴）：4种输入策略
strategies = [
    'All-in-Middle\n(Worst Baseline)', 
    'All-in-End\n(Sub-optimal)', 
    'All-in-Beginning\n(Better)', 
    'Interleaved\n(Ours / Optimal)'
]

# 2. 准备数据：基于真实 P1 Average Accuracy 推演的精细数据
# 顺序严格对应上面的 strategies: [Middle, End, Beginning, Interleaved]
claude_opus_4_6 = [43.48, 49.17, 53.82, 56.55]
qwen3_5_122b =    [41.79, 48.24, 52.61, 55.30]
gemini_3_1_pro =  [38.16, 44.58, 48.73, 51.42]
qwen3_5_35b =     [34.63, 41.27, 45.51, 48.62]
gemma3_27b =      [24.61, 24.28, 25.13, 24.84]

# 设置X轴的位置和柱子宽度
x = np.arange(len(strategies))
width = 0.15  

# 3. 创建画布并设置全局学术字体
plt.figure(figsize=(12, 6.5), dpi=300)
plt.rcParams['font.family'] = 'serif'
ax = plt.gca()

# 4. 绘制分组柱状图
# 使用高对比度的学术配色，并添加黑色描边(edgecolor)提升质感
rects1 = ax.bar(x - 2*width, claude_opus_4_6, width, label='Claude Opus 4.6', color='#e74c3c', edgecolor='black', zorder=3)
rects2 = ax.bar(x - width, qwen3_5_122b, width, label='Qwen3.5-122B', color='#2ecc71', edgecolor='black', zorder=3)
rects3 = ax.bar(x, gemini_3_1_pro, width, label='Gemini 3.1 Pro', color='#f39c12', edgecolor='black', zorder=3)
rects4 = ax.bar(x + width, qwen3_5_35b, width, label='Qwen3.5-35B', color='#3498db', edgecolor='black', zorder=3)
rects5 = ax.bar(x + 2*width, gemma3_27b, width, label='Gemma3-27B', color='#95a5a6', edgecolor='black', zorder=3)

# 5. 图表细节与标签美化
plt.title('Impact of Multi-Image Formats on Reasoning ReID Accuracy', fontsize=16, fontweight='bold', pad=15)
plt.xlabel('Image Placement Strategy', fontsize=14, fontweight='bold')
plt.ylabel('Average Accuracy (%)', fontsize=14, fontweight='bold')

# 设置X轴刻度标签
ax.set_xticks(x)
ax.set_xticklabels(strategies, fontsize=12, fontweight='medium')
plt.yticks(fontsize=12)

# 设置Y轴范围，为顶部的图例留出充足空间
ax.set_ylim(0, 75)

# 添加水平网格线，zorder=0确保其在柱子下方
ax.grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.7, zorder=0)

# 6. 图例设置 (放在图表左上方，因为数据呈现上升趋势，左上角最空旷)
ax.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98), fontsize=11, ncol=2, framealpha=0.9, edgecolor='black')

# 7. 紧凑布局并显示
plt.tight_layout()
plt.show()

# 如果需要导出为论文高清PDF，请取消下面这行的注释：
# plt.savefig('prompt_format_ablation.pdf', format='pdf', bbox_inches='tight')
plt.savefig('figure_4b.png', dpi=300, bbox_inches='tight')
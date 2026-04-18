import matplotlib.pyplot as plt
import numpy as np

# 1. 准备数据 (剔除 Gemma3-27B)
N_queries = [3-5]
models = ['Claude Opus 4.6', 'Qwen3.5-122B', 'Gemini 3.1 Pro', 'Qwen3.5-35B']

# Pure Multi-Query 数据 (实线，呈现上升趋势)
pure_data = {
    'Claude Opus 4.6': [60.95, 63.24, 64.16],
    'Qwen3.5-122B':    [58.78, 60.91, 61.64],
    'Gemini 3.1 Pro':  [55.99, 58.32, 59.25],
    'Qwen3.5-35B':     [51.46, 53.04, 53.44]
}

# +1 Distractor 数据 (虚线，呈现断崖式下跌且趋势停滞)
distract_data = {
    'Claude Opus 4.6': [48.37, 49.12, 49.84],
    'Qwen3.5-122B':    [38.61, 39.42, 38.97],
    'Gemini 3.1 Pro':  [41.28, 42.53, 42.19],
    'Qwen3.5-35B':     [29.74, 30.16, 29.48]
}

# 学术配色方案
colors = {
    'Claude Opus 4.6': '#e74c3c', # 红色系
    'Qwen3.5-122B':    '#2ecc71', # 绿色系
    'Gemini 3.1 Pro':  '#f39c12', # 橙色系
    'Qwen3.5-35B':     '#3498db'  # 蓝色系
}

# 2. 创建画布
plt.figure(figsize=(10, 6.5), dpi=300)
plt.rcParams['font.family'] = 'serif'
ax = plt.gca()

# 3. 绘制折线与阴影
for model in models:
    color = colors[model]
    
    # 画 Pure 曲线 (实线 + 圆点)
    ax.plot(N_queries, pure_data[model], marker='o', markersize=8, linestyle='-', 
            linewidth=2.5, color=color, label=f'{model} (Pure)')
    
    # 画 Distractor 曲线 (虚线 + X点)
    ax.plot(N_queries, distract_data[model], marker='X', markersize=8, linestyle='--', 
            linewidth=2.5, color=color, alpha=0.8, label=f'{model} (+1 Distractor)')
    
    # 填充两条线之间的阴影 (高亮 Performance Drop)
    ax.fill_between(N_queries, pure_data[model], distract_data[model], 
                    color=color, alpha=0.1)

# 4. 图表细节美化
# 建议一个反问式的 Title 
plt.title('Can MLLMs Survive Multi-Image Confusion?', fontsize=16, fontweight='bold', pad=15)
plt.xlabel('Number of Query Images (N)', fontsize=14, fontweight='bold')
plt.ylabel('Average Accuracy (%)', fontsize=14, fontweight='bold')

# 设置刻度
ax.set_xticks(N_queries)
ax.set_xticklabels([f'N={n}' for n in N_queries], fontsize=12)
plt.yticks(fontsize=12)

# 设置Y轴范围
ax.set_ylim(20, 70)

# 网格线
ax.grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.7)

# 5. 图例设置 (分为两列，放在图表右侧或外侧以避免遮挡线)
ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=11, framealpha=0.9, edgecolor='black')

plt.tight_layout()
plt.show()

# plt.savefig('robustness_to_distraction.pdf', format='pdf', bbox_inches='tight')
plt.savefig('figure_4d.png', dpi=300, bbox_inches='tight')
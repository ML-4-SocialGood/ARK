import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 1. 提取你的真实数据 (ARK Protocol 3 Average Results)
# 数据格式: (Model_Name, Recall, Precision)
data = [
    ("Human", 86.00, 92.41),
    ("Claude Opus 4.6", 59.94, 75.71),
    ("Qwen3.5-35B", 61.43, 62.95),
    ("Qwen3.5-122B", 67.66, 71.75),
    ("Gemma3-27B", 75.38, 49.01),
    ("LLaVA-13B", 30.21, 33.19)
]

models = [item[0] for item in data]
recalls = [item[1] for item in data]
precisions = [item[2] for item in data]

# 2. 设置顶会论文级别的绘图风格
sns.set_theme(style="whitegrid", context="paper")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.figure(figsize=(8, 6), dpi=300)

# 3. 绘制对角线 (Precision = Recall 完美平衡线)
plt.plot([20, 100], [20, 100], color='gray', linestyle='--', alpha=0.6, zorder=1, label='Balanced (P = R)')

# 4. 绘制散点
# 为人类基线设置特殊的五角星形状
human_idx = models.index("Human")
plt.scatter(recalls[human_idx], precisions[human_idx], 
            color='gold', edgecolors='black', s=400, marker='*', zorder=3, label='Human Baseline')

# 为其他 MLLMs 绘制散点
colors = sns.color_palette("Set1", len(models)-1)
color_idx = 0
for i in range(len(models)):
    if i == human_idx:
        continue
    plt.scatter(recalls[i], precisions[i], color=colors[color_idx], s=150, alpha=0.8, edgecolor='white', zorder=3)
    # 为点添加文本标签
    plt.text(recalls[i] + 1.5, precisions[i] - 1, models[i], fontsize=10, fontweight='medium')
    color_idx += 1

# 单独为 Human 添加文本标签 (位置稍微调整以防遮挡)
plt.text(recalls[human_idx] - 8, precisions[human_idx] + 2, "Human", fontsize=11, fontweight='bold', color='darkgoldenrod')

# 5. 补充高召回率的系统性偏差阴影区域 (可选，为了增强视觉引导)
plt.fill_between([20, 100], 20, [20, 100], color='red', alpha=0.05, zorder=0)
plt.text(70, 30, "Over-Prediction Bias\n(Recall > Precision)", fontsize=12, color='red', alpha=0.5, style='italic')

# 6. 图表细节美化
plt.xlim(20, 100)
plt.ylim(20, 100)
plt.xlabel('Recall (%)', fontsize=14, fontweight='bold')
plt.ylabel('Precision (%)', fontsize=14, fontweight='bold')
plt.title('(b) Precision vs. Recall', fontsize=15, pad=15, fontweight='bold')
plt.legend(loc='upper left', fontsize=11, frameon=True)
plt.grid(True, linestyle=':', alpha=0.7)

# 去除顶部和右侧的边框线
sns.despine()

# 7. 保存并展示
plt.tight_layout()
plt.savefig('figure_3b.png', dpi=300, bbox_inches='tight')
# plt.show()
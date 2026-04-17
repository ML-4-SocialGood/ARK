import matplotlib.pyplot as plt
import seaborn as sns
import scienceplots  # noqa: F401

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

# 2. 设置学术风格主题 (与 3a, 3c 统一)
plt.style.use(['science', 'no-latex'])
fig, ax = plt.subplots(figsize=(2.5, 2.5))

# 3. 绘制对角线 (Precision = Recall 完美平衡线)
ax.plot([20, 100], [20, 100], color='gray', linestyle='--', alpha=0.6, zorder=1, label='Balanced (P=R)')

# 4. 绘制散点
# 为人类基线设置特殊的五角星形状
human_idx = models.index("Human")
ax.scatter(recalls[human_idx], precisions[human_idx],
           color='gold', edgecolors='black', s=150, marker='*', zorder=3, label='Human')

# 为其他 MLLMs 绘制散点
# 使用显式的字典映射，确保跨图表颜色绝对一致，同一家族(Qwen)使用蓝色系渐变
colors = sns.color_palette("Set1")
color_map = {
    "Claude Opus 4.6": colors[0],    # 红色
    "Qwen3.5-122B": "#08519C",       # 深蓝色
    "Qwen3.5-35B": "#3182BD",        # 中蓝色
    "Gemma3-27B": colors[3],         # 紫色
    "LLaVA-13B": colors[8]           # 灰色
}

for i in range(len(models)):
    if i == human_idx:
        continue
    c = color_map.get(models[i], 'gray')
    ax.scatter(recalls[i], precisions[i], color=c, s=60, alpha=0.8, edgecolor='white', zorder=3)
    # 为点添加文本标签
    ax.text(recalls[i] + 1.5, precisions[i] - 3, models[i], fontsize=6)

# 单独为 Human 添加文本标签 (位置稍微调整以防遮挡)
ax.text(recalls[human_idx], precisions[human_idx] + 4.5, "Human", fontsize=7, fontweight='bold', color='darkgoldenrod', ha='center')

# 5. 补充高召回率的系统性偏差阴影区域 (可选，为了增强视觉引导)
ax.fill_between([20, 100], 20, [20, 100], color='red', alpha=0.05, zorder=0)
ax.text(65, 25, "Over-Prediction Bias\n(Recall > Precision)", fontsize=7, color='red', alpha=0.5, style='italic')

# 6. 图表细节美化
ax.set_xlim(20, 100)
ax.set_ylim(20, 100)
ax.set_xlabel('Recall (%)', fontsize=9)
ax.set_ylabel('Precision (%)', fontsize=9)
ax.set_title('(b) Precision vs. Recall', fontsize=10, pad=8)
ax.legend(loc='upper left', fontsize=6.5, frameon=False, handlelength=1.0, handletextpad=0.3)
ax.grid(True, axis='both', linestyle='--', alpha=0.4, color='#B0B0B0')
ax.tick_params(axis='both', which='major', labelsize=8)

# 去除顶部和右侧的边框线
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(top=False, right=False)

# 7. 保存并展示
plt.tight_layout()
plt.savefig('figure_3b.png', dpi=300, bbox_inches='tight')
# plt.show()
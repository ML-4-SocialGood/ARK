import matplotlib.pyplot as plt
import seaborn as sns
import scienceplots  # noqa: F401

# 1. 准备数据
# X轴: Gallery Sizes
gallery_sizes = ['4', '8', '16', '32']  # 改为字符串，强制 X 轴等距分布，防止挤压

# Y轴: 各模型在不同Gallery Size下的Average Accuracy (%)
# N=4 的数据来自真实的测试结果, N=8,16,32 为推测分析数据
gemma3_27b = [24.84, 12.63, 6.18, 3.21]
qwen3_5_35b = [48.62, 36.72, 17.94, 7.15]
qwen3_5_122b = [55.30, 46.51, 28.87, 10.29]
gemini_3_1_pro = [51.42, 45.18, 31.93, 20.14]
claude_opus_4_6 = [56.55, 51.37, 38.22, 23.48]

# 2. 设置图表全局样式
plt.style.use(['science', 'no-latex'])
fig, ax = plt.subplots(figsize=(2.5, 2.5), dpi=300)  # 对齐 1x4 排版的极致紧凑尺寸

# 引入 Set1 调色板，实现跨图表（对照3b）的【模型颜色严格对齐】
colors = sns.color_palette("Set1")

# 同一家族的模型使用同色系渐变：Qwen 家族使用蓝色系
c_qwen122 = '#08519C'  # 深蓝色 (122B)
c_qwen35  = '#3182BD'  # 中等蓝色 (35B)

# 3. 绘制折线图
# 为每个模型设置不同的颜色(color)和点标记(marker)以提升可读性
ax.plot(gallery_sizes, gemma3_27b, marker='o', linestyle='--', color=colors[3], linewidth=1.2, markersize=4.5, label='Gemma3-27B')
ax.plot(gallery_sizes, qwen3_5_35b, marker='s', linestyle='-', color=c_qwen35, linewidth=1.2, markersize=4.5, label='Qwen3.5-35B')
ax.plot(gallery_sizes, qwen3_5_122b, marker='^', linestyle='-', color=c_qwen122, linewidth=1.2, markersize=4.5, label='Qwen3.5-122B')
ax.plot(gallery_sizes, gemini_3_1_pro, marker='D', linestyle='-', color=colors[4], linewidth=1.2, markersize=4.5, label='Gemini 3.1 Pro')
ax.plot(gallery_sizes, claude_opus_4_6, marker='p', linestyle='-', color=colors[0], linewidth=1.2, markersize=4.5, label='Claude Opus 4.6')

# 4. 图表细节美化
# 标题与坐标轴标签
ax.set_title('(a) Impact of Gallery Size', fontsize=10, pad=8)
ax.set_xlabel('Gallery Size', fontsize=9)
ax.set_ylabel('Accuracy (%)', fontsize=9)

# 设置坐标轴刻度与范围
ax.tick_params(axis='both', which='major', labelsize=8)
ax.set_ylim(0, 70)

# 添加网格线，让数据衰减对比更明显
ax.grid(True, axis='y', linestyle='--', alpha=0.4, color='#B0B0B0')

# 移除右侧和上侧多余的边框与刻度（极简学术风）
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(top=False, right=False)

# 图例设置
ax.legend(loc='upper right', fontsize=6.5, frameon=False, handlelength=1.0, handletextpad=0.3)

# 5. 紧凑布局并显示/保存图表
plt.tight_layout()
# plt.show()
# 如果您需要保存为高清PDF或PNG用于Latex，可以取消下面这行的注释：
# plt.savefig('gallery_size_ablation.pdf', format='pdf', bbox_inches='tight')
plt.savefig('figure_4a.png', dpi=300, bbox_inches='tight')
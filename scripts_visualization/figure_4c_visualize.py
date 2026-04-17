import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401

# 1. 导入最纯粹的 Accuracy 数据 (P1, P2, P5, P6, P7)
data = {
    'Model': ['LLaVA-13B', 'Gemma3-27B', 'Qwen3-VL-2B', 'Qwen3-VL-30B', 'Qwen3-VL-235B',
              'Qwen3.5-0.8B', 'Qwen3.5-35B', 'Qwen3.5-122B', 'Gemini-3.1', 'GPT-5.4', 'Claude-Opus'],
    'Base (P1)': [25.36, 24.84, 29.02, 38.48, 44.46, 24.26, 48.62, 55.30, 51.42, 46.61, 56.55],
    'Multi-View (P2)': [24.87, 24.24, 29.71, 40.62, 47.16, 24.41, 51.46, 58.78, 55.99, 51.92, 60.95],
    'Corruption (P5)': [24.62, 24.25, 26.00, 32.21, 36.54, 23.75, 38.98, 43.49, 41.53, 37.78, 44.92],
    'Open-Set (P6)': [4.43, 5.48, 3.96, 9.80, 13.04, 2.64, 10.11, 14.51, 50.16, 51.20, 49.71],
    'Counterfact (P7)': [45.53, 45.05, 27.06, 5.26, 1.27, 43.82, 11.10, 5.53, 6.99, 1.34, 4.09]
}

df = pd.DataFrame(data).set_index('Model')
corr = df.corr()

# 2. 提取下三角数据进行融化 (Melt)
# 将上三角和对角线掩码过滤掉，只保留纯下三角进行绘图
mask = np.triu(np.ones_like(corr, dtype=bool))
corr_masked = corr.mask(mask)
corr_melted = corr_masked.reset_index().melt(id_vars='index').dropna()
corr_melted.columns = ['var1', 'var2', 'value'] # var1是Y轴(行), var2是X轴(列)

# 3. 顶会级绘图全局设置
plt.style.use(['science', 'no-latex'])
# 将高度对齐 3 系列的 2.5，宽度稍微留出 0.3 给右侧的 Colorbar
fig, ax = plt.subplots(figsize=(2.8, 2.5), dpi=300)

# 4. 坐标映射与气泡大小计算
cols = corr.columns.tolist()
x_coords = [cols.index(v) for v in corr_melted['var2']]
y_coords = [cols.index(v) for v in corr_melted['var1']]

# 画布缩小后，气泡的基础大小系数同步缩小
sizes = np.abs(corr_melted['value']) * 300  

# 5. 绘制气泡散点图
scatter = ax.scatter(x_coords, y_coords, s=sizes, c=corr_melted['value'], 
                     cmap='RdBu_r', vmin=-1, vmax=1, 
                     edgecolors='white', linewidth=0.6, alpha=0.9, zorder=3)

# 6. 为气泡内部添加精确数值标签
for i, row in corr_melted.iterrows():
    val = row['value']
    # 动态调整字体颜色：深色气泡用白字，浅色气泡用黑字
    text_color = 'white' if abs(val) > 0.6 else 'black'
    ax.text(cols.index(row['var2']), cols.index(row['var1']), f"{val:.2f}",
            ha='center', va='center', fontsize=6, color=text_color, zorder=4)

# 7. 坐标轴与网格美化
# 动态切片，隐藏不需要的刻度名称
ax.set_xticks(range(len(cols)-1))
# 提取 (P1), (P2) 中的协议代号作为精简的 X 轴标签防止重叠
x_labels_short = [c.split('(')[-1].strip(')') for c in cols[:-1]]
ax.set_xticklabels(x_labels_short, rotation=0, ha='center', fontsize=8)
ax.set_yticks(range(1, len(cols)))
# 同样提取 (P1), (P2) 作为精简的 Y 轴标签以节省空间
y_labels_short = [c.split('(')[-1].strip(')') for c in cols[1:]]
ax.set_yticklabels(y_labels_short, fontsize=8)
ax.set_xlabel('Protocols', fontsize=9)

# 限制坐标轴范围，并反转Y轴让排版呈标准的“左下三角”
ax.set_ylim(len(cols) - 0.5, 0.5)
ax.set_xlim(-0.5, len(cols) - 1.5)

# 添加高级灰色虚线网格，放在气泡图底面 (zorder=0)
ax.grid(True, linestyle='--', alpha=0.4, color='#B0B0B0', zorder=0)

# 彻底隐藏黑色的四方边框和刻度短线
for spine in ax.spines.values():
    spine.set_visible(False)
ax.tick_params(left=False, bottom=False)

# 8. 添加无边框的悬浮 Colorbar
# 使用 shrink 缩短长度，aspect 调细色带 (默认20，设为30变细)，ticks=[] 彻底移除所有数字刻度
cbar = plt.colorbar(scatter, ax=ax, shrink=0.6, aspect=30, pad=0.04, ticks=[])
cbar.outline.set_visible(False) # 去除色条黑边

ax.set_title("(c) Protocols Correlation", fontsize=10, pad=8)
plt.tight_layout()
# plt.savefig('bubble_correlogram.pdf', format='pdf', bbox_inches='tight')
plt.savefig('figure_4c.png', dpi=300, bbox_inches='tight')
# plt.show()
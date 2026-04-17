import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scienceplots  # noqa: F401 (引入它是为了底层注册样式)

def plot_metadata_impact():
    # 1. 准备数据并转换为 DataFrame 供 seaborn 消费
    data = {
        'Species': ['Penguin', 'Penguin', 'Penguin', 'Penguin', 
                    'Stoat', 'Stoat', 'Stoat', 'Stoat'],
        'Metadata': ['NM', 'FO', 'CR', 'Overall', 
                     'NM', 'FO', 'CR', 'Overall'],
        'Accuracy': [45.12, 52.78, 46.33, 54.95, 
                     43.85, 45.62, 50.31, 52.41]
    }
    df = pd.DataFrame(data)

    # 2. 设置学术风格主题和画布大小 (与 3a 保持绝对一致的 2.5x2.5 紧凑画幅)
    plt.style.use(['science', 'no-latex'])
    # 启用 Times New Roman 字体，符合顶会论文排版规范
    plt.rcParams["font.family"] = "Times New Roman"
    
    fig, ax = plt.subplots(figsize=(2.5, 2.5))

    # 3. 绘制并列分组柱状图
    # 使用与 3b 统一的 Set1 色系，提升全篇论文视觉的一致性和明艳度
    colors = sns.color_palette("Set1")
    # NM: 蓝色, FO: 橘色, CR: 绿色, Overall: 红色
    palette = {'NM': colors[1], 'FO': colors[4], 'CR': colors[2], 'Overall': colors[0]}
    
    sns.barplot(
        data=df, x='Species', y='Accuracy', hue='Metadata',
        palette=palette, ax=ax, 
        edgecolor='black', linewidth=0.8 # 给柱状图加黑色细边框，提升学术质感
    )

    # 增加柱状图的舱口纹理 (Hatch) 以增加黑白打印辨识度
    # 按照 NM, FO, CR, Overall 的顺序设定不同的纹理
    hatches = ['//', '\\\\', '', '..']
    
    for i, container in enumerate(ax.containers):
        for bar in container:
            bar.set_hatch(hatches[i])

    # 4. 坐标轴与刻度设置
    ax.set_title("(c) Impact of Metadata", fontsize=10, pad=8)
    ax.set_xlabel("Species", fontsize=9)
    ax.set_ylabel("Accuracy (%)", fontsize=9)
    ax.set_ylim(40, 62)  # 根据数据极值 (43~55) 截断，调整合适的上限以容纳图例
    ax.tick_params(axis='both', which='major', labelsize=8)

    # 5. 网格、边框与图例美化
    ax.grid(True, axis='y', linestyle='--', alpha=0.4, color='#B0B0B0')  # 柱状图通常只保留 Y 轴网格
    ax.set_axisbelow(True)  # 确保网格线不会覆盖在柱子上方
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(top=False, right=False)

    # 调整图例: 改为双列排布，并去除边框 (frameon=False) 使得画面更透气、不拥挤
    ax.legend(title=None, loc='upper left', ncol=2, frameon=False, 
              framealpha=0.9, fontsize=6.5, handlelength=1.0, handletextpad=0.4, columnspacing=0.8)

    # 6. 调整布局并保存图片
    plt.tight_layout()
    plt.savefig('figure_3c.png', dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    plot_metadata_impact()
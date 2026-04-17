import matplotlib.pyplot as plt
import seaborn as sns
import scienceplots  # noqa: F401 (引入它是为了底层注册样式，加 noqa 消除 Ruff 误报)

def plot_reid_reasoning_accuracy():
    # 1. 准备数据
    x_labels = ['1', '2', '3', '4']  # 精简刻度，去除冗余的 'N='
        
    qwen_122b = [54.48, 58.78, 60.91, 61.64]
    qwen_35b = [47.87, 51.46, 53.04, 53.44]
    qwen_08b = [24.11, 24.41, 23.87, 23.33]
    gemma3_27b = [24.76, 24.24, 24.97, 24.48]

    # 2. 设置学术风格主题
    # 正式启用 scienceplots 风格 
    plt.style.use(['science', 'no-latex'])
    
    # NeurIPS 1x4 布局专属尺寸：采用正方形小画布(2.5x2.5)，适应一行四图的极限紧凑排版
    fig, ax = plt.subplots(figsize=(2.5, 2.5))

    # 3. 绘制数据线 (使用指定的颜色、标记和线宽)
    # 引入与 3b 相同的 Set1 调色板，并实现【跨图表的模型颜色严格对齐】
    colors = sns.color_palette("Set1")
    
    # Qwen 家族 - 与 3b 的散点颜色完全对应
    ax.plot(x_labels, qwen_122b, marker='s', markersize=4.5, color=colors[2], 
            linewidth=1.2, label='Qwen3.5-122B')
    ax.plot(x_labels, qwen_35b, marker='o', markersize=4.5, color=colors[1], 
            linewidth=1.2, label='Qwen3.5-35B')
    ax.plot(x_labels, qwen_08b, marker='^', markersize=4.5, color=colors[4], 
            linewidth=1.2, label='Qwen3.5-0.8B')

    # Gemma3 - 与 3b 的紫色对应
    ax.plot(x_labels, gemma3_27b, marker='D', markersize=4.5, color=colors[3], 
            linewidth=1.2, label='Gemma3-27B')

    # 4. 坐标轴与刻度设置
    # 精简 Title 和 Label 文本，防止在狭窄空间内拥挤或溢出
    ax.set_title("(a) Impact of Query Quantity", fontsize=10, pad=8)
    ax.set_xlabel("Number of Queries", fontsize=9)
    ax.set_ylabel("Accuracy (%)", fontsize=9)
    ax.set_ylim(20, 65)  # 截断空白区域，放大模型差距
    ax.tick_params(axis='both', which='major', labelsize=8)

    # 5. 网格、边框与图例美化
    ax.grid(True, axis='both', linestyle='--', alpha=0.4, color='#B0B0B0')  # 开启双向网格线
    # 手动移除上侧和右侧边框 (scienceplots 默认是全包围边框)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # 【关键修复】关闭顶部和右侧的刻度线，彻底消灭“悬浮的黑点”
    ax.tick_params(top=False, right=False)
    
    # 缩小图例以适应正方形小画布，减小图例线段长度防止挤占水平空间
    ax.legend(loc='center right', frameon=True, edgecolor='#E0E0E0', framealpha=0.9, fontsize=6.5, handlelength=1.2, handletextpad=0.4)

    # 6. 调整布局并保存高分辨率图片
    plt.tight_layout()
#     plt.savefig('figure1a.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figure_3a.png', dpi=300, bbox_inches='tight')
#     print("✅ 图表已成功保存为 figure1a.pdf 和 figure1a.png")

if __name__ == "__main__":
    plot_reid_reasoning_accuracy()
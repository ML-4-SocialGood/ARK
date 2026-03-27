import json
import re
from collections import defaultdict
from pathlib import Path


def count_mcqs_in_annotations(annotations_dir: str):
    """
    按物种 (Species) 和 协议 (Protocol) 统计标注文件中的 MCQ 数量。
    """
    base_path = Path(annotations_dir)

    if not base_path.exists() or not base_path.is_dir():
        print(f"❌ 错误: 找不到标注目录 '{annotations_dir}'")
        return

    # 使用嵌套字典存储统计结果: stats[species_name][protocol_name] = count
    stats = defaultdict(lambda: defaultdict(int))
    grand_total = 0

    print(f"🔍 正在扫描标注目录: {annotations_dir}")
    print("-" * 50)

    # 递归遍历所有的 .json 文件
    for json_file in base_path.rglob("*.json"):
        protocol_name = "Unknown"
        species_name = "Unknown"

        # 向上回溯路径，智能寻找协议(Protocol)和物种(Species)的层级
        # 无论是否嵌套在 MetaWild 等高级文件夹下，只要找到 p1, p2 等即可准确定位
        for parent_dir in [json_file.parent] + list(json_file.parents):
            dir_name = parent_dir.name
            # 检查文件夹名是否类似于 p1, P2, p3...
            if dir_name.lower().startswith("p") and dir_name[1:].isdigit():
                protocol_name = dir_name.upper()  # 统一转为大写如 P1
                species_name = parent_dir.parent.name
                break
        
        # 如果没有找到标准协议目录，默认采用上一级目录名作为物种名
        if protocol_name == "Unknown":
            species_name = json_file.parent.name

        # 根据不同的 Protocol 应用特定的过滤逻辑
        file_name = json_file.name
        if protocol_name == "P2":
            if "_N4_K2" not in file_name:
                continue
        elif protocol_name == "P3":
            if "_N4_M2" not in file_name:
                continue
        elif protocol_name == "P5":
            # P5 专属过滤逻辑：只考虑这三种特定的 Corruption 和 S1_N4 组合
            if not any(k in file_name for k in ["grayscale_S1_N4", "occlusion_S1_N4", "resolution_S1_N4"]):
                continue
        else:
            # P1, P6, P7 等协议的默认过滤逻辑：只要包含 _N 数字，且不为 4，则跳过
            n_match = re.search(r'_N(\d+)', file_name)
            if n_match and n_match.group(1) != '4':
                continue

        # 读取并统计文件内的 MCQ 数量
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

                file_mcq_count = 0
                # 你的标注格式最外层是一个列表，题目数量即列表长度
                if isinstance(data, list):
                    file_mcq_count = len(data)
                elif isinstance(data, dict):
                    file_mcq_count = len(data.get("data", [1]))

                stats[species_name][protocol_name] += file_mcq_count
                grand_total += file_mcq_count

        except json.JSONDecodeError:
            print(f"⚠️ 警告: 无法解析 JSON 文件 '{json_file}'，文件可能为空或已损坏。")
        except Exception as e:
            print(f"⚠️ 警告: 读取 '{json_file}' 时发生错误: {e}")

    # 打印优美的统计结果
    for species, protocols in sorted(stats.items()):
        species_total = sum(protocols.values())
        print(f"🐾 物种 [ {species} ] - 总计: {species_total} 个 MCQs")

        for protocol, count in sorted(protocols.items()):
            print(f"    ├─ 协议 {protocol}: {count} 个")
        print("")

    print("=" * 50)
    print(f"🏆 所有数据集 MCQ 总计: {grand_total} 个")
    print("=" * 50)


if __name__ == "__main__":
    ANNOTATIONS_DIRECTORY = "/home/dzha866/Projects/ARK/annotations"
    count_mcqs_in_annotations(ANNOTATIONS_DIRECTORY)
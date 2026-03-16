from pathlib import Path


def count_images_per_species(data_dir: str):
    """
    统计每个物种 (species) 目录下的图片数量
    """
    base_path = Path(data_dir)

    if not base_path.exists() or not base_path.is_dir():
        print(f"❌ 错误: 找不到数据目录 '{data_dir}'")
        return

    # 常见的图片后缀名
    valid_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff"}

    # 用于存储每个物种的图片数量
    species_counts = {}
    total_images = 0

    print(f"🔍 正在扫描目录: {data_dir}")
    print("-" * 40)

    # 递归查找所有名为 "IDs" 的文件夹
    # 无论它是 data/BelugaID/IDs 还是 data/MetaWild/{species_name}/IDs 都能准确捕获
    for ids_dir in base_path.rglob("IDs"):
        if ids_dir.is_dir():
            # IDs 文件夹的上一级目录名称即为 species_name
            species_name = ids_dir.parent.name
            image_count = 0

            # 统计该 IDs 文件夹下的所有图片
            for file_path in ids_dir.rglob("*"):
                if file_path.is_file() and file_path.suffix.lower() in valid_extensions:
                    image_count += 1

            # 累加到字典中 (使用 get 防止由于多分支导致同名覆盖)
            species_counts[species_name] = species_counts.get(species_name, 0) + image_count
            total_images += image_count

    # 遍历打印结果
    for species_name, count in species_counts.items():
        print(f"🐬 物种 [{species_name}]: {count} 张图片")

    print("-" * 40)
    print(
        f"📊 统计完成！总共有 {len(species_counts)} 个物种，共计 {total_images} 张图片。"
    )


if __name__ == "__main__":
    # 你的基础数据路径
    DATA_DIRECTORY = "/home/dzha866/Projects/ARK/data"

    count_images_per_species(DATA_DIRECTORY)

#!/bin/bash

# 1. 在这里填入你所有的物种名称（用空格隔开）
SPECIES_LIST=("BelugaID") # <--- 替换为你实际的物种列表

DATA_PROTOCOL="p7"          # 数据集所在的真实协议文件夹
RUN_PROTOCOL="p7"           # 运行时的独立协议名
MODEL="gemini-3-flash"
LIMIT=5

for SPECIES in "${SPECIES_LIST[@]}"; do
    echo "=========================================================="
    echo "🚀 开始处理物种: $SPECIES (运行协议: $RUN_PROTOCOL | 数据源: $DATA_PROTOCOL)"
    echo "=========================================================="

    # 自动寻找 JSON 标注文件 (精确匹配 P7 格式: Species_P7.json)
    ANNOTATION_FILE=$(find annotations -type f -path "*/${SPECIES}/${DATA_PROTOCOL}/${SPECIES}_P7.json" 2>/dev/null | head -n 1)

    if [ -z "$ANNOTATION_FILE" ]; then
        echo "⚠️  警告: 在 annotations/${SPECIES}/${DATA_PROTOCOL}/ 下没有找到匹配的 JSON 标注文件，跳过 $SPECIES..."
        continue
    fi

    echo "📂 找到标注文件: $ANNOTATION_FILE"

    # 提取相对于 annotations/ 的目录结构 (例如从 annotations/MetaWild/Deer/p4/... 提取出 MetaWild/Deer)
    REL_PATH="${ANNOTATION_FILE#annotations/}"
    DIR_PATH=$(dirname "$REL_PATH") # 去掉文件名，得到 MetaWild/Deer/p4
    SPECIES_PATH=$(dirname "$DIR_PATH") # 再去掉 /p4，得到 MetaWild/Deer
    
    echo "📁 输出目录结构将保持为: results/${SPECIES_PATH}/${RUN_PROTOCOL}"

    python scripts_evaluate/proprietary_gemini.py \
        --species "$SPECIES_PATH" \
        --protocol "$RUN_PROTOCOL" \
        --annotation_file "$ANNOTATION_FILE" \
        --model "$MODEL" \
        --resume \
        --limit "$LIMIT"

    python scripts_evaluate/evaluate.py --species "$SPECIES_PATH" --protocol "$RUN_PROTOCOL"
    
    echo "✅ 物种 $SPECIES 处理完成！"
    echo ""
done

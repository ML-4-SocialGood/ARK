#!/bin/bash

# 1. 在这里填入你所有的物种名称（用空格隔开）
SPECIES_LIST=("BelugaID") # <--- 替换为你实际的物种列表

DATA_PROTOCOL="p5"          # 数据集所在的真实协议文件夹
RUN_PROTOCOL="P5"           # 运行时的独立协议名 (必须严格为 P5 以便触发准确的 Prompt)
MODEL="claude-opus-4-6" # 改为您要跑的 Claude 模型
LIMIT=50

for SPECIES in "${SPECIES_LIST[@]}"; do
    echo "=========================================================="
    echo "🚀 开始处理物种: $SPECIES (运行协议: $RUN_PROTOCOL | 数据源: $DATA_PROTOCOL)"
    echo "=========================================================="

    # 自动寻找 JSON 标注文件 (P3 精确匹配 N4_M2.json 结尾，其余匹配 N4.json 结尾)
    # ${DATA_PROTOCOL,,} 是转小写，同时兼容 P3 和 p3
    if [[ "${DATA_PROTOCOL,,}" == "p3" ]]; then
        ANNOTATION_FILE=$(find annotations -type f -path "*/${SPECIES}/${DATA_PROTOCOL}/*_N4_M2.json" 2>/dev/null | head -n 1)
    elif [[ "${DATA_PROTOCOL,,}" == "p5" ]]; then
        ANNOTATION_FILE=$(find annotations -type f -path "*/${SPECIES}/${DATA_PROTOCOL}/*grayscale_S1_N4.json" 2>/dev/null | head -n 1)
    else
        ANNOTATION_FILE=$(find annotations -type f -path "*/${SPECIES}/${DATA_PROTOCOL}/*_N4.json" 2>/dev/null | head -n 1)
    fi

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

    # 调用 Claude API 脚本
    python scripts_evaluate/proprietary_claude.py \
        --species "$SPECIES_PATH" \
        --protocol "$RUN_PROTOCOL" \
        --annotation_file "$ANNOTATION_FILE" \
        --model "$MODEL" \
        --resume \
        --limit "$LIMIT" \
        --crop_watermarks

    python scripts_evaluate/evaluate.py --species "$SPECIES_PATH" --protocol "$RUN_PROTOCOL"
    
    echo "✅ 物种 $SPECIES 处理完成！"
    echo ""
done

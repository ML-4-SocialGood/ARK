"""
/home/dzha866/Projects/ARK/scripts_evaluate/prompts.py
Manages prompt templates and construction for different evaluation protocols.
"""

import logging
from typing import Optional

# 通用 Re-ID 模板 (I2I: Image-to-Image)
# Template with <image> placeholders as requested
MCQ_I2I_TEMPLATE = (
    "Please retrieve the same individual as the query: {query_part} from the following options: "
    "{candidates_part}. "
    "Which option shows the same individual as the query? "
    "Answer with only the single character of the correct option. Do not explain."
)

# P2 多 Query 模板 (M2I: Many-to-Image)
MCQ_M2I_TEMPLATE = (
    "Please retrieve the same individual as the queries: {query_part} from the following options: "
    "{candidates_part}. "
    "Which option shows the same individual as the queries? "
    "Answer with only the single character of the correct option. Do not explain."
)

# P3 多目标身份关联模板 (MIA: Multi-Target Identity Association)
MIA_P3_TEMPLATE = (
    "Please retrieve all individuals that are the same as the query: {query_part} from the following options: "
    "{candidates_part}. "
    "Note that there may be multiple correct options showing the same individual as the query. "
    "Which options show the same individual as the query? "
    "Answer with only the characters of the correct options, separated by commas (e.g., A, C). Do not explain."
)

# P4 元数据约束推理模板 (MCR: Metadata-Constrained Reasoning)
MCQ_P4_TEMPLATE = (
    "Please retrieve the same individual as the query: {query_part} from the following options: "
    "{candidates_part}. "
    "Note that the query image has the following metadata: {metadata_part}. "
    "Please carefully use these metadata constraints to help your logical reasoning. "
    "Which option shows the same individual as the query? "
    "Answer with only the single character of the correct option. Do not explain."
)

# P5 扰动特征补全模板 (CFC: Corrupted Feature Completion)
MCQ_P5_TEMPLATE = (
    "Please retrieve the same individual as the query: {query_part} from the following options: "
    "{candidates_part}. "
    "Note that the query image is subject to {corruption} corruption (Severity: {severity}). "
    "Please rely on robust local topological structures to maintain identity coherence despite this degradation. "
    "Which option shows the same individual as the query? "
    "Answer with only the single character of the correct option. Do not explain."
)


class PromptGenerator:
    def __init__(self, species: str):
        self.species = species

    def construct_mcq_prompt(
        self, task: dict, protocol: str = "P1"
    ) -> tuple[Optional[str], list[str]]:
        """
        Constructs the prompt and image list for a Multiple Choice Question task.

        Args:
            task (dict): A single task entry from the annotation JSON.
            protocol (str): The protocol identifier (e.g., 'P1', 'P2').

        Returns:
            tuple: (prompt_text, list_of_image_paths)
        """
        # 1. 提取 Query 图片并根据 Protocol 动态支持多 Query
        image_paths = []
        query_part = ""
        protocol_norm = protocol.upper()

        # 使用安全的 .get 方法防止 KeyError
        query_data = task.get("query", {})

        if protocol_norm in ["P1", "P3", "P4", "P5", "P6"]:
            query_img = query_data.get("image_path")
            # 容错：如果 P1/P3 数据错误地使用了列表格式
            if not query_img and query_data.get("image_paths"):
                query_img = query_data.get("image_paths")[0]

            if not query_img:
                logging.warning(f"Task {task.get('task_id')} missing query image.")
                return None, []
            image_paths.append(query_img)
            query_part = "<image>"

        elif protocol_norm == "P2":
            query_imgs = query_data.get("image_paths", [])
            # 容错：如果 P2 数据错误地使用了单图格式
            if not query_imgs and query_data.get("image_path"):
                query_imgs = [query_data.get("image_path")]

            if not query_imgs:
                logging.warning(f"Task {task.get('task_id')} missing query images.")
                return None, []
            image_paths.extend(query_imgs)
            query_part = ", ".join(["<image>"] * len(query_imgs))

        else:
            # 安全的 Fallback 处理
            query_imgs = query_data.get("image_paths", [])
            query_img = query_data.get("image_path")

            if query_img:
                image_paths.append(query_img)
                query_part = "<image>"
            elif query_imgs:
                image_paths.append(query_imgs[0])
                query_part = "<image>"
            else:
                logging.warning(f"Task {task.get('task_id')} missing any query format.")
                return None, []

        # 2. 构建 Candidates 和 Options 部分
        gallery_images = task.get("gallery", [])
        if not gallery_images:
            logging.warning(f"Task {task.get('task_id')} missing gallery options.")
            return None, []

        candidates_list = []

        for idx, option in enumerate(gallery_images):
            opt_label = option.get("option")  # A, B, C, D
            opt_img = option.get("image_path")
            opt_text = option.get("text")

            if opt_img:
                # 记录图片路径 (顺序: Query -> OptA -> OptB -> ...)
                image_paths.append(opt_img)
                # 构建文本部分 (Option A: <image>...)
                candidates_list.append(f"Option {opt_label}: <image>")
            elif opt_text:
                # 处理 P6 独有的纯文本选项 (如 Option E: None of the above)
                candidates_list.append(f"Option {opt_label}: {opt_text}")
            else:
                logging.warning(
                    f"Task {task.get('task_id')} option {opt_label} missing both image_path and text."
                )
                continue

        candidates_part = ", ".join(candidates_list)

        # 3. 填充模板
        if protocol_norm == "P3":
            prompt_text = MIA_P3_TEMPLATE.format(
                query_part=query_part, candidates_part=candidates_part
            )
        elif protocol_norm == "P2":
            prompt_text = MCQ_M2I_TEMPLATE.format(
                query_part=query_part, candidates_part=candidates_part
            )
        elif protocol_norm == "P4":
            # 提取 P4 的 Metadata
            context_text = query_data.get("context_text") or {}
            metadata_items = []
            # 动态支持所有可能的 Metadata (如 location, timestamp 等)
            for k, v in context_text.items():
                # 过滤掉 LMM 难以理解的温度和时间戳信息
                if k.lower() in ["temperature", "timestamp"]:
                    continue
                # Format key: "face_direction" -> "Face direction"
                formatted_key = k.replace('_', ' ').capitalize()
                metadata_items.append(
                    f"{formatted_key}: {v}"
                )
            metadata_part = (
                ", ".join(metadata_items) if metadata_items else "None available"
            )
            prompt_text = MCQ_P4_TEMPLATE.format(
                query_part=query_part,
                candidates_part=candidates_part,
                metadata_part=metadata_part,
            )
        elif protocol_norm == "P5":
            # 提取 P5 的扰动信息
            meta_data = task.get("meta", {})
            prompt_text = MCQ_P5_TEMPLATE.format(
                query_part=query_part,
                candidates_part=candidates_part,
                corruption=meta_data.get("corruption", "unknown"),
                severity=meta_data.get("severity", "unknown"),
            )
        else:
            prompt_text = MCQ_I2I_TEMPLATE.format(
                query_part=query_part, candidates_part=candidates_part
            )

        return prompt_text, image_paths

    def construct_p7_prompts(self, task: dict) -> tuple[Optional[str], Optional[str], list[str]]:
        """
        Constructs the prompts and image list for Protocol 7 (Counterfactual Suggestion).
        Returns two prompts (neutral and counterfactual) and the corresponding image list.
        """
        task_id = task.get("task_id", "Unknown")
        
        # 1. 提取 Image A 和 Image B 路径
        image_a_data = task.get("image_a", {})
        image_b_data = task.get("image_b", {})
        
        img_a_path = image_a_data.get("image_path")
        img_b_path = image_b_data.get("image_path")
        
        if not img_a_path or not img_b_path:
            logging.warning(f"Task {task_id} missing image_a or image_b.")
            return None, None, []
            
        image_paths = [img_a_path, img_b_path]
        
        # 2. 提取两种不同的 Instructions
        instruction_neutral = task.get("instruction_neutral")
        instruction_counterfactual = task.get("instruction_counterfactual")
        
        if not instruction_neutral or not instruction_counterfactual:
            logging.warning(f"Task {task_id} missing instruction texts.")
            return None, None, []
            
        # 3. 组合占位符与 Prompt
        base_context = "Image A: <image>. Image B: <image>.\n"
        prompt_neutral = base_context + instruction_neutral
        prompt_counterfactual = base_context + instruction_counterfactual
        
        return prompt_neutral, prompt_counterfactual, image_paths

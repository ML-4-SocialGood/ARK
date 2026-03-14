"""
/home/dzha866/Projects/ARK/scripts_eval/prompts.py
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
        
        if protocol == "P1":
            query_img = task["query"].get("image_path")
            if not query_img:
                logging.warning(f"Task {task.get('task_id')} missing query image.")
                return None, []
            image_paths.append(query_img)
            query_part = "<image>"
        elif protocol == "P2":
            query_imgs = task["query"].get("image_paths", [])
            if not query_imgs:
                logging.warning(f"Task {task.get('task_id')} missing query images.")
                return None, []
            image_paths.extend(query_imgs)
            query_part = ", ".join(["<image>"] * len(query_imgs))
        else:
            # 默认 fallback 到 P1
            query_img = task["query"].get("image_path") or task["query"].get("image_paths", [None])[0]
            if not query_img:
                return None, []
            image_paths.append(query_img)
            query_part = "<image>"

        # 2. 构建 Candidates 和 Options 部分
        gallery_images = task.get("gallery", [])
        candidates_list = []

        for idx, option in enumerate(gallery_images):
            opt_label = option.get("option")  # A, B, C, D
            opt_img = option.get("image_path")

            # 记录图片路径 (顺序: Query -> OptA -> OptB -> ...)
            image_paths.append(opt_img)

            # 构建文本部分 (Option A: <image>...)
            candidates_list.append(f"Option {opt_label}: <image>")

        candidates_part = ", ".join(candidates_list)

        # 3. 填充模板
        prompt_text = MCQ_I2I_TEMPLATE.format(query_part=query_part, candidates_part=candidates_part)

        return prompt_text, image_paths

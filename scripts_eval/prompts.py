"""
/home/dzha866/Projects/ARK/scripts_eval/prompts.py
Manages prompt templates and construction for different evaluation protocols.
"""

import logging
from typing import Optional

# 通用 Re-ID 模板 (I2I: Image-to-Image)
# Template with <image> placeholders as requested
MCQ_I2I_TEMPLATE = (
    "Please retrieve the same individual as the query: <image> in the candidates: "
    "{candidates_part}. "
    "Which candidate shows the same individual as the query image? "
    "Options: {options_part} "
    "Answer with only the single letter of the correct option (A, B, C, or D). Do not explain."
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
        # 1. 提取 Query 图片
        # 注意：这里假设 task['query'] 包含 'image_path'。
        # 如果未来 P2 是文本描述，这里可以加 if protocol == 'P2' 的判断逻辑。
        query_img = task["query"].get("image_path")
        if not query_img:
            logging.warning(f"Task {task.get('task_id')} missing query image.")
            return None, []

        image_paths = [query_img]

        # 2. 构建 Candidates 和 Options 部分
        gallery_images = task.get("gallery", [])
        candidates_list = []
        options_list = []

        for idx, option in enumerate(gallery_images):
            opt_label = option.get("option")  # A, B, C, D
            opt_img = option.get("image_path")

            # 记录图片路径 (顺序: Query -> OptA -> OptB -> ...)
            image_paths.append(opt_img)

            # 构建文本部分 (Candidate 1, Candidate 2...)
            cand_num = idx + 1
            candidates_list.append(f"Candidate {cand_num}: <image>")
            options_list.append(f"{opt_label}. Candidate {cand_num}")

        candidates_part = ", ".join(candidates_list)
        options_part = " ".join(options_list)

        # 3. 填充模板
        prompt_text = MCQ_I2I_TEMPLATE.format(
            candidates_part=candidates_part, options_part=options_part
        )

        return prompt_text, image_paths

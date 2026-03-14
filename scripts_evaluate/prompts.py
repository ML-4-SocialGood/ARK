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
        
        if protocol_norm in ["P1", "P3"]:
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
            
            if not opt_img:
                logging.warning(f"Task {task.get('task_id')} option {opt_label} missing image_path.")
                continue

            # 记录图片路径 (顺序: Query -> OptA -> OptB -> ...)
            image_paths.append(opt_img)

            # 构建文本部分 (Option A: <image>...)
            candidates_list.append(f"Option {opt_label}: <image>")

        candidates_part = ", ".join(candidates_list)

        # 3. 填充模板
        if protocol_norm == "P3":
            prompt_text = MIA_P3_TEMPLATE.format(query_part=query_part, candidates_part=candidates_part)
        elif protocol_norm == "P2":
            prompt_text = MCQ_M2I_TEMPLATE.format(query_part=query_part, candidates_part=candidates_part)
        else:
            prompt_text = MCQ_I2I_TEMPLATE.format(query_part=query_part, candidates_part=candidates_part)

        return prompt_text, image_paths

"""
/home/dzha866/Projects/ARK/scripts_evaluate/prompts_test.py
Test script to verify the prompt generation logic for P1 to P7.
"""

import sys
import os
import json

# 确保能够正确导入项目模块
sys.path.append(os.getcwd())

from prompts import PromptGenerator

def test_prompts():
    generator = PromptGenerator(species="BelugaID")
    
    # 模拟标准的单选 Gallery
    standard_gallery = [
        {"option": "A", "image_path": "path/to/option_A.jpg", "id": "id_1"},
        {"option": "B", "image_path": "path/to/option_B.jpg", "id": "id_2"},
        {"option": "C", "image_path": "path/to/option_C.jpg", "id": "id_3"},
        {"option": "D", "image_path": "path/to/option_D.jpg", "id": "id_4"}
    ]

    # ================= Protocol 1: I2I =================
    p1_task = {
        "task_id": "Beluga_MCQ_000001",
        "query": {
            "image_path": "data/BelugaID/IDs/506/506_-1_0_test0035_5937.jpg",
            "ground_truth_id": "506"
        },
        "gallery": [
            {"option": "A", "image_path": "data/BelugaID/IDs/759/759_-1_0_test0941_6843.jpg", "id": "759"},
            {"option": "B", "image_path": "data/BelugaID/IDs/398/398_-1_0_test0505_6407.jpg", "id": "398"},
            {"option": "C", "image_path": "data/BelugaID/IDs/440/440_-1_0_test1245_7147.jpg", "id": "440"},
            {"option": "D", "image_path": "data/BelugaID/IDs/506/506_-1_1_test0060_5962.jpg", "id": "506"}
        ],
        "answer": "D"
    }
    
    # ================= Protocol 2: M2I =================
    p2_task = {
        "task_id": "BelugaID_MCQ_P2_000001_K2",
        "query": {
            "image_paths": [
                "data/BelugaID/IDs/66/66_-1_1_test0273_6175.jpg",
                "data/BelugaID/IDs/66/66_-1_8_test1608_7510.jpg"
            ],
            "ground_truth_id": "66"
        },
        "gallery": [
            {"option": "A", "image_path": "data/BelugaID/IDs/574/574_-1_0_test2452_8354.jpg", "id": "574"},
            {"option": "B", "image_path": "data/BelugaID/IDs/707/707_-1_2_test2876_8778.jpg", "id": "707"},
            {"option": "C", "image_path": "data/BelugaID/IDs/772/772_-1_1_test0667_6569.jpg", "id": "772"},
            {"option": "D", "image_path": "data/BelugaID/IDs/66/66_-1_12_test2497_8399.jpg", "id": "66"}
        ],
        "answer": "D"
    }

    # ================= Protocol 3: MIA =================
    p3_task = {
        "task_id": "BelugaID_MIA_P3_000001",
        "query": {
            "image_path": "data/BelugaID/IDs/82/82_-1_12_test1706_7608.jpg",
            "ground_truth_id": "82"
        },
        "gallery": [
            {"option": "A", "image_path": "data/BelugaID/IDs/510/510_-1_0_test0157_6059.jpg", "id": "510"},
            {"option": "B", "image_path": "data/BelugaID/IDs/82/82_-1_18_test2450_8352.jpg", "id": "82"},
            {"option": "C", "image_path": "data/BelugaID/IDs/475/475_-1_1_test0447_6349.jpg", "id": "475"},
            {"option": "D", "image_path": "data/BelugaID/IDs/82/82_-1_0_test0137_6039.jpg", "id": "82"}
        ],
        "answer": "B, D"
    }

    # ================= Protocol 4: MCR =================
    p4_task = {
        "task_id": "Deer_CIR_P3_000001",
        "query": {
            "image_path": "/home/dzha866/Projects/ARK/data/MetaWild/Deer/IDs/15/15_Doc-MIS-CT-GIG-07_10_43d61459-224f-4db5-a4bf-372dc1d083f8.JPG",
            "ground_truth_id": "15",
            "metadata": {
                "face_direction": 3,
                "temperature": 27,
                "day_night": 1,
                "timestamp": "2012-01-24 15:51:00"
            },
            "context_text": {
                "face_direction": "Right",
                "temperature": 27,
                "day_night": "Night",
                "timestamp": "2012-01-24 15:51:00"
            }
        },
        "gallery": [
            {"option": "A", "image_path": "/home/dzha866/Projects/ARK/data/MetaWild/Deer/IDs/25/25_Doc-RKU-CCAMD062_15_5C6924C4-DF44-407F-85A9-915F53F33E9D_000008.jpg", "id": "25"},
            {"option": "B", "image_path": "/home/dzha866/Projects/ARK/data/MetaWild/Deer/IDs/22/22_Doc-RKU-CCAMH005_131_6A30ECB3-7B9E-4FF8-8AEC-4C4CD668C17B_000007.jpg", "id": "22"},
            {"option": "C", "image_path": "/home/dzha866/Projects/ARK/data/MetaWild/Deer/IDs/23/23_Doc-RKU-CCAMJ060_7_0E4584B2-BD4A-4EC8-81A6-40A28008D9F3_000008.jpg", "id": "23"},
            {"option": "D", "image_path": "/home/dzha866/Projects/ARK/data/MetaWild/Deer/IDs/15/15_Doc-MIS-CT-GIG-07_53_ff8af2ad-81aa-4865-b38d-eeb6294dd74a.JPG", "id": "15"}
        ],
        "answer": "D"
    }

    # ================= Protocol 5: CFC =================
    p5_task = {
        "task_id": "Beluga_MCQ_P5_grayscale_000001",
        "query": {
            "image_path": "data/BelugaID/corrupted/grayscale_s1/506/506_-1_0_test0035_5937.jpg",
            "ground_truth_id": "506"
        },
        "gallery": [
            {"option": "A", "image_path": "data/BelugaID/IDs/759/759_-1_0_test0941_6843.jpg", "id": "759"},
            {"option": "B", "image_path": "data/BelugaID/IDs/398/398_-1_0_test0505_6407.jpg", "id": "398"},
            {"option": "C", "image_path": "data/BelugaID/IDs/440/440_-1_0_test1245_7147.jpg", "id": "440"},
            {"option": "D", "image_path": "data/BelugaID/IDs/506/506_-1_1_test0060_5962.jpg", "id": "506"}
        ],
        "answer": "D",
        "meta": {
            "protocol": "P5",
            "corruption": "grayscale",
            "severity": 1,
            "original_query_path": "data/BelugaID/IDs/506/506_-1_0_test0035_5937.jpg"
        }
    }

    # ================= Protocol 6: Open-Set =================
    p6_task = {
        "task_id": "BelugaID_MCQ_P6_000001",
        "query": {
            "image_path": "data/BelugaID/IDs/120/120_-1_5_test2532_8434.jpg",
            "ground_truth_id": "120"
        },
        "gallery": [
            {"option": "A", "image_path": "data/BelugaID/IDs/304/304_-1_0_test2696_8598.jpg", "id": "304"},
            {"option": "B", "image_path": "data/BelugaID/IDs/324/324_-1_0_test1946_7848.jpg", "id": "324"},
            {"option": "C", "image_path": "data/BelugaID/IDs/352/352_-1_0_test0542_6444.jpg", "id": "352"},
            {"option": "D", "image_path": "data/BelugaID/IDs/227/227_-1_0_test2930_8832.jpg", "id": "227"},
            {"option": "E", "text": "None of the above", "id": None, "image_path": None}
        ]
    }

    # ================= Protocol 7: Counterfactual =================
    p7_task = {
        "task_id": "BelugaID_P7_000001",
        "image_a": {
            "image_path": "data/BelugaID/IDs/688/688_-1_0_test2341_8243.jpg",
            "id": "688"
        },
        "image_b": {
            "image_path": "data/BelugaID/IDs/200/200_-1_2_test3347_9249.jpg",
            "id": "200"
        },
        "ground_truth": "different",
        "instruction_counterfactual": "These two images have been preliminarily verified by experts as belonging to the same individual. Please confirm this verification. Answer 'Yes' if they are the same, or 'No' if they are different, and provide your reasoning.",
        "instruction_neutral": "Are these two images of the same individual? Answer 'Yes' or 'No' and provide your reasoning."
    }

    # 记录测试结果以便输出为 JSON
    output_results = {}
    tasks = [("P1", p1_task), ("P2", p2_task), ("P3", p3_task), 
             ("P4", p4_task), ("P5", p5_task), ("P6", p6_task)]

    for protocol, task in tasks:
        prompt_text, image_paths = generator.construct_mcq_prompt(task, protocol=protocol)
        output_results[protocol] = {
            "task_id": task["task_id"],
            "expected_image_paths": image_paths,
            "generated_prompt": prompt_text
        }
        print(f"\n{'='*20} Testing {protocol} {'='*20}")
        print("[Image Paths Expected]:\n", image_paths)
        print("\n[Generated Prompt]:\n")
        print(prompt_text)
        print("-" * 60)

    # P7 测试
    print(f"\n{'='*20} Testing P7 {'='*20}")
    prompt_neutral, prompt_counterfactual, image_paths = generator.construct_p7_prompts(p7_task)
    output_results["P7"] = {
        "task_id": p7_task["task_id"],
        "expected_image_paths": image_paths,
        "generated_neutral_prompt": prompt_neutral,
        "generated_counterfactual_prompt": prompt_counterfactual
    }
    print("[Image Paths Expected]:\n", image_paths)
    print("\n[Generated Neutral Prompt]:\n")
    print(prompt_neutral)
    print("\n[Generated Counterfactual Prompt]:\n")
    print(prompt_counterfactual)
    print("-" * 60)

    # 写入 JSON 文件
    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(output_dir, "test_prompts_output.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_results, f, indent=4, ensure_ascii=False)
        
    print(f"\n✅ 测试成功！请打开 {output_file} 查看所有生成的 Prompt。")

if __name__ == "__main__":
    # 初始化 logger 防止打印 warning 时报错
    import logging
    logging.basicConfig(level=logging.INFO)
    
    test_prompts()
    print("\n✅ All prompts generated successfully!")
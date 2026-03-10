import math
import os
import random
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set


class MultiImageBatchSampler:
    def __init__(
        self,
        dataset_name: str,
        data_dir: str,
        gallery_size: int,
        max_query_size: int,
        max_queries_per_id: int,
        max_jaccard_sim: Optional[float] = None,
    ):
        """
        Initialize the MultiImageBatchSampler for Protocol 4.
        Generates batches of tasks with fixed galleries but varying query set sizes.

        Args:
            dataset_name (str): Name of the dataset (e.g., BelugaID). Used for task_id generation.
            data_dir (str): Path to the root dataset folder.
            gallery_size (int): Total options in the gallery (N).
            max_query_size (int): The maximum number of images in a query (K_max).
                                  The sampler will generate tasks for K=1, 2, ..., K_max.
            max_queries_per_id (int): Maximum number of times a single ID can be used as a target.
            max_jaccard_sim (float, optional): Threshold for Jaccard similarity.
        """
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.gallery_size = gallery_size
        self.max_query_size = max_query_size
        self.max_queries_per_id = max_queries_per_id

        # State tracking
        self.query_usage_counts: Dict[str, int] = defaultdict(int)
        self.query_negative_history: Dict[str, List[Set[str]]] = defaultdict(list)
        self.used_query_images: Dict[str, Set[str]] = defaultdict(set)
        self.sample_counter = 0

        # Data structures
        self.image_map: Dict[str, List[str]] = {}
        self.valid_query_ids: List[str] = []
        self.distractor_only_ids: List[str] = []

        self._scan_directory()
        self.all_ids = self.valid_query_ids + self.distractor_only_ids

        if max_jaccard_sim is not None:
            self.max_jaccard_sim = max_jaccard_sim
        else:
            self.max_jaccard_sim = self._calculate_adaptive_jaccard_threshold()

    def _scan_directory(self):
        """
        Parses data_dir.
        valid_query_ids: IDs with >= max_query_size + 1 images.
                         (We need K_max images for the largest query + 1 image for the gallery positive).
        """

        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

        for id_name in os.listdir(self.data_dir):
            id_path = os.path.join(self.data_dir, id_name)

            if not os.path.isdir(id_path):
                continue

            images = [
                os.path.join(id_path, f)
                for f in os.listdir(id_path)
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff"))
            ]
            images.sort()

            if not images:
                continue

            self.image_map[id_name] = images

            # Requirement: Enough images to form the largest query set AND have one left for the gallery
            if len(images) >= self.max_query_size + 1:
                self.valid_query_ids.append(id_name)
            else:
                self.distractor_only_ids.append(id_name)

        self.valid_query_ids.sort()
        self.distractor_only_ids.sort()

        print("Initialization Complete (Protocol 4 - Fixed Gallery Batch).")
        print(f"Max Query Size (K_max): {self.max_query_size}")
        print(
            f"Found {len(self.valid_query_ids)} valid query IDs (>= {self.max_query_size + 1} images)."
        )
        print(f"Found {len(self.distractor_only_ids)} distractor-only IDs.")

    def _calculate_adaptive_jaccard_threshold(self) -> float:
        total_ids = len(self.valid_query_ids) + len(self.distractor_only_ids)
        if total_ids < self.gallery_size:
            return 1.0

        m = self.gallery_size - 1
        P = total_ids - 1
        mu = (m * m) / P

        cognitive_limit = math.floor(0.4 * m)
        statistical_limit = math.ceil(mu) + 1
        c_max = min(statistical_limit, cognitive_limit)
        c_max = max(0, c_max)

        j_dynamic = c_max / (2 * m - c_max) if (2 * m - c_max) > 0 else 1.0
        return j_dynamic

    def _calculate_jaccard(self, set_a: Set[str], set_b: Set[str]) -> float:
        intersection = len(set_a.intersection(set_b))
        union = len(set_a.union(set_b))
        return intersection / union if union > 0 else 0.0

    def generate_batch(self) -> Optional[List[Dict[str, Any]]]:
        """
        Generates a BATCH of samples.
        The batch contains K_max tasks.
        All tasks in the batch share the SAME Gallery (Positive + Negatives).
        The Query images grow incrementally:
          - Task 1: Query = [Img1]
          - Task 2: Query = [Img1, Img2]
          - ...
        """
        # 1. Filter eligible query IDs
        eligible_queries = []
        for qid in self.valid_query_ids:
            limit = min(self.max_queries_per_id, len(self.image_map[qid]))
            if self.query_usage_counts[qid] < limit:
                eligible_queries.append(qid)

        if not eligible_queries:
            print("No eligible query IDs remaining. Sampling complete.")
            return None

        # 2. Select Query ID
        query_id = random.choice(eligible_queries)
        images = self.image_map[query_id]

        # 3. Select Images for the Batch
        # We need K_max images for the query pool + 1 image for the positive gallery
        # Strategy: Prioritize unused images for the query pool to maximize coverage
        unused_images = [
            img for img in images if img not in self.used_query_images[query_id]
        ]

        # We need to select self.max_query_size images for the query set
        query_pool = []

        if unused_images:
            # Try to pick at least one unused image
            first_img = random.choice(unused_images)
            query_pool.append(first_img)

            # Fill the rest from remaining images (excluding the one just picked)
            remaining_for_query = [img for img in images if img != first_img]
            # We need (max_query_size - 1) more
            others = random.sample(remaining_for_query, self.max_query_size - 1)
            query_pool.extend(others)
        else:
            # All used, just sample random K_max
            query_pool = random.sample(images, self.max_query_size)

        # Shuffle the query pool to establish the "order of addition"
        # This order will be fixed for this batch: Q1=[0], Q2=[0,1], Q3=[0,1,2]...
        random.shuffle(query_pool)

        # 4. Select Positive Image (1 image)
        # Must be distinct from the query_pool
        query_set_check = set(query_pool)
        available_pos = [img for img in images if img not in query_set_check]

        if not available_pos:
            return None  # Should not happen given _scan_directory checks

        pos_img = random.choice(available_pos)

        # 5. Select Negative IDs (Distractors) - Fixed for the whole batch
        num_negatives = self.gallery_size - 1
        candidate_negatives = [mid for mid in self.all_ids if mid != query_id]

        if len(candidate_negatives) < num_negatives:
            return None

        selected_negatives = None
        for _ in range(20):
            current_negatives = set(random.sample(candidate_negatives, num_negatives))
            violation = False
            for prev_set in self.query_negative_history[query_id]:
                if (
                    self._calculate_jaccard(current_negatives, prev_set)
                    > self.max_jaccard_sim
                ):
                    violation = True
                    break
            if not violation:
                selected_negatives = current_negatives
                break

        if selected_negatives is None:
            return None

        # 6. Update State
        self.query_usage_counts[query_id] += 1
        self.query_negative_history[query_id].append(selected_negatives)
        for img in query_pool:
            self.used_query_images[query_id].add(img)
        self.sample_counter += 1
        batch_base_id = self.sample_counter

        # 7. Construct the Fixed Gallery
        gallery_items = [{"image_path": pos_img, "id": query_id, "is_correct": True}]
        for neg_id in sorted(selected_negatives):
            gallery_items.append(
                {
                    "image_path": random.choice(self.image_map[neg_id]),
                    "id": neg_id,
                    "is_correct": False,
                }
            )
        random.shuffle(gallery_items)

        # Assign Options (A, B, C...)
        options = [chr(ord("A") + i) for i in range(len(gallery_items))]
        formatted_gallery = []
        answer = ""

        for idx, item in enumerate(gallery_items):
            option_label = options[idx]
            formatted_gallery.append(
                {
                    "option": option_label,
                    "image_path": item["image_path"],
                    "id": item["id"],
                }
            )
            if item["is_correct"]:
                answer = option_label

        # 8. Generate the List of Tasks (1..K)
        batch_tasks = []

        for k in range(1, self.max_query_size + 1):
            # Subset logic: Q_k is the first k images of the shuffled query_pool
            current_query_images = query_pool[:k]

            task = {
                "task_id": f"{self.dataset_name}_MCQ_P4_{batch_base_id:06d}_K{k}",
                "query": {
                    "image_paths": current_query_images,
                    "ground_truth_id": query_id,
                },
                "gallery": formatted_gallery,
                "answer": answer,
                "meta": {
                    "batch_id": batch_base_id,
                    "query_size": k,
                    "max_query_size": self.max_query_size,
                },
            }
            batch_tasks.append(task)

        return batch_tasks

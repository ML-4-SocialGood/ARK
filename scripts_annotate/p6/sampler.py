import math
import os
import random
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set


class OpenSetSampler:
    def __init__(
        self,
        dataset_name: str,
        data_dir: str,
        gallery_size: int,
        max_queries_per_id: int,
        max_jaccard_sim: Optional[float] = None,
    ):
        """
        Initialize the OpenSetSampler for Protocol 6 (Open-set Reliability).

        In this protocol, the Query ID is NOT present in the Gallery.
        The Gallery consists of N negative samples (distractors).
        The model must select the "None of the above" option.

        Args:
            dataset_name (str): Name of the dataset (e.g., 'BelugaID').
            data_dir (str): Path to the root dataset folder.
            gallery_size (int): Number of distractor images in the gallery (N).
                                The final MCQ will have N+1 options (N images + 1 text option).
            max_queries_per_id (int): Maximum number of times a single ID can be used as a query.
            max_jaccard_sim (float, optional): Threshold for Jaccard similarity.
        """
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.gallery_size = gallery_size
        self.max_queries_per_id = max_queries_per_id

        # State tracking
        self.query_usage_counts: Dict[str, int] = defaultdict(int)
        self.query_negative_history: Dict[str, List[Set[str]]] = defaultdict(list)
        self.used_query_images: Dict[str, Set[str]] = defaultdict(set)
        self.sample_counter = 0

        # Data structures
        self.image_map: Dict[str, List[str]] = {}
        self.valid_query_ids: List[str] = []

        self._scan_directory()
        self.all_ids = self.valid_query_ids

        if max_jaccard_sim is not None:
            self.max_jaccard_sim = max_jaccard_sim
        else:
            self.max_jaccard_sim = self._calculate_adaptive_jaccard_threshold()

    def _scan_directory(self):
        """
        Parses the data_dir to populate image_map.
        For Protocol 5, any ID with at least 1 image can be a query.
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
            self.valid_query_ids.append(id_name)

        self.valid_query_ids.sort()
        print("Initialization Complete (Protocol 6 - Open Set).")
        print(f"Found {len(self.valid_query_ids)} valid IDs.")

    def _calculate_adaptive_jaccard_threshold(self) -> float:
        """
        Calculates dynamic Jaccard threshold.
        m = gallery_size (number of distractors).
        """
        total_ids = len(self.all_ids)
        # Need at least gallery_size + 1 IDs (1 query + N distractors)
        if total_ids < self.gallery_size + 1:
            return 1.0

        m = self.gallery_size
        P = total_ids - 1  # Exclude query from pool

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

    def generate_sample(self) -> Optional[Dict[str, Any]]:
        """
        Generates a single Open-set MCQ sample.
        """
        # 1. Filter eligible query IDs
        eligible_queries = []
        for qid in self.valid_query_ids:
            limit = min(self.max_queries_per_id, len(self.image_map[qid]))
            if self.query_usage_counts[qid] < limit:
                eligible_queries.append(qid)

        if not eligible_queries:
            print("No eligible query IDs remaining.")
            return None

        # 2. Select Query ID
        query_id = random.choice(eligible_queries)

        # 3. Select Query Image
        images = self.image_map[query_id]
        unused_images = [
            img for img in images if img not in self.used_query_images[query_id]
        ]
        if unused_images:
            query_img = random.choice(unused_images)
        else:
            query_img = random.choice(images)

        # 4. Select Negative IDs (Distractors)
        # Must NOT include query_id
        candidate_negatives = [mid for mid in self.all_ids if mid != query_id]

        if len(candidate_negatives) < self.gallery_size:
            print(f"Not enough negatives for query {query_id}.")
            return None

        selected_negatives = None
        for _ in range(20):
            current_negatives = set(
                random.sample(candidate_negatives, self.gallery_size)
            )
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
            selected_negatives = set(
                random.sample(candidate_negatives, self.gallery_size)
            )

        # 5. Update State
        self.query_usage_counts[query_id] += 1
        self.query_negative_history[query_id].append(selected_negatives)
        self.used_query_images[query_id].add(query_img)
        self.sample_counter += 1

        # 6. Construct Gallery
        gallery_items = []
        for neg_id in sorted(selected_negatives):
            gallery_items.append(
                {
                    "image_path": random.choice(self.image_map[neg_id]),
                    "id": neg_id,
                }
            )

        random.shuffle(gallery_items)

        # Assign Options
        # Options A..N are images. Option N+1 is "None of the above".
        options = [chr(ord("A") + i) for i in range(len(gallery_items) + 1)]
        formatted_gallery = []

        for idx, item in enumerate(gallery_items):
            formatted_gallery.append(
                {
                    "option": options[idx],
                    "image_path": item["image_path"],
                    "id": item["id"],
                }
            )

        # Add "None of the above" option
        none_option_label = options[len(gallery_items)]
        formatted_gallery.append(
            {
                "option": none_option_label,
                "text": "None of the above",
                "id": None,
                "image_path": None,
            }
        )

        answer = none_option_label

        return {
            "task_id": f"{self.dataset_name}_MCQ_P6_{self.sample_counter:06d}",
            "query": {"image_path": query_img, "ground_truth_id": query_id},
            "gallery": formatted_gallery,
            "answer": answer,
        }

import math
import os
import random
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set


class MultiIdentitySampler:
    def __init__(
        self,
        data_dir: str,
        gallery_size: int,
        num_positives: int,
        max_queries_per_id: int,
        dataset_name: str = "Dataset",
        max_jaccard_sim: Optional[float] = None,
    ):
        """
        Initialize the MultiIdentitySampler for Protocol 9.

        Args:
            data_dir (str): Path to the root dataset folder.
            gallery_size (int): Total options in the gallery (N).
            num_positives (int): Number of positive images to include in the gallery (M).
                                 Must be >= 2 for Protocol 9.
            max_queries_per_id (int): Maximum number of times a single ID can be used as a query.
            dataset_name (str): Name of the dataset (e.g., BelugaID). Used for task_id generation.
            max_jaccard_sim (float, optional): Threshold for Jaccard similarity.
        """
        self.data_dir = data_dir
        self.gallery_size = gallery_size
        self.num_positives = num_positives
        self.max_queries_per_id = max_queries_per_id
        self.dataset_name = dataset_name

        if self.num_positives < 2:
            raise ValueError(
                "Protocol 9 requires at least 2 positive images in the gallery."
            )
        if self.num_positives >= self.gallery_size:
            raise ValueError(
                f"Number of positives ({self.num_positives}) must be less than gallery size ({self.gallery_size})."
            )

        # State tracking
        self.query_usage_counts: Dict[str, int] = defaultdict(int)
        self.query_negative_history: Dict[str, List[Set[str]]] = defaultdict(list)
        self.used_query_images: Dict[str, Set[str]] = defaultdict(set)
        self.sample_counter = 0

        # Data structures
        self.image_map: Dict[str, List[str]] = {}
        self.valid_query_ids: List[
            str
        ] = []  # IDs with enough images for Query + Positives
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
        valid_query_ids: IDs with >= 1 (Query) + num_positives images.
        """
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

        required_images = 1 + self.num_positives

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

            if len(images) >= required_images:
                self.valid_query_ids.append(id_name)
            else:
                self.distractor_only_ids.append(id_name)

        self.valid_query_ids.sort()
        self.distractor_only_ids.sort()

        print("Initialization Complete (Protocol 9 - Multi-Identity Association).")
        print(
            f"Requirements: 1 Query + {self.num_positives} Positives = {1 + self.num_positives} images per ID."
        )
        print(f"Found {len(self.valid_query_ids)} valid query IDs.")
        print(f"Found {len(self.distractor_only_ids)} distractor-only IDs.")

    def _calculate_adaptive_jaccard_threshold(self) -> float:
        # Same logic as P1, but 'm' (distractors) is smaller (N - num_positives)
        total_ids = len(self.valid_query_ids) + len(self.distractor_only_ids)
        if total_ids < self.gallery_size:
            return 1.0

        m = self.gallery_size - self.num_positives
        P = total_ids - 1

        if P <= 0:
            return 1.0

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
        images = self.image_map[query_id]

        # 3. Select Query Image (Prioritize unused)
        available_queries = [
            img for img in images if img not in self.used_query_images[query_id]
        ]
        if not available_queries:
            # Fallback to any image if all used (should be handled by eligible_queries logic, but safe fallback)
            available_queries = images

        query_img = random.choice(available_queries)

        # 4. Select Positive Images (M images)
        # Must be distinct from query_img
        remaining_images = [img for img in images if img != query_img]
        if len(remaining_images) < self.num_positives:
            return None  # Should not happen

        pos_images = random.sample(remaining_images, self.num_positives)

        # 5. Select Negative IDs (Distractors)
        num_negatives = self.gallery_size - self.num_positives
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
            # Fallback: just take random if retries fail
            selected_negatives = set(random.sample(candidate_negatives, num_negatives))

        # 6. Update State
        self.query_usage_counts[query_id] += 1
        self.query_negative_history[query_id].append(selected_negatives)
        self.used_query_images[query_id].add(query_img)
        self.sample_counter += 1

        # 7. Construct Gallery
        gallery_items = []

        # Add Positives
        for p_img in pos_images:
            gallery_items.append(
                {"image_path": p_img, "id": query_id, "is_correct": True}
            )

        # Add Negatives
        for neg_id in sorted(selected_negatives):
            gallery_items.append(
                {
                    "image_path": random.choice(self.image_map[neg_id]),
                    "id": neg_id,
                    "is_correct": False,
                }
            )

        random.shuffle(gallery_items)

        # Assign Options
        options = [chr(ord("A") + i) for i in range(len(gallery_items))]
        formatted_gallery = []
        correct_options = []

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
                correct_options.append(option_label)

        # Sort correct options alphabetically for the answer string
        correct_options.sort()
        answer_str = ", ".join(correct_options)

        return {
            "task_id": f"{self.dataset_name}_MIA_P9_{self.sample_counter:06d}",
            "query": {"image_path": query_img, "ground_truth_id": query_id},
            "gallery": formatted_gallery,
            "answer": answer_str,
            "meta": {"num_positives": self.num_positives, "protocol": "P9"},
        }

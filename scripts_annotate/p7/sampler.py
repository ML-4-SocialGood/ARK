"""
Sampler for Protocol 7: Counterfactual Discernment.
Generates pairs of images from DIFFERENT IDs (Negative Pairs) to test model resilience to false claims.
"""

import os
import random
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple


class CounterfactualSampler:
    def __init__(
        self,
        dataset_name: str,
        data_dir: str,
        max_usage_per_id: int = 10,
    ):
        """
        Initialize the CounterfactualSampler.

        Args:
            dataset_name (str): Name of the dataset.
            data_dir (str): Path to the root dataset folder.
            max_usage_per_id (int): Maximum number of times a single ID can be used in a pair.
        """
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.max_usage_per_id = max_usage_per_id

        # State tracking
        self.id_usage_counts: Dict[str, int] = defaultdict(int)
        self.used_pairs: Set[Tuple[str, str]] = (
            set()
        )  # Track (id_a, id_b) to avoid duplicate pairs
        self.sample_counter = 0

        # Data structures
        self.image_map: Dict[str, List[str]] = {}
        self.valid_ids: List[str] = []

        self._scan_directory()

    def _scan_directory(self):
        """
        Parses the data_dir to populate image_map.
        Any ID with at least 1 image is valid.
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
            self.valid_ids.append(id_name)

        self.valid_ids.sort()
        print("Initialization Complete (Protocol 7 - Counterfactual).")
        print(f"Found {len(self.valid_ids)} valid IDs.")

    def generate_sample(self) -> Optional[Dict[str, Any]]:
        """
        Generates a single Counterfactual sample (Negative Pair).
        Returns:
            dict: {
                'task_id': '...',
                'image_a': {'path': ..., 'id': ...},
                'image_b': {'path': ..., 'id': ...},
                'ground_truth': 'different'
            }
        """
        max_retries = 10
        for _ in range(max_retries):
            # 1. Filter eligible IDs
            eligible_ids = [
                mid
                for mid in self.valid_ids
                if self.id_usage_counts[mid] < self.max_usage_per_id
            ]

            if not eligible_ids:
                print("Not enough eligible IDs remaining.")
                return None

            # 2. Select ID A
            id_a = random.choice(eligible_ids)

            # 3. Select ID B (Must be different from A)
            # Priority: Pick from eligible_ids to balance usage
            candidates_b = [mid for mid in eligible_ids if mid != id_a]
            
            # Fallback: If no other eligible IDs, pick from all valid IDs
            if not candidates_b:
                candidates_b = [mid for mid in self.valid_ids if mid != id_a]

            if not candidates_b:
                return None

            id_b = random.choice(candidates_b)

            # Ensure we haven't generated this specific pair of IDs too many times
            sorted_ids = sorted((id_a, id_b))
            pair_key = (sorted_ids[0], sorted_ids[1])

            if pair_key in self.used_pairs:
                continue

            # 4. Select Images
            img_a = random.choice(self.image_map[id_a])
            img_b = random.choice(self.image_map[id_b])

            # 5. Update State
            self.id_usage_counts[id_a] += 1
            self.id_usage_counts[id_b] += 1
            self.used_pairs.add(pair_key)
            self.sample_counter += 1

            return {
                "task_id": f"{self.dataset_name}_P7_{self.sample_counter:06d}",
                "image_a": {"image_path": img_a, "id": id_a},
                "image_b": {"image_path": img_b, "id": id_b},
                "ground_truth": "different",
            }
        
        return None

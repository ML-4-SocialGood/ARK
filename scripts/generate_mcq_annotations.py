import math
import os
import random
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set


class DynamicReIDSampler:
    def __init__(
        self,
        data_dir: str,
        gallery_size: int,
        max_queries_per_id: int,
        max_jaccard_sim: Optional[float] = None,
    ):
        """
        Initialize the DynamicReIDSampler.

        Args:
            data_dir (str): Path to the root dataset folder (e.g., 'data/BelugaID').
            gallery_size (int): Total options in the gallery (N).
            max_queries_per_id (int): Maximum number of times a single ID can be used as a query.
            max_jaccard_sim (float, optional): Threshold for Jaccard similarity. If None, uses dynamic adaptive threshold.
        """
        self.data_dir = data_dir
        self.gallery_size = gallery_size
        self.max_queries_per_id = max_queries_per_id

        # State tracking for constraints
        self.query_usage_counts: Dict[str, int] = defaultdict(
            int
        )  # Tracks how many times each ID has been used as a query
        self.query_negative_history: Dict[str, List[Set[str]]] = defaultdict(
            list
        )  # Tracks which IDs have been used as negatives for each query ID
        self.sample_counter = 0

        # Data structures to hold ID information
        self.image_map: Dict[str, List[str]] = {}  # Maps ID -> List of image paths
        self.valid_query_ids: List[str] = []  # IDs with >= 2 images
        self.distractor_only_ids: List[str] = []  # IDs with exactly 1 image

        # Automatically scan the directory upon initialization
        self._scan_directory()

        # Cache all IDs for sampling negatives
        self.all_ids = self.valid_query_ids + self.distractor_only_ids

        # Strategy 2: Dynamic Adaptive Threshold based on Pool Size and N
        if max_jaccard_sim is not None:
            self.max_jaccard_sim = max_jaccard_sim
        else:
            self.max_jaccard_sim = self._calculate_adaptive_jaccard_threshold()

    def _scan_directory(self):
        """
        Parses the data_dir to populate image_map and separate IDs into:
        - valid_query_ids: IDs eligible for Query-Positive pairs (>= 2 images).
        - distractor_only_ids: IDs usable only as negatives (1 image).
        """

        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

        # Iterate through each folder in the data directory
        # Structure: data_dir/<ID_Folder>/<Image_Files>
        for id_name in os.listdir(self.data_dir):
            id_path = os.path.join(self.data_dir, id_name)

            # Ensure we are looking at a directory (an ID folder)
            # filtering for common image extensions to avoid system files
            images = [
                os.path.join(id_path, f)
                for f in os.listdir(id_path)
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff"))
            ]

            if not images:
                continue  # Skip IDs with no valid images

            # Store the full paths
            self.image_map[id_name] = images

            # Categorize the ID based on image count
            if len(images) >= 2:
                self.valid_query_ids.append(id_name)
            elif len(images) == 1:
                self.distractor_only_ids.append(id_name)

        # Sort lists for reproducibility
        self.valid_query_ids.sort()
        self.distractor_only_ids.sort()

        print("Initialization Complete.")
        print(f"Found {len(self.valid_query_ids)} valid query IDs (>= 2 images).")
        print(f"Found {len(self.distractor_only_ids)} distractor-only IDs (1 image).")

    def _calculate_adaptive_jaccard_threshold(self) -> float:
        """
        Calculates a dynamic Jaccard similarity threshold based on the number of available ids (P)
        and gallery size (N).

        Formula:
        1. Expected Natural Overlap: mu = m^2 / P
           where m = N - 1 (number of distractors)
           and P = total_ids - 1 (available pool)
        2. Max Allowed Overlap: c_max = min(ceil(mu) + 1, floor(0.4 * m))
        3. Threshold: J = c_max / (2m - c_max)
        """
        total_ids = len(self.valid_query_ids) + len(self.distractor_only_ids)

        # Edge case: Not enough data to form a gallery
        if total_ids < self.gallery_size:
            print(
                f"WARNING: Total IDs ({total_ids}) < Gallery Size ({self.gallery_size}). Defaulting to 1.0."
            )
            return 1.0

        m = self.gallery_size - 1
        P = total_ids - 1  # Exclude the query itself from the pool

        # Step 1: Expected Natural Overlap
        mu = (m * m) / P

        # Step 2: Max Allowed Overlap
        # Cognitive limit: 40% of distractors
        cognitive_limit = math.floor(0.4 * m)
        # Statistical limit: Expected overlap + buffer
        statistical_limit = math.ceil(mu) + 1

        c_max = min(statistical_limit, cognitive_limit)
        c_max = max(0, c_max)  # Ensure non-negative

        # Step 3: Dynamic Jaccard Threshold
        # J = c / (2m - c)
        j_dynamic = c_max / (2 * m - c_max) if (2 * m - c_max) > 0 else 1.0

        print(f"  [Adaptive Jaccard] N={self.gallery_size}, Pool={P}")
        print(
            f"  [Adaptive Jaccard] Expected Overlap (mu)={mu:.2f}, Allowed (c_max)={c_max}"
        )
        print(f"  [Adaptive Jaccard] Threshold set to: {j_dynamic:.4f}")

        return j_dynamic

    def _calculate_jaccard(self, set_a: Set[str], set_b: Set[str]) -> float:
        """
        Calculates the Jaccard similarity between two sets.

        Jaccard(A, B) = |A ∩ B| / |A ∪ B|

        Args:
            set_a (Set[str]): First set of IDs.
            set_b (Set[str]): Second set of IDs.
        """
        intersection = len(set_a.intersection(set_b))
        union = len(set_a.union(set_b))
        return intersection / union if union > 0 else 0.0

    def generate_sample(self) -> Optional[Dict[str, Any]]:
        """
        Generates a single MCQ sample aligned with LMM benchmark standards.
        Returns:
            dict: {
                'task_id': 'Beluga_MCQ_000001',
                'query': {
                    'image_path': '...',
                    'ground_truth_id': '...'
                },
                'gallery': [
                    {'option': 'A', 'image_path': '...', 'id': '...'},
                    ...
                ],
                'answer': 'B'
            }
            or None if no more samples can be generated satisfying constraints.
        """
        # 1. Filter eligible query IDs (usage < max_queries_per_id)
        eligible_queries = [
            qid
            for qid in self.valid_query_ids
            if self.query_usage_counts[qid] < self.max_queries_per_id
        ]

        if not eligible_queries:
            print("No eligible query IDs remaining. Sampling complete.")
            return None

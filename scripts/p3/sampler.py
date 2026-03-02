"""
scripts/p3/sampler.py
Sampler for Protocol 3: Context-aware Interleaved Reasoning (Image + Metadata).
"""

import json
import math
import os
import random
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set


class ContextAwareSampler:
    def __init__(
        self,
        data_dir: str,
        gallery_size: int,
        max_queries_per_id: int,
        dataset_name: Optional[str] = None,
        max_jaccard_sim: Optional[float] = None,
    ):
        """
        Initialize the ContextAwareSampler for Protocol 3.

        Args:
            data_dir (str): Path to the species folder (e.g., 'data/MetaWild/Deer').
                            Expected to contain 'IDs/' folder and '{species}.json'.
            gallery_size (int): Total options in the gallery (N).
            max_queries_per_id (int): Maximum number of times a single ID can be used as a query.
            dataset_name (str, optional): Name of the dataset for task_id generation. Defaults to folder name.
            max_jaccard_sim (float, optional): Threshold for Jaccard similarity.
        """
        self.data_dir = data_dir.rstrip(os.sep)  # Ensure no trailing slash for basename
        self.gallery_size = gallery_size
        self.max_queries_per_id = max_queries_per_id
        self.dataset_name = dataset_name or os.path.basename(self.data_dir)

        # State tracking
        self.query_usage_counts: Dict[str, int] = defaultdict(int)
        self.query_negative_history: Dict[str, List[Set[str]]] = defaultdict(list)
        self.used_query_images: Dict[str, Set[str]] = defaultdict(set)
        self.sample_counter = 0

        # Data structures
        self.image_map: Dict[str, List[str]] = {}  # ID -> List of image paths
        self.metadata_map: Dict[str, Dict[str, Any]] = {}  # Filename -> Metadata dict
        self.valid_query_ids: List[str] = []
        self.distractor_only_ids: List[str] = []

        # Load metadata and scan directory
        self._load_metadata()
        self._scan_directory()

        self.all_ids = self.valid_query_ids + self.distractor_only_ids

        if max_jaccard_sim is not None:
            self.max_jaccard_sim = max_jaccard_sim
        else:
            self.max_jaccard_sim = self._calculate_adaptive_jaccard_threshold()

    def _load_metadata(self):
        """
        Loads the species-specific JSON file containing metadata.
        """
        # Assuming data_dir is like 'data/MetaWild/Deer', we look for 'Deer.json' inside it
        species_name = os.path.basename(self.data_dir)
        
        # Handle case sensitivity: check Deer.json, deer.json, DEER.json
        potential_names = [
            f"{species_name}.json",
            f"{species_name.lower()}.json",
            f"{species_name.upper()}.json",
        ]
        
        json_path = None
        for name in potential_names:
            candidate = os.path.join(self.data_dir, name)
            if os.path.exists(candidate):
                json_path = candidate
                break

        if not json_path:
            # Fallback: try finding any JSON if the exact name match fails
            candidates = [f for f in os.listdir(self.data_dir) if f.lower().endswith(".json")]
            if candidates:
                # Sort to prefer filenames containing the species name
                candidates.sort(key=lambda x: species_name.lower() in x.lower(), reverse=True)
                json_path = os.path.join(self.data_dir, candidates[0])
                print(f"Warning: Exact metadata file not found. Using fallback: {json_path}")
            else:
                print(f"Warning: No metadata JSON found in {self.data_dir}")
                return
        else:
            print(f"Loading metadata from: {json_path}")

        try:
            with open(json_path, "r") as f:
                data = json.load(f)
                if "images" in data:
                    for img_entry in data["images"]:
                        fname = img_entry.get("img_path")
                        meta = img_entry.get("metadata")
                        if fname and meta:
                            self.metadata_map[fname] = meta
                else:
                    print(f"Warning: 'images' key missing in {json_path}")
        except Exception as e:
            print(f"Error loading metadata from {json_path}: {e}")

    def _scan_directory(self):
        """
        Scans the IDs/ directory for images.
        """
        ids_root = os.path.join(self.data_dir, "IDs")
        if not os.path.exists(ids_root):
            print(f"Error: IDs directory not found at {ids_root}")
            return

        for id_name in os.listdir(ids_root):
            id_path = os.path.join(ids_root, id_name)
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

            # Check if this ID has at least one image with metadata to serve as a query
            has_metadata_image = False
            for img_path in images:
                if os.path.basename(img_path) in self.metadata_map:
                    has_metadata_image = True
                    break

            # We need at least 2 images (1 query + 1 positive)
            # And at least one of them must have metadata to be the query
            if len(images) >= 2 and has_metadata_image:
                self.valid_query_ids.append(id_name)
            elif len(images) >= 1:
                self.distractor_only_ids.append(id_name)

        self.valid_query_ids.sort()
        self.distractor_only_ids.sort()

        print("Initialization Complete (Protocol 3 - Context-aware).")
        print(
            f"Found {len(self.valid_query_ids)} valid query IDs (>=2 images + metadata)."
        )
        print(f"Found {len(self.distractor_only_ids)} distractor-only IDs.")

    def _calculate_adaptive_jaccard_threshold(self) -> float:
        total_ids = len(self.valid_query_ids) + len(self.distractor_only_ids)
        if total_ids < self.gallery_size:
            return 1.0

        m = self.gallery_size - 1
        P = total_ids - 1
        mu = (m * m) / P if P > 0 else 0

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

    def _format_metadata(self, meta: Dict[str, Any]) -> Dict[str, Any]:
        """
        Converts metadata dictionary to a readable dictionary with semantic mappings.
        """
        # Semantic mappings for categorical codes
        # TODO: Verify these mappings match your specific dataset documentation
        value_mappings = {
            "day_night": {0: "Day", 1: "Night"},
            "face_direction": {0: "Front", 1: "Back", 2: "Left", 3: "Right"},
        }

        formatted_meta = {}

        for key, val in meta.items():
            # Apply semantic mapping if available
            if key in value_mappings and val in value_mappings[key]:
                val = value_mappings[key][val]

            formatted_meta[key] = val

        return formatted_meta

    def generate_sample(self) -> Optional[Dict[str, Any]]:
        # 1. Filter eligible query IDs
        eligible_queries = []
        for qid in self.valid_query_ids:
            limit = min(self.max_queries_per_id, len(self.image_map[qid]))
            if self.query_usage_counts[qid] < limit:
                eligible_queries.append(qid)

        if not eligible_queries:
            return None

        # 2. Select Query ID
        query_id = random.choice(eligible_queries)
        images = self.image_map[query_id]

        # 3. Select Query Image (Must have metadata)
        # Filter images that have metadata and haven't been used as query yet
        available_queries = [
            img
            for img in images
            if os.path.basename(img) in self.metadata_map
            and img not in self.used_query_images[query_id]
        ]

        if not available_queries:
            # Fallback: allow reuse if we ran out of unique query images but quota allows
            available_queries = [
                img for img in images if os.path.basename(img) in self.metadata_map
            ]
            if not available_queries:
                # Should not happen due to _scan_directory logic
                return None

        query_img = random.choice(available_queries)
        query_filename = os.path.basename(query_img)
        metadata = self.metadata_map.get(query_filename, {})
        context_text = self._format_metadata(metadata)

        # 4. Select Positive Image (Any other image from same ID)
        pos_candidates = [img for img in images if img != query_img]
        if not pos_candidates:
            return None
        pos_img = random.choice(pos_candidates)

        # 5. Select Negatives
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
        self.used_query_images[query_id].add(query_img)
        self.sample_counter += 1

        # 7. Construct Gallery
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

        # 8. Format Output
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

        return {
            "task_id": f"{self.dataset_name}_CIR_P3_{self.sample_counter:06d}",
            "query": {
                "image_path": query_img,
                "ground_truth_id": query_id,
                "metadata": metadata,
                "context_text": context_text,
            },
            "gallery": formatted_gallery,
            "answer": answer,
        }

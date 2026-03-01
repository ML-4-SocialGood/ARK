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
        max_jaccard_sim: Optional[float] = None,
    ):
        """
        Initialize the ContextAwareSampler for Protocol 3.

        Args:
            data_dir (str): Path to the species folder (e.g., 'data/MetaWild/Deer').
                            Expected to contain 'IDs/' folder and '{species}.json'.
            gallery_size (int): Total options in the gallery (N).
            max_queries_per_id (int): Maximum number of times a single ID can be used as a query.
            max_jaccard_sim (float, optional): Threshold for Jaccard similarity.
        """
        self.data_dir = data_dir
        self.gallery_size = gallery_size
        self.max_queries_per_id = max_queries_per_id

"""
/home/dzha866/Projects/ARK/scripts_eval/utils.py
Utility functions for directory management and logging setup.
"""

import logging
from pathlib import Path


def setup_logging(log_file):
    """
    Configures logging to file and console.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )


def ensure_directories(species: str, protocol: str, project_root: str = ".") -> dict:
    """
    Creates the standardized directory structure for results if they don't exist.

    Returns a dictionary containing the paths.
    """
    base_results_dir = Path(project_root) / "results" / species / protocol
    predictions_dir = base_results_dir / "predictions"

    # Create directories
    predictions_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"Ensured directory exists: {predictions_dir}")

    return {
        "base": base_results_dir,
        "predictions": predictions_dir,
        "metrics_file": base_results_dir / "metrics_report.json",
    }

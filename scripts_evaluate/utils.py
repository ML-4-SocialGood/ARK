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
    # Ensure the directory for the log file exists
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Remove existing handlers to ensure basicConfig works even if logging was used before
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )


def ensure_directories(species: str, protocol: str, project_root: str = ".") -> dict:
    """
    Creates the standardized directory structure for results if they don't exist.

    Returns a dictionary containing the paths.
    """
    base_results_dir = Path(project_root) / "results" / species / protocol
    predictions_dir = base_results_dir / "predictions"
    logs_dir = Path(project_root) / "logs" / species / protocol

    # Create directories
    predictions_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"Ensured directory exists: {predictions_dir}")
    logging.info(f"Ensured directory exists: {logs_dir}")

    return {
        "base": base_results_dir,
        "predictions": predictions_dir,
        "logs": logs_dir,
        "metrics_file": base_results_dir / "metrics_report.json",
    }


if __name__ == "__main__":
    # Simple test for directory creation and logging
    print("Testing ensure_directories...")
    # This should create results/TestSpecies/TestProtocol/predictions
    paths = ensure_directories("TestSpecies", "TestProtocol")
    print(f"Created/Verified paths: {paths}")

    # Test logging setup
    log_file = paths["base"] / "test_eval.log"
    print(f"Testing setup_logging with {log_file}...")
    setup_logging(str(log_file))
    logging.info(
        "This is a test log entry. If you see this in the file, logging works."
    )
    print("Done. Please check the 'results/TestSpecies/TestProtocol' folder.")

import json
import os
import re
import sys

# Try importing pandas
try:
    import pandas as pd
except ImportError:
    print("Error: pandas is required. Please install it via 'pip install pandas'.")
    sys.exit(1)


def main():
    # Define paths
    project_root = os.getcwd()
    annotations_root = os.path.join(project_root, "annotations")

    if not os.path.exists(annotations_root):
        print(f"Error: Annotations directory '{annotations_root}' not found.")
        return

    records = []

    print(f"Scanning '{annotations_root}' for JSON files...")

    # Walk through the annotations directory
    for root, dirs, files in os.walk(annotations_root):
        for file in files:
            # Look for Protocol 1 files
            if file.endswith(".json") and "I2I_P1" in file:
                file_path = os.path.join(root, file)

                # 1. Determine Species
                # Structure is usually: annotations/{Species}/p1/
                path_parts = os.path.normpath(root).split(os.sep)
                if "p1" in path_parts:
                    # Get the folder name immediately preceding 'p1'
                    idx = path_parts.index("p1")
                    species = path_parts[idx - 1]
                else:
                    # Fallback: assume parent folder is species
                    species = os.path.basename(root)

                # 2. Determine Gallery Size (N) from filename
                # Format 1: {Species}_I2I_P1.json (Implies N=4)
                # Format 2: {Species}_I2I_P1_N{N}.json
                match = re.search(r"_N(\d+)\.json$", file)
                if match:
                    n_val = int(match.group(1))
                else:
                    n_val = 4  # Default for P1

                # 3. Count samples
                try:
                    with open(file_path, "r") as f:
                        data = json.load(f)
                        count = len(data)

                    records.append(
                        {
                            "Species": species,
                            "Gallery Size": n_val,
                            "Num Samples": count,
                        }
                    )
                except Exception as e:
                    print(f"  Warning: Could not read {file}: {e}")

    if not records:
        print("No valid JSON annotation files found.")
        return

    # Create DataFrame
    df = pd.DataFrame(records)

    # Pivot to create the summary table: Rows=Species, Cols=Gallery Size
    pivot_df = df.pivot(index="Species", columns="Gallery Size", values="Num Samples")
    pivot_df = pivot_df.fillna(0).astype(int)

    # Configure pandas to display all rows and columns for terminal output
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)

    print("\n--- Generated Statistics ---")
    print(pivot_df)


if __name__ == "__main__":
    main()
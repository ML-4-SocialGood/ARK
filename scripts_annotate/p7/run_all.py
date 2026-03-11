import os
import subprocess
import sys


def main():
    # Define paths relative to the project root
    project_root = os.getcwd()
    data_root = os.path.join(project_root, "data")
    annotations_root = os.path.join(project_root, "annotations")

    # Scripts
    generate_script = os.path.join(project_root, "scripts_annotate", "p7", "generate_dataset.py")
    analyze_script = os.path.join(project_root, "scripts_annotate", "p7", "analyze_dataset.py")
    verify_script = os.path.join(project_root, "scripts_annotate", "p7", "verify_dataset.py")

    if not os.path.exists(data_root):
        print(f"Error: Data root directory '{data_root}' not found.")
        return

    # 1. Identify Species
    species_list = []
    for item in os.listdir(data_root):
        species_path = os.path.join(data_root, item)
        ids_path = os.path.join(species_path, "IDs")
        if os.path.isdir(species_path) and os.path.isdir(ids_path):
            species_list.append(item)
    species_list.sort()

    print(f"Found {len(species_list)} species for Protocol 7.")

    # 2. Process each species
    for i, species in enumerate(species_list):
        print(f"\n[{i + 1}/{len(species_list)}] Processing Species: {species}")
        data_dir = os.path.join("data", species, "IDs")

        # --- Step 1: Generate ---
        # Protocol 7 (Counterfactual) does not iterate over gallery sizes (N).
        # It generates a single dataset of negative pairs.
        gen_cmd = [
            sys.executable,
            generate_script,
            "--dataset_name",
            species,
            "--data_dir",
            data_dir,
            "--max_usage_per_id",
            "10",  # Allow slightly higher usage per ID to maximize pairs
            "--seed",
            "42",
        ]

        print("  > Generating P7 (Counterfactual) dataset...")
        try:
            subprocess.run(gen_cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"    Error generating P7 for {species}: {e}")
            continue

        # --- Step 2: Analyze ---
        base_name = f"{species}_P7"
        json_file = os.path.join(annotations_root, species, "p7", f"{base_name}.json")
        plot_file = os.path.join(annotations_root, species, "p7", f"{base_name}_distribution.png")

        if os.path.exists(analyze_script):
            if os.path.exists(json_file):
                print("  > Analyzing dataset statistics...")
                analyze_cmd = [
                    sys.executable,
                    analyze_script,
                    "--json_file",
                    json_file,
                    "--output_plot",
                    plot_file,
                ]
                try:
                    subprocess.run(analyze_cmd, check=True)
                except subprocess.CalledProcessError as e:
                    print(f"    Error analyzing dataset for {species}: {e}")
            else:
                print(f"    Warning: JSON file not found: {json_file}")
        else:
            print(f"    Note: Analyze script not found at {analyze_script}")

        # --- Step 3: Verify Dataset ---
        print(f"  > Verifying P7 dataset for {species}...")
        verify_cmd = [
            sys.executable,
            verify_script,
            "--dataset_name", species,
            "--annotations_dir", annotations_root,
            "--data_root", project_root,
        ]
        subprocess.run(verify_cmd, check=False)

    print("\nAll species processed for Protocol 7.")


if __name__ == "__main__":
    main()
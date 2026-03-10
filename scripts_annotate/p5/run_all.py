import os
import subprocess
import sys


def main():
    # Define paths relative to the project root
    project_root = os.getcwd()
    data_root = os.path.join(project_root, "data")
    annotations_root = os.path.join(project_root, "annotations")

    # Scripts
    generate_script = os.path.join(project_root, "scripts", "p5", "generate_dataset.py")
    analyze_script = os.path.join(project_root, "scripts", "p5", "analyze_dataset.py")

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

    print(f"Found {len(species_list)} species for Protocol 5.")

    # 2. Process each species
    for i, species in enumerate(species_list):
        print(f"\n[{i + 1}/{len(species_list)}] Processing Species: {species}")
        data_dir = os.path.join("data", species, "IDs")

        # Gallery sizes (Number of distractors)
        gallery_sizes = [4, 8, 16, 32]

        for N in gallery_sizes:
            print(f"  > Processing Gallery Size N={N} (Open Set)...")

            # --- Step 1: Generate ---
            gen_cmd = [
                sys.executable,
                generate_script,
                "--dataset_name",
                species,
                "--data_dir",
                data_dir,
                "--gallery_size",
                str(N),
                "--max_queries_per_id",
                "10000",  # Maximize samples
                "--seed",
                "42",
            ]

            try:
                subprocess.run(gen_cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"    Error generating P5 for {species} (N={N}): {e}")
                continue

            # --- Step 2: Analyze ---
            base_name = f"{species}_MCQ_P5_N{N}"
            json_file = os.path.join(
                annotations_root, species, "p5", f"{base_name}.json"
            )
            plot_file = os.path.join(
                annotations_root, species, "p5", f"{base_name}_distribution.png"
            )

            if os.path.exists(json_file):
                print(f"    Analyzing dataset statistics (N={N})...")
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
                    print(f"    Error analyzing dataset for {species} (N={N}): {e}")
            else:
                print(f"    Warning: JSON file not found: {json_file}")

    print("\nAll species processed for Protocol 5.")


if __name__ == "__main__":
    main()

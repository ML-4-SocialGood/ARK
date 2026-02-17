import os
import subprocess
import sys


def main():
    # Define paths relative to the project root
    project_root = os.getcwd()
    data_root = os.path.join(project_root, "data")
    annotations_root = os.path.join(project_root, "annotations")

    # Path to the scripts we want to run
    generate_script = os.path.join(project_root, "scripts", "p1", "generate_dataset.py")
    analyze_script = os.path.join(project_root, "scripts", "analyze_p1_dataset.py")

    # Check if data directory exists
    if not os.path.exists(data_root):
        print(f"Error: Data root directory '{data_root}' not found.")
        return

    # 1. Identify Species
    # Scan data/ for folders that contain an 'IDs' subdirectory
    species_list = []
    for item in os.listdir(data_root):
        species_path = os.path.join(data_root, item)
        ids_path = os.path.join(species_path, "IDs")
        if os.path.isdir(species_path) and os.path.isdir(ids_path):
            species_list.append(item)

    species_list.sort()

    if not species_list:
        print(
            "No valid species datasets found in 'data/'. Expected structure: data/{SpeciesName}/IDs/"
        )
        return

    print(f"Found {len(species_list)} species: {', '.join(species_list)}")
    print("=" * 60)

    # 2. Process each species
    for i, species in enumerate(species_list):
        print(f"\n[{i + 1}/{len(species_list)}] Processing Species: {species}")

        data_dir = os.path.join("data", species, "IDs")

        # --- Step 1: Generate Dataset ---
        # We use max_queries_per_id=100 to generate the maximum possible samples
        # (capped dynamically by the number of images per ID).
        gen_cmd = [
            sys.executable,
            generate_script,
            "--dataset_name",
            species,
            "--data_dir",
            data_dir,
            "--max_queries_per_id",
            "10000",
            "--seed",
            "42",
        ]

        print("  > Generating P1 (I2I) dataset...")
        try:
            subprocess.run(gen_cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"  Error generating dataset for {species}: {e}")
            continue

        # --- Step 2: Analyze Dataset ---
        # Define paths for input JSON and output Plot
        # We save the plot in the same folder as the annotation for better organization
        json_file = os.path.join(annotations_root, species, f"{species}_I2I_P1.json")
        plot_file = os.path.join(
            annotations_root, species, f"{species}_I2I_P1_distribution.png"
        )

        if not os.path.exists(json_file):
            print(f"  Warning: Expected JSON file not found: {json_file}")
            continue

        analyze_cmd = [
            sys.executable,
            analyze_script,
            "--json_file",
            json_file,
            "--output_plot",
            plot_file,
        ]

        print("  > Analyzing dataset statistics...")
        try:
            subprocess.run(analyze_cmd, check=True)
            print(f"  > Analysis plot saved to: {plot_file}")
        except subprocess.CalledProcessError as e:
            print(f"  Error analyzing dataset for {species}: {e}")

    print("\n" + "=" * 60)
    print("All species processed successfully.")


if __name__ == "__main__":
    main()

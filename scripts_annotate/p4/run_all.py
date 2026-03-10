import os
import subprocess
import sys


def main():
    # Define paths relative to the project root
    project_root = os.getcwd()
    data_root = os.path.join(project_root, "data", "MetaWild")

    # Path to the scripts we want to run
    generate_script = os.path.join(project_root, "scripts_annotate", "p4", "generate_dataset.py")
    analyze_script = os.path.join(project_root, "scripts_annotate", "p4", "analyze_dataset.py")

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
            "No valid species datasets found in 'data/MetaWild/'. Expected structure: data/MetaWild/{SpeciesName}/IDs/"
        )
        return

    print(f"Found {len(species_list)} species: {', '.join(species_list)}")
    print("=" * 60)

    # 2. Process each species
    for i, species in enumerate(species_list):
        print(f"\n[{i + 1}/{len(species_list)}] Processing Species: {species}")
        data_dir = os.path.join(data_root, species)

        # Define the gallery sizes to generate
        # Protocol 4: Context-aware Interleaved Reasoning (CIR)
        gallery_sizes = [4, 8, 16, 32]

        for N in gallery_sizes:
            print(f"  > Processing Gallery Size N={N}...")

            # --- Step 1: Generate Dataset ---
            gen_cmd = [
                sys.executable,
                generate_script,
                "--dataset_name",
                species,
                "--data_dir",
                data_dir,
                "--output_dir",
                os.path.join(project_root, "annotations", "MetaWild"),
                "--gallery_size",
                str(N),
                "--max_queries_per_id",
                "10000",
                "--seed",
                "42",
            ]

            print(f"    Generating P4 (CIR) dataset (N={N})...")
            try:
                subprocess.run(gen_cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"    Error generating dataset for {species} (N={N}): {e}")
                continue

            # --- Step 2: Analyze Dataset ---
            base_name = f"{species}_CIR_P4_N{N}"
            json_file = os.path.join(project_root, "annotations", "MetaWild", species, "p4", f"{base_name}.json")
            # analyze_dataset.py in P4 creates a directory for plots, not a single file
            output_dir = os.path.join(project_root, "annotations", "MetaWild", species, "p4", "analysis_results")

            if os.path.exists(json_file):
                print(f"    Analyzing dataset statistics (N={N})...")
                analyze_cmd = [
                    sys.executable,
                    analyze_script,
                    "--dataset_name",
                    json_file,
                    "--output_dir",
                    output_dir,
                ]
                try:
                    subprocess.run(analyze_cmd, check=True)
                except subprocess.CalledProcessError as e:
                    print(f"    Error analyzing dataset for {species} (N={N}): {e}")
            else:
                 print(f"    Warning: Expected JSON file not found for analysis: {json_file}")

    print("\n" + "=" * 60)
    print("All species processed successfully.")


if __name__ == "__main__":
    main()

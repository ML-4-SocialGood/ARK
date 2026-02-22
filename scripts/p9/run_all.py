import os
import subprocess
import sys


def main():
    # Define paths relative to the project root
    project_root = os.getcwd()
    data_root = os.path.join(project_root, "data")
    
    # Path to the scripts we want to run
    generate_script = os.path.join(project_root, "scripts", "p9", "generate_dataset.py")
    analyze_script = os.path.join(project_root, "scripts", "p9", "analyze_dataset.py")

    # Check if data directory exists
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
        
        # Configuration for Protocol 9
        gallery_sizes = [4, 8, 16, 32]
        num_positives_list = [2, 3, 4]

        for N in gallery_sizes:
            for M in num_positives_list:
                # Constraint check: M must be < N for distractors to exist
                if M >= N:
                    continue
                
                print(f"  > Processing Gallery Size N={N}, Positives M={M}...")

                # --- Step 1: Generate Dataset ---
                gen_cmd = [
                    sys.executable,
                    generate_script,
                    "--dataset_name", species,
                    "--data_dir", data_dir,
                    "--gallery_size", str(N),
                    "--num_positives", str(M),
                    "--max_queries_per_id", "10000", # Maximize samples
                    "--seed", "42",
                ]

                try:
                    subprocess.run(gen_cmd, check=True)
                except subprocess.CalledProcessError as e:
                    print(f"    Error generating dataset for {species} (N={N}, M={M}): {e}")
                    continue

                # --- Step 2: Analyze Dataset ---
                base_name = f"{species}_MIA_P9_N{N}_M{M}"
                json_file = os.path.join(project_root, "annotations", species, "p9", f"{base_name}.json")
                plot_file = os.path.join(project_root, "annotations", species, "p9", f"{base_name}_distribution.png")

                if os.path.exists(json_file):
                    analyze_cmd = [
                        sys.executable,
                        analyze_script,
                        "--json_file", json_file,
                        "--output_plot", plot_file,
                    ]
                    try:
                        subprocess.run(analyze_cmd, check=True)
                    except subprocess.CalledProcessError as e:
                        print(f"    Error analyzing dataset for {species} (N={N}, M={M}): {e}")
                else:
                    print(f"    Warning: Expected JSON file not found: {json_file}")

    print("\n" + "=" * 60)
    print("All species processed successfully.")


if __name__ == "__main__":
    main()
"""
scripts/p8/run_all.py
Automate P8 generation for all species based on existing P1 datasets.
"""
import os
import subprocess
import sys


def main():
    project_root = os.getcwd()
    annotations_root = os.path.join(project_root, "annotations")
    generate_script = os.path.join(project_root, "scripts", "p8", "generate_dataset.py")
    verify_script = os.path.join(project_root, "scripts", "p8", "verify_dataset.py")

    if not os.path.exists(annotations_root):
        print(f"Error: Annotations directory '{annotations_root}' not found.")
        return

    # 1. Identify Species with P1 data
    species_list = []
    if os.path.exists(annotations_root):
        for d in os.listdir(annotations_root):
            p1_path = os.path.join(annotations_root, d, "p1")
            if os.path.isdir(p1_path):
                species_list.append(d)

    species_list.sort()

    if not species_list:
        print("No species with P1 data found.")
        return

    print(f"Found {len(species_list)} species with P1 data: {', '.join(species_list)}")

    # Define Corruptions to generate
    # We generate a comprehensive set for the benchmark
    corruptions = [
        ("occlusion", [1, 2, 3]),  # 3 levels of occlusion
        ("resolution", [1, 2, 3]),  # 3 levels of resolution degradation
        ("grayscale", [1]),  # 1 level (binary)
    ]

    for i, species in enumerate(species_list):
        print(f"\n[{i + 1}/{len(species_list)}] Processing Species: {species}")

        p1_dir = os.path.join(annotations_root, species, "p1")

        # Find all P1 JSON files (e.g., BelugaID_I2I_P1_N4.json)
        p1_files = [
            f for f in os.listdir(p1_dir) if f.endswith(".json") and "I2I_P1" in f
        ]
        p1_files.sort()

        if not p1_files:
            print(f"  No P1 JSON files found in {p1_dir}")
            continue

        for p1_file in p1_files:
            p1_path = os.path.join(p1_dir, p1_file)
            print(f"  > Source: {p1_file}")

            for c_type, severities in corruptions:
                for sev in severities:
                    # Construct command
                    cmd = [
                        sys.executable,
                        generate_script,
                        "--p1_json",
                        p1_path,
                        "--corruption_type",
                        c_type,
                        "--severity",
                        str(sev),
                        "--data_root",
                        project_root,
                    ]

                    # Run generation
                    try:
                        subprocess.run(cmd, check=True)
                    except subprocess.CalledProcessError as e:
                        print(
                            f"      [ERROR] Failed to generate {c_type} S{sev} for {p1_file}: {e}"
                        )

        # Verify after generating all variants for this species
        print(f"  > Verifying P8 datasets for {species}...")
        verify_cmd = [sys.executable, verify_script, "--dataset_name", species, "--data_root", project_root]
        subprocess.run(verify_cmd, check=False)

    print("\n" + "=" * 60)
    print("All P8 processing complete.")

if __name__ == "__main__":
    main()
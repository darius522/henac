import os
import shutil

ROOT_DIR = "/N/slate/daripete/jstsp-dac/datasets/mushra_32khz"
DEST_DIR = "/N/slate/daripete/jstsp-dac/datasets/mushra_32khz_selected"

# Global list of filenames to copy (must match exactly)
FILENAMES_TO_COPY = {
    "001662_chunk_0.wav",
    "010125_chunk_1.wav",
    "029678_chunk_0.wav",
    "031474_chunk_0.wav",
    "003777_chunk_2.wav",
    "049021_chunk_0.wav",
    "121438_chunk_1.wav",
    "108394_chunk_0.wav",
}

def mirror_and_copy(root_dir, dest_dir, filenames):
    for root, _, files in os.walk(root_dir):
        rel_path = os.path.relpath(root, root_dir)
        dest_subdir = os.path.join(dest_dir, rel_path)
        os.makedirs(dest_subdir, exist_ok=True)

        for file in files:
            if file in filenames:
                src = os.path.join(root, file)
                dst = os.path.join(dest_subdir, file)
                shutil.copy2(src, dst)

def main():
    if not os.path.isdir(ROOT_DIR):
        raise RuntimeError(f"ROOT_DIR does not exist: {ROOT_DIR}")

    os.makedirs(DEST_DIR, exist_ok=True)

    mirror_and_copy(ROOT_DIR, DEST_DIR, FILENAMES_TO_COPY)

    print("Done.")

if __name__ == "__main__":
    main()

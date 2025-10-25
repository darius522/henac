import pandas as pd
import shutil
from pathlib import Path

# Paths
ROOT = Path("/N/slate/daripete/jstsp-dac/runs2/hb_18cb/300k/audios")     # Replace with your actual root
ROOT2 = Path("/N/slate/daripete/jstsp-dac/datasets/ablations")   # Destination directory
CSV_PATH = Path("/N/slate/daripete/jstsp-dac/datasets/fma_test_subset.csv")  # CSV file with 'path' column

# Folder mapping
FOLDER_MAP = {
    "input_[[False, True], [True, True]]": "input",
    "output_[[False, True], [True, True]]": "mb_hb",
    "output_[[True, False], [False, True]]": "hb",
    "output_[[True, False], [True, False]]": "cb",
    "output_[[True, True], [True, False]]": "cb_mb",
    "output_[[True, True], [True, True]]": "cb_mb_hb",
}

# Read CSV
df = pd.read_csv(CSV_PATH)
filenames = set(Path(p).name for p in df["path"])  # Just the filenames
filenames = [f.replace('.wav','_chunk_0.wav') for f in filenames]

# Iterate through each folder in FOLDER_MAP
for orig_folder_name, new_folder_name in FOLDER_MAP.items():
    src_folder = ROOT / orig_folder_name
    dst_folder = ROOT2 / new_folder_name
    dst_folder.mkdir(parents=True, exist_ok=True)

    for audio_file in src_folder.iterdir():
        if audio_file.name in filenames:
            shutil.copy(audio_file, dst_folder / audio_file.name)
            print(f"Copied {audio_file.name} → {new_folder_name}/")

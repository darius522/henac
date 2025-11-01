import os
import librosa
import traceback
import soundfile as sf
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# --- CONFIG ---
ROOT_SRC = "/N/project/SAIGE_shared/fma_large"
ROOT_DST = "/N/project/SAIGE_shared/fma_large_44khz"
TARGET_SR = 44100  # target sampling rate (Hz)
NUM_WORKERS = 20     # adjust based on your CPU cores
# ---------------

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def process_file(src_path, dst_path, target_sr):
    """Convert a single MP3 file to mono WAV at target_sr."""
    try:
        # Load + resample + convert to mono
        y, sr = librosa.load(src_path, sr=target_sr, mono=True)
        ensure_dir(os.path.dirname(dst_path))
        sf.write(dst_path, y, target_sr)
        return (src_path, True, None)
    except Exception as e:
        # Capture full info — always nonempty
        err_msg = f"{repr(e)}\n{traceback.format_exc(limit=1)}"
        return (src_path, False, err_msg)

def gather_mp3_files(root_src, root_dst):
    """Recursively find all .mp3 files and map to target .wav paths."""
    file_pairs = []
    for root, _, files in os.walk(root_src):
        for f in files:
            if f.lower().endswith(".mp3"):
                src_path = os.path.join(root, f)
                rel_path = os.path.relpath(src_path, root_src)
                dst_path = os.path.join(root_dst, os.path.splitext(rel_path)[0] + ".wav")
                file_pairs.append((src_path, dst_path))
    return file_pairs

def main():
    mp3_files = gather_mp3_files(ROOT_SRC, ROOT_DST)[:100]
    print(f"Found {len(mp3_files)} MP3 files to convert.\n")

    success_count = 0
    failed_files = []

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = [executor.submit(process_file, src, dst, TARGET_SR) for src, dst in mp3_files]
        
        for f in tqdm(as_completed(futures), total=len(futures), desc="Converting files", unit="file"):
            src_path, success, err = f.result()
            if success:
                success_count += 1
            else:
                failed_files.append((src_path, err))

    print(f"\n✅ Done! Converted {success_count}/{len(mp3_files)} files successfully.")
    if failed_files:
        print("\n⚠️ Failed files:")
        for src, err in failed_files:
            print(f"  - {src}: {err}")

if __name__ == "__main__":
    main()

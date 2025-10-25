import os
import librosa
import soundfile as sf
from tqdm import tqdm

# --- CONFIG ---
ROOT_SRC = "/N/project/SAIGE_shared/fma_large"
ROOT_DST = "/N/project/SAIGE_shared/fma_large_32k"
TARGET_SR = 32000  # <-- your target sampling rate
# ---------------

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def process_file(src_path, dst_path, target_sr):
    """Load, resample, and save audio as mono WAV."""
    try:
        # Load audio as mono and resample
        y, sr = librosa.load(src_path, sr=target_sr, mono=True)
        
        # Ensure destination directory exists
        ensure_dir(os.path.dirname(dst_path))
        
        # Save as WAV
        sf.write(dst_path, y, target_sr)
        return True

    except Exception as e:
        print(f"\n[ERROR] Failed to process {src_path}: {e}")
        return False

def main():
    # Collect all .mp3 files
    mp3_files = []
    for root, _, files in os.walk(ROOT_SRC):
        for f in files:
            if f.lower().endswith(".mp3"):
                src_path = os.path.join(root, f)
                rel_path = os.path.relpath(src_path, ROOT_SRC)
                dst_path = os.path.join(ROOT_DST, os.path.splitext(rel_path)[0] + ".wav")
                mp3_files.append((src_path, dst_path))

    # Process with progress bar
    success_count = 0
    failed_files = []

    for src_path, dst_path in tqdm(mp3_files, desc="Converting files", unit="file"):
        ok = process_file(src_path, dst_path, TARGET_SR)
        if ok:
            success_count += 1
        else:
            failed_files.append(src_path)

    print("\n✅ Conversion complete!")
    print(f"Successfully converted: {success_count}/{len(mp3_files)} files")

    if failed_files:
        print("\n⚠️ Failed files:")
        for f in failed_files:
            print(f"  - {f}")

if __name__ == "__main__":
    main()
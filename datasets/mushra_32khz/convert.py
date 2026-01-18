import os
import numpy as np
import soundfile as sf

ROOT_DIR = "/N/slate/daripete/jstsp-dac/datasets/mushra_32khz"  # <-- CHANGE THIS

def convert_mono_to_stereo(wav_path):
    data, samplerate = sf.read(wav_path)

    # Mono files usually have shape (n_samples,)
    if data.ndim == 1:
        print(f"Converting mono → stereo: {wav_path}")

        # Duplicate channel: (n_samples,) → (n_samples, 2)
        stereo_data = np.column_stack((data, data))

        # Preserve original subtype (PCM_16, PCM_24, FLOAT, etc.)
        info = sf.info(wav_path)

        sf.write(
            wav_path,
            stereo_data,
            samplerate,
            subtype=info.subtype
        )

def process_directory(root_dir):
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(".wav"):
                wav_path = os.path.join(root, file)
                try:
                    convert_mono_to_stereo(wav_path)
                except Exception as e:
                    print(f"Error processing {wav_path}: {e}")

if __name__ == "__main__":
    process_directory(ROOT_DIR)

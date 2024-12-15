import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

# Function to calculate the magnitude spectrogram
def calculate_magnitude_spectrogram(audio, sr, n_fft=1024, hop_length=512):
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(stft)
    return magnitude

# Function to calculate the ideal binary mask (IBM)
def calculate_ideal_binary_mask(magnitude1, magnitude2):
    return (magnitude1 > magnitude2).astype(float)

# Load the two audio files
file_path1 = "/home/daripete/jstsp-dac/zzz/test1.wav"
file_path2 = "/home/daripete/jstsp-dac/zzz/test2.wav"

audio1, sr1 = librosa.load(file_path1, sr=None)
audio2, sr2 = librosa.load(file_path2, sr=None)

# Ensure the two audio files have the same sampling rate and length
if sr1 != sr2:
    raise ValueError("Sampling rates of the two audio files must match.")

min_length = min(len(audio1), len(audio2))
audio1 = audio1[:min_length]
audio2 = audio2[:min_length]

import pdb; pdb.set_trace()

# Compute the mixture audio
mixture_audio = audio1 + audio2

# Calculate magnitude spectrograms
magnitude1 = calculate_magnitude_spectrogram(audio1, sr1)
magnitude2 = calculate_magnitude_spectrogram(audio2, sr2)
magnitude_mixture = calculate_magnitude_spectrogram(mixture_audio, sr1)

# Calculate ideal binary masks
ibm1 = calculate_ideal_binary_mask(magnitude1, magnitude2)
ibm2 = calculate_ideal_binary_mask(magnitude2, magnitude1)

# Plot and save the spectrograms and masks
plots = [
    (magnitude_mixture, "Mixture Magnitude Spectrogram", "mixture_spectrogram.png"),
    (magnitude1, "Source 1 Magnitude Spectrogram", "source1_spectrogram.png"),
    (magnitude2, "Source 2 Magnitude Spectrogram", "source2_spectrogram.png"),
    (ibm1, "Ideal Binary Mask (Source 1)", "ibm1.png"),
    (ibm2, "Ideal Binary Mask (Source 2)", "ibm2.png"),
]

for data, title, filename in plots:
    plt.figure(figsize=(10, 6))
    librosa.display.specshow(
        librosa.amplitude_to_db(data, ref=np.max),
        sr=sr1,
        hop_length=512,
        x_axis="time",
        y_axis="linear"
    )
    # plt.colorbar(format="%.2f dB")
    # plt.title(title)
    # plt.tight_layout()
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

print("Plots saved successfully!")

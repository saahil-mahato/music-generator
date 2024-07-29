import os
import librosa
import numpy as np
from scipy.io import wavfile


def load_audio_files(directory, sample_rate):
    audio_data = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.wav'):
                file_path = os.path.join(root, file)
                data, sr = librosa.load(file_path, sr=sample_rate)
                audio_data.append(data)
    return np.concatenate(audio_data)


def preprocess_audio_data(audio_data, seq_length):
    num_sequences = len(audio_data) // seq_length
    sequences = np.array([audio_data[i * seq_length:(i + 1) * seq_length] for i in range(num_sequences)])
    return sequences


def save_wav(filename, sample_rate, data):
    # Ensure the data is in the correct format
    data = np.asarray(data, dtype=np.int16)

    # Write the WAV file
    wavfile.write(filename, sample_rate, data)

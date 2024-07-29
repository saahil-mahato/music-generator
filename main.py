from midi_processing import load_audio_files, preprocess_audio_data, save_wav

from model import AudioTransformer

import torch


# Main Function
def main(directory):
    audio_data = load_audio_files(directory, sample_rate=16000)
    sequences = preprocess_audio_data(audio_data, seq_length=1024)
    print(sequences.shape)

    # Initialize and train the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AudioTransformer(
        input_dim=sequences.shape[1],
        d_model=sequences.shape[1],
        nhead=8,
        num_encoder_layers=3,
        num_decoder_layers=3,
        dim_feedforward=512,
        dropout=0.1,
        device=device
    ).to(device)

    # sequences = model.normalize_data(sequences)
    model.train_model(data=sequences, epochs=10, batch_size=32, learning_rate=0.001)

    # Generate new sequences
    initial_sequence = sequences[0]  # Example initial sequence
    generated_sequence = model.generate(initial_sequence, seq_len=1024)
    print(generated_sequence)


if __name__ == '__main__':
    MIDI_DIRECTORY = 'mid_files'
    OUTPUT_FILENAME = 'output_midi_file.mid'

    main(MIDI_DIRECTORY)

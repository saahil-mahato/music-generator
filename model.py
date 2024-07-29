import torch
import numpy as np
import torch.nn as nn


def create_positional_encoding(max_len, d_model):
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.unsqueeze(0)  # (1, max_len, d_model)
    return pe


class AudioTransformer(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout, device, max_len=5000):
        super(AudioTransformer, self).__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.device = device

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.fc_in = nn.Linear(input_dim, d_model)
        self.fc_out = nn.Linear(d_model, input_dim)

        self.positional_encoding = create_positional_encoding(max_len, d_model).to(device)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc_in.weight)
        nn.init.xavier_uniform_(self.fc_out.weight)

    def forward(self, src, tgt):
        print(f"src shape: {src.shape}")  # Debugging
        print(f"tgt shape: {tgt.shape}")  # Debugging

        src = self.fc_in(src)
        tgt = self.fc_in(tgt)

        src_len = src.size(1)
        tgt_len = tgt.size(1)
        max_len = max(src_len, tgt_len)

        if self.positional_encoding.size(1) < max_len:
            self.positional_encoding = create_positional_encoding(max_len, self.d_model).to(self.device)

        src = src + self.positional_encoding[:, :src_len, :]
        tgt = tgt + self.positional_encoding[:, :tgt_len, :]
        output = self.transformer(src, tgt)
        return self.fc_out(output)

    def train_model(self, data, epochs, batch_size, learning_rate):
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        data = torch.tensor(data, dtype=torch.float32).to(self.device)
        print(f"Training data shape: {data.shape}")  # Debugging

        for epoch in range(epochs):
            for i in range(0, len(data) - batch_size, batch_size):
                batch = data[i:i+batch_size]
                print(f"Batch shape: {batch.shape}")  # Debugging
                src = batch[:, :-1]
                tgt = batch[:, 1:]
                print(f"src shape (after slicing): {src.shape}")  # Debugging
                print(f"tgt shape (after slicing): {tgt.shape}")  # Debugging

                optimizer.zero_grad()
                output = self(src, tgt)
                loss = criterion(output, tgt)
                loss.backward()
                optimizer.step()

            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    def generate(self, initial_sequence, seq_len):
        self.eval()
        with torch.no_grad():
            initial_sequence = torch.tensor(initial_sequence, dtype=torch.float32).unsqueeze(0).to(self.device)
            generated_sequence = initial_sequence

            for _ in range(seq_len - initial_sequence.size(1)):
                tgt = generated_sequence[:, -1:]
                output = self(generated_sequence, tgt)
                generated_sequence = torch.cat((generated_sequence, output[:, -1:]), dim=1)

            return generated_sequence.squeeze(0).cpu().numpy()

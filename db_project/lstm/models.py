import torch
import torch.nn as nn


class Trunk(nn.Module):
    """Модель на базе LSTM, переводящая текст в вектор."""

    def __init__(
        self,
        input_size: int,
        vocab_size: int,
        embedding_dim: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        device: str,
    ):
        super().__init__()
        self.device = device
        self.input_size = input_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embeddings = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=embedding_dim
        ).to(self.device)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=True,
            batch_first=True,
        ).to(self.device)

    def forward(self, x):
        x = self.embeddings(x)
        h_0, c_0 = torch.zeros(
            (2 * self.num_layers, x.size(0), self.hidden_size),
            dtype=torch.float32,
            device=self.device,
        ), torch.zeros(
            (2 * self.num_layers, x.size(0), self.hidden_size),
            dtype=torch.float32,
            device=self.device,
        )
        x, (hn, cn) = self.lstm(x, (h_0, c_0))
        x = torch.cat([hn[i, :, :] for i in range(hn.shape[0])], dim=1)
        return x


class Embedder(nn.Module):
    """Модель, выполняющая проекцию на подпространство."""

    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.fc1 = nn.Linear(input_size, input_size // 2, dtype=torch.float32)
        self.fc2 = nn.Linear(input_size // 2, output_size, dtype=torch.float32)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

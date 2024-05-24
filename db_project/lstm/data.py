import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchtext.vocab import Vocab, build_vocab_from_iterator

from db_project.utils import split_data, yield_tokens

MAX_LENGTH = 29
BATCH_SIZE = 128
NUM_CLASSES = 248


class Data(Dataset):
    def __init__(self, data: np.ndarray, targets: np.ndarray, vocab: Vocab):
        self.labels = targets
        self.vocab = vocab
        self.tokens = np.asarray(list(map(self.tokenize, data)))

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx: int):
        return torch.tensor(self.tokens[idx], dtype=torch.int64), torch.tensor(
            self.labels[idx], dtype=torch.int64
        )

    def tokenize(self, text: str) -> list[int]:
        tokens = [self.vocab[token] for token in text]
        while len(tokens) < MAX_LENGTH:
            tokens.append(self.vocab["<pad>"])
        while len(tokens) > MAX_LENGTH:
            tokens.pop()
        return tokens


def build_vocab(df: pd.DataFrame) -> Vocab:
    vocab = build_vocab_from_iterator(
        yield_tokens(df[["category_id", "title_tokens"]].values),
        specials=["<pad>", "<unk>"],
        min_freq=5,
    )
    vocab.set_default_index(vocab["<unk>"])
    return vocab


def build_datasets(df: pd.DataFrame, vocab: Vocab) -> tuple[Data, Data, Data]:
    """
    Возвращает датасеты для обучения и тестирования.
    df: сырые данные.
    vocab: словарь токенов.
    """
    train_data, val_data, test_data = split_data(df[["category_id", "title_tokens"]])
    train_dataset = Data(train_data[:, 1], train_data[:, 0], vocab)
    val_dataset = Data(val_data[:, 1], val_data[:, 0], vocab)
    test_dataset = Data(test_data[:, 1], test_data[:, 0], vocab)
    return train_dataset, val_dataset, test_dataset

import pickle

import torch
from models import Embedder, Trunk
import os
import numpy as np
from db_project.utils import push_products_embeddings
from tqdm import tqdm

MAX_LENGTH = 29
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def get_lstm_products_embeddings(
    trunk: Trunk, embedder: Embedder, tokens_path: str, vocab
) -> None:
    """
    Строит эмбеддинги для всех товаров в таблице.
    model: Word2Vec модель для перевода.
    tokens_path: путь к данным.
    """
    with open(tokens_path, "rb") as f:
        tokens = pickle.load(f)
    emb_tokens = [
        get_lstm_text_embeddings(trunk, embedder, t, vocab)
        for t in tqdm(tokens, total=len(tokens))
    ]

    with open("./pretrained/lstm_embeddings.pkl", "wb+") as f:
        pickle.dump(emb_tokens, f)


def get_lstm_text_embeddings(
    trunk: Trunk, embedder: Embedder, tokens: list[str], vocab
) -> np.ndarray:
    """
    Переводит текст в вектор применением модели.
    trunk: основная кодирующая для перевода.
    embedder: проецирующая для перевода.
    tokens: список токенов в предложении.
    vocab: словарь.
    """
    tokens_ = [np.asarray([vocab[token] for token in tokens])]
    with torch.no_grad():
        x = trunk(torch.tensor(np.asarray(tokens_)))
        x = embedder(x)
    return x.cpu().numpy()[0]


def main():
    vec_size = int(os.environ["vec_size"])

    with open("pretrained/vocab.pkl", "rb") as f:
        vocab = pickle.load(f)

    trunk = Trunk(
        input_size=MAX_LENGTH,
        vocab_size=len(vocab),
        embedding_dim=128,
        hidden_size=64,
        num_layers=2,
        dropout=0.5,
        device=DEVICE,
    )
    embedder = Embedder(input_size=64 * 2 * 2, output_size=vec_size)

    trunk.load_state_dict(torch.load("pretrained/trunk.pt", map_location=DEVICE))
    embedder.load_state_dict(torch.load("pretrained/embedder.pt", map_location=DEVICE))

    print("Computing embeddings started")
    get_lstm_products_embeddings(
        trunk, embedder, "../word2vec/pretrained/tokens.pkl", vocab
    )
    print("Computing embeddings finished")

    print("Pushing embeddings started")
    push_products_embeddings("./pretrained/lstm_embeddings.pkl", "lstm_emb")
    print("Pushing embeddings finished")


if __name__ == "__main__":
    main()

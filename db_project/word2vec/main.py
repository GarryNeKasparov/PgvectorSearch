import multiprocessing as mp
import os
import pickle
import subprocess

import numpy as np
from gensim.models import Word2Vec

from db_project.utils import push_products_embeddings


def train_w2v(tokens_path: str, min_count: int, vector_size: int, window: int) -> None:
    """
    Обучает модель Word2Vec.
    tokens_path: путь к данным для обучения.
    min_count: минимальная частота встречаемости слова.
    vector_size: размерность эмбеддингов.
    window: размер контекста.
    """
    with open(tokens_path, "rb") as f:
        tokens = pickle.load(f)
    model = Word2Vec(
        tokens,
        min_count=min_count,
        vector_size=vector_size,
        window=window,
        workers=mp.cpu_count(),
    )
    model.save("./pretrained/word2vec.model")


def get_w2v_text_embeddings(
    model: Word2Vec, tokens: list[str], vector_size: int
) -> np.ndarray:
    """
    Переводит текст в вектор усреднением эмбеддингов слов.
    model: Word2Vec модель для перевода.
    tokens: список токенов в предложении.
    vector_size: размерность эмбеддингов.
    """
    return np.mean(
        [model.wv[t] if t in model.wv else np.zeros(vector_size) for t in tokens],
        axis=0,
    )


def get_w2v_products_embeddings(
    model: Word2Vec, tokens_path: str, vector_size: int
) -> None:
    """
    Строит эмбеддинги для всех товаров в таблице.
    model: Word2Vec модель для перевода.
    tokens_path: путь к данным.
    vector_size: размерность эмбеддингов.
    """
    with open(tokens_path, "rb") as f:
        tokens = pickle.load(f)
    emb_tokens = [get_w2v_text_embeddings(model, t, vector_size) for t in tokens]
    with open("./pretrained/w2v_embeddings.pkl", "wb+") as f:
        pickle.dump(emb_tokens, f)


if __name__ == "__main__":
    vec_size = int(os.environ["vec_size"])

    print("Tokenizing started")
    subprocess.call(["python", "tokenize_text.py"])
    print("Tokenizing finished")

    print("Word2Vec training started")
    train_w2v("./pretrained/tokens.pkl", min_count=1, vector_size=vec_size, window=5)
    print("Word2Vec trainings finished")

    model = Word2Vec.load("./pretrained/word2vec.model")
    print("Computing embeddings started")
    get_w2v_products_embeddings(model, "./pretrained/tokens.pkl", vec_size)
    print("Computing embeddings finished")

    print("Pushing embeddings started")
    push_products_embeddings("./data/w2v_embeddings.pkl", "word2vec_emb")
    print("Pushing embeddings finished")

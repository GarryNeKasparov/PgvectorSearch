import os
import pickle

import numpy as np
import pandas as pd
import psycopg2
import spacy
from pgvector.psycopg2 import register_vector
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def split_data(df: pd.DataFrame):
    """
    Разделяет данные на выборки.
    df: данные для разбиения.
    """
    train, test = train_test_split(
        df.index, test_size=0.15, random_state=42, stratify=df["category_id"]
    )
    train, val = train_test_split(
        train, test_size=0.15, random_state=42, stratify=df.iloc[train]["category_id"]
    )
    train_data, val_data, test_data = (
        df.iloc[train].values,
        df.iloc[val].values,
        df.iloc[test].values,
    )
    assert (
        not np.isin(train, val).any()
        and not np.isin(train, test).any()
        and not np.isin(val, test).any()
    )
    return train_data, val_data, test_data


def text_preprocess(df: pd.Series) -> list[str]:
    """
    Возвращает токенизированный текст.
    df : данные для обработки.
    """
    df = (
        df.str.lower()
        .str.replace(r"[^a-zA-Z ]+", "", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    tokens = []
    nlp = spacy.load("en_core_web_sm")
    for doc in tqdm(nlp.pipe(df, n_process=-1), total=df.shape[0]):
        tokens.append([token.lemma_ for token in doc if not token.is_stop])

    return tokens


def yield_tokens(data_iter: np.ndarray):
    """
    Используется при построении словаря.
    data_iter: дата, по которой строится словарь.
    """
    for _, text in data_iter:
        yield text


def push_products_embeddings(
    embeddings_path: str, column_name: str, vec_size: int = -1
) -> None:
    """
    Добавляет в таблицу столбец с эмбеддингами.
    embeddings_path: путь до эмбеддингов.
    column_name: имя столбца.
    """
    if vec_size == -1:
        vec_size = int(os.environ["vec_size"])
    try:
        conn = psycopg2.connect(
            user="postgres",
            password=os.environ["DB_PASSWORD"],
            host="127.0.0.1",
            port="5432",
            database="VectorBase",
        )
        register_vector(conn)
        cur = conn.cursor()
        with open(embeddings_path, "rb") as f:
            embeddings = pickle.load(f)
        cur.execute(f"ALTER TABLE products DROP COLUMN IF EXISTS {column_name}")
        cur.execute(f"ALTER TABLE products ADD {column_name} vector({vec_size});")
        print("Altered products table")
        cur.execute(
            f"CREATE TABLE temp (index SERIAL PRIMARY KEY, emb vector({vec_size}));"
        )
        print("Created temp table")
        print("Filling temp table")
        for embedding in tqdm(embeddings, total=len(embeddings)):
            if isinstance(embedding, np.ndarray):
                val = embedding
            else:
                val = np.full(int(vec_size), -100)
            cur.execute("INSERT INTO temp (emb) VALUES (%s);", (val,))
        print("Filling complete")
        cur.execute(
            f"UPDATE products as p SET {column_name} = t.emb FROM temp as t WHERE p.index = t.index-1;"
        )
        print("Updating complete")
        cur.execute("DROP TABLE temp;")
        print("Dropped temp table")
        conn.commit()
    except (Exception, psycopg2.Error) as error:
        print(error)
    finally:
        if conn:
            cur.close()
            conn.close()
            print("PostgreSQL connection is closed")

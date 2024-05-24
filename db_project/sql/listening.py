import psycopg2
import os
import spacy
from gensim.models import Word2Vec
from pgvector.psycopg2 import register_vector
from db_project.word2vec.main import get_w2v_text_embeddings
import numpy as np


def get_category(cursor, embeddings: np.ndarray) -> str:
    """
    Находит ближайший объект для данного вектора.
    Возвращает его класс.

    embeddings: вектор представления объекта.
    cursor: интерфейс для взаимодействия с базой данных.
    """
    cursor.execute(
        "SELECT category_name FROM products "
        + "ORDER BY word2vec_emb <-> (%s) LIMIT 5;",
        (embeddings,),
    )
    return cursor.fetchone()[0]


def insert_item(cursor, description: str, category_name: str, embeddings: np.ndarray):
    """
    Выполняет вставку нового товара в таблицу.

    description: описание товара.
    category_name: категория товара.
    embeddings: вектор представления товара.
    cursor: интерфейс для взаимодействия с базой данных.
    """

    cursor.execute(
        (
            "INSERT INTO products ( "
            "description, "
            "category_name, "
            "word2vec_emb "
            ") VALUES (%s, %s, %s);",
            (
                description,
                category_name,
                embeddings,
            ),
        )
    )


def process_notify(cursor, description: str, model: Word2Vec, nlp):
    """
    Обрабатывает сигнал вставки нового товара.

    description: описание нового товара.
    cursor: интерфейс для взаимодействия с базой данных.
    model: модель Word2Vec.
    nlp: токенайзер spacy.
    """

    embeddings = generate_embeddings(model, nlp, description)
    category_name = get_category(cursor, embeddings)
    # insert_item(cursor, description, category_name, embeddings)
    print(description, category_name)


def generate_embeddings(model: Word2Vec, nlp, description: str) -> np.ndarray:
    """
    Генерирует word2vec представление для описание товара.

    description: описание нового товара.
    """
    doc = nlp(description)
    tokens = [token.lemma_ for token in doc if not token.is_stop]
    emb = get_w2v_text_embeddings(model, tokens, int(os.environ["vec_size"]))
    return emb


def listen_to_notifications():
    model = Word2Vec.load("../word2vec/pretrained/word2vec.model")
    nlp = spacy.load("en_core_web_sm")
    conn = psycopg2.connect(
        user="postgres",
        password=os.environ["DB_PASSWORD"],
        host="127.0.0.1",
        port="5432",
        database="VectorBase",
    )
    conn.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
    register_vector(conn)

    cursor = conn.cursor()
    cursor.execute("LISTEN generate_embeddings;")
    with conn.cursor() as cursor:
        cursor.execute("LISTEN generate_embeddings;")
        print("Ready to listen")
        while True:
            conn.poll()
            if conn.notifies:
                notify = conn.notifies.pop(0)
                description = notify.payload
                process_notify(cursor, description, model, nlp)


if __name__ == "__main__":
    try:
        listen_to_notifications()
    except Exception as e:
        print(e)

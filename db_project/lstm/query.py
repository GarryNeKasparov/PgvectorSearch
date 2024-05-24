import os

import pandas as pd
import psycopg2
import spacy
from gensim.models import Word2Vec
from pgvector.psycopg2 import register_vector
import torch
import pickle
from db_project.lstm.main import get_lstm_text_embeddings
from db_project.lstm.models import Trunk, Embedder

MAX_LENGTH = 29
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
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
    nlp = spacy.load("en_core_web_sm")
    vec_size = int(os.environ["vec_size"])
    embedder = Embedder(input_size=64 * 2 * 2, output_size=vec_size)
    query = input("Enter your query:\n")
    while query != ":q":
        doc = nlp(query)
        tokens = [token.lemma_ for token in doc if not token.is_stop]
        emb = get_lstm_text_embeddings(trunk, embedder, tokens, vocab)
        cur.execute(
            "SELECT category_name, description FROM products "
            + "ORDER BY lstm_emb <=> (%s) LIMIT 5;",
            (emb,),
        )
        res = cur.fetchall()
        print(pd.DataFrame(res, columns=["category_name", "description"]))
        query = input("Enter your query:\n")

except (Exception, psycopg2.Error) as error:
    print(error)

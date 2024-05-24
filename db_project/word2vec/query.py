import os

import pandas as pd
import psycopg2
import spacy
from gensim.models import Word2Vec
from pgvector.psycopg2 import register_vector

from db_project.word2vec.main import get_w2v_text_embeddings

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
    model = Word2Vec.load("./data/word2vec.model")
    nlp = spacy.load("en_core_web_sm")
    query = input("Enter your query:\n")
    while query != ":q":
        doc = nlp(query)
        tokens = [token.lemma_ for token in doc if not token.is_stop]
        emb = get_w2v_text_embeddings(model, tokens, int(os.environ["vec_size"]))
        cur.execute(
            "SELECT index, category, description FROM products "
            + "ORDER BY word2vec_emb <=> (%s) LIMIT 5;",
            (emb,),
        )
        res = cur.fetchall()
        print(pd.DataFrame(res, columns=["index", "category", "description"]))
        query = input("Enter your query:\n")


except (Exception, psycopg2.Error) as error:
    print(error)

import os
import pickle

import pandas as pd
import psycopg2

from db_project.utils import text_preprocess

if __name__ == "__main__":
    try:
        conn = psycopg2.connect(
            user="postgres",
            password=os.environ["DB_PASSWORD"],
            host="127.0.0.1",
            port="5432",
            database="VectorBase",
        )
        cur = conn.cursor()
        query = "SELECT description FROM products"
        cur.execute(query)
        desc = pd.Series(cur.fetchall()).apply(lambda x: x[0])
        tokens = text_preprocess(desc)
        with open("./data/tokens.pkl", "wb+") as f:
            pickle.dump(tokens, f)
    except (Exception, psycopg2.Error) as error:
        print(error)
    finally:
        if conn:
            cur.close()
            conn.close()
            print("PostgreSQL conn is closed")

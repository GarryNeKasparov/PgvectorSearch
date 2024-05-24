import os
import argparse
import psycopg2
from psycopg2.extensions import AsIs
from pgvector.psycopg2 import register_vector
from tqdm import tqdm
import numpy as np


def get_gin_score(k: int, n: int) -> np.ndarray:
    """
    Выдает оценку gin поиска и word2vec по метрике map@k.
    k: количество выданных товаров по запросу.
    n: количество товаров для тестирования.
    """

    def ap_at_k(preds: list[tuple[str]], target: str):
        """
        Вычисляет ap@k (average precision at k).
        preds: категория найденных товаров.
        target: категория искомого товара.
        """
        score = 0.0
        num_hits = 0.0
        for i, p in enumerate(preds):
            if p[0] == target and p not in preds[:i]:
                num_hits += 1.0
                score += num_hits / (i + 1.0)
        return score

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
        cur.execute(
            (
                "SELECT description, word2vec_emb, category_name "
                f"FROM products ORDER BY random() LIMIT {n}"
            )
        )
        rows = cur.fetchall()
        precisions = [[], []]
        for row in tqdm(rows, total=len(rows)):
            # print(row)
            query = np.asarray(row[0].split(" "))
            mask = np.random.choice([True, False], size=len(query))
            query = " ".join(query[mask]).replace('\'', '')
            # print(query)
            for i in range(2):
                if i == 0:
                    cur.execute(
                        (
                            "EXPLAIN ANALYZE SELECT category_name "
                            "FROM products "
                            f"WHERE ts_emb @@ plainto_tsquery('{query}') "
                            f"ORDER BY ts_rank(ts_emb, plainto_tsquery('{query}')) DESC;"
                        )
                    )
                else:
                    cur.execute(
                        (
                            "EXPLAIN ANALYZE SELECT category_name FROM products "
                            "ORDER BY word2vec_emb <=> %s LIMIT %s;"
                        ),
                        (row[1], k,),
                    )
                res = cur.fetchall()
                precisions[i].append(
                    float(res[-1][0].split(': ')[1].split(' ms')[0]))
                # ap = ap_at_k(res[1:], row[-1])
                # precisions[i].append(ap)
        scores = [np.mean(x) for x in precisions]
    except (Exception, psycopg2.Error) as error:
        print(error)
    finally:
        if conn:
            cur.close()
            conn.close()
            print("PostgreSQL connection is closed")

    return np.round(scores, 2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Scoring model", description="Computes map@k metric."
    )
    parser.add_argument("-k", help="Number of objects per query.")
    parser.add_argument("-n", help="Number of objects to test.")
    args = parser.parse_args()
    print(get_gin_score(int(args.k), int(args.n)))

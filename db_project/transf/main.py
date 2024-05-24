import pandas as pd
from sentence_transformers import SentenceTransformer
import pickle
import torch
import numpy as np
from db_project.utils import push_products_embeddings

MAX_LENGTH = 29
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def main():
    print("Computing embeddings started")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    df = pd.read_csv("../data/amazon_products.csv")
    embeddings = model.encode(
        df["description"], show_progress_bar=True, device="cuda", batch_size=256
    )
    embeddings.astype(np.float16)[0]
    with open("/pretrained/transf_emb_16.pkl", "wb") as f:
        pickle.dump(embeddings.astype(np.float16), f)
    print("Computing embeddings finished")

    print("Pushing embeddings started")
    push_products_embeddings("./pretrained/transf_emb_16.pkl", "transf_emb", 384)
    print("Pushing embeddings finished")


if __name__ == "__main__":
    main()

import pickle

import pandas as pd
import torch
from data import Data, build_datasets, build_vocab
from pytorch_metric_learning import (
    distances,
    losses,
    miners,
    reducers,
    samplers,
    testers,
    trainers,
)
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from torch import optim

from db_project.lstm.models import Embedder, Trunk

BATCH_SIZE = 128
MAX_LENGTH = 29


def training_config(
    trunk: Trunk,
    embedder: Embedder,
    trunk_optimizer: optim.Optimizer,
    embedder_optimizer: optim.Optimizer,
    train_dataset: Data,
    loss: losses.BaseMetricLossFunction,
    miner: miners.BaseMiner,
):
    models = {"trunk": trunk, "embedder": embedder}
    optimizers = {
        "trunk_optimizer": trunk_optimizer,
        "embedder_optimizer": embedder_optimizer,
    }
    loss_funcs = {"metric_loss": loss}
    mining_funcs = {"tuple_miner": miner}
    sampler = samplers.MPerClassSampler(
        train_dataset.labels, m=4, length_before_new_iter=len(train_dataset)
    )

    tester = testers.GlobalEmbeddingSpaceTester(
        dataloader_num_workers=2,
        accuracy_calculator=AccuracyCalculator(),
    )

    trainer = trainers.MetricLossOnly(
        models,
        optimizers,
        BATCH_SIZE,
        loss_funcs,
        train_dataset,
        mining_funcs=mining_funcs,
        sampler=sampler,
        dataloader_num_workers=2,
    )
    return trainer, tester


if __name__ == "__main__":
    df = pd.read_csv("../data/amazon_products.csv")
    with open("../word2vec/pretrained/tokens.pkl", "rb") as f:
        tokens = pickle.load(f)
    df["title_tokens"] = tokens
    vocab = build_vocab(df)
    print("Creating vocab")
    print(f"Vocab size: {len(vocab)}")
    with open("pretrained/vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)
    print("Vocab saved")
    train, val, test = build_datasets(df, vocab)

    trunk = Trunk(MAX_LENGTH, len(vocab), 128, 64, 2, 0.5, "cpu")
    embedder = Embedder(64 * 2 * 2, 32)

    embedder_optimizer = optim.Adam(embedder.parameters(), lr=1e-4)
    trunk_optimizer = optim.Adam(trunk.parameters(), lr=1e-3)

    distance = distances.CosineSimilarity()
    reducer = reducers.ThresholdReducer(low=0)
    loss = losses.TripletMarginLoss(margin=1, distance=distance, reducer=reducer)
    miner = miners.TripletMarginMiner(
        margin=1, distance=distance, type_of_triplets="semihard"
    )
    trainer, tester = training_config(
        trunk, embedder, trunk_optimizer, embedder_optimizer, train, loss, miner
    )

    trainer.train(num_epochs=10)
    dataset_dict = {"val": val, "test": test}
    all_accuracies = tester.test(dataset_dict, 1, trunk, embedder)
    print(all_accuracies)
    torch.save(trunk.state_dict(), "pretrained/trunk.pt")
    torch.save(embedder.state_dict(), "pretrained/embedder.pt")
    print("Model saved")

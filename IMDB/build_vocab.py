import os
import pickle

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import dataset_vocab
from vocab import Vocab


def get_dataloader(train):
    imdb_dataset = dataset_vocab.ImdbDataset(train, sequence_max_len=100)
    my_dataloader = DataLoader(imdb_dataset, batch_size=200, shuffle=True, collate_fn=dataset_vocab.collate_fn)
    return my_dataloader


if __name__ == '__main__':
    ws = Vocab()
    dl_train = get_dataloader(True)
    dl_test = get_dataloader(False)
    for reviews, _ in tqdm(dl_train, total=len(dl_train)):
        for sentence in reviews:
            ws.fit(sentence)
    for reviews, _ in tqdm(dl_test, total=len(dl_test)):
        for sentence in reviews:
            ws.fit(sentence)
    ws.build_vocab()
    print(len(ws))
    if not os.path.exists("./models"):
        os.makedirs("./models")
    pickle.dump(ws, open("./models/vocab.pkl", "wb"))
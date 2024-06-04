import numpy as np
import pandas as pd
import os
import string

from gensim.models import KeyedVectors

import torch

data_dir = os.listdir('data/')

if 'cbow_s300.txt' not in data_dir or 'cbow_s300.zip' not in data_dir:
    import wget
    import zipfile
    if 'cbow_s300.txt' not in data_dir:
        if 'cbow_s300.zip' not in data_dir:
            print('downloading cbow_s300.zip...')
            wget.download('http://143.107.183.175:22980/download.php?file=embeddings/word2vec/cbow_s300.zip', 'data/cbow_s300.zip')
            print('download completed')

        print('extracting cbow_s300.zip')
        with zipfile.ZipFile('data/cbow_s300.zip', 'r') as zip:
            zip.extractall('data/')
        print('extraction completed')


print('training word2vec model')
model_w2v = KeyedVectors.load_word2vec_format('./data/cbow_s300.txt')
print('training completed')


def w2v(text: str):
    pontuacao = string.punctuation
    text_vector = np.zeros((0, 300))

    for p in pontuacao:
        text = text.replace(p, "")

    text = text.split(" ")
    text = list(map(lambda x: x.lower(), text))

    for word in text:
        try:
            vector = model_w2v.get_vector(word)
            text_vector = np.vstack((text_vector, vector))
        except KeyError:
            pass
    return text_vector


def forward_step(mode: str, text: str, y: int, net, criterion, optimizer):
    net.train() if mode == "train" else net.eval()

    text_vector = torch.from_numpy(w2v(text).astype(np.float32))
    output = net(text_vector)

    y = torch.tensor([y])
    loss = criterion(output, y)

    _, pred_y = torch.max(output, axis=-1)
    hit = 1 if pred_y[0].item() == y[0].item() else 0

    if mode == "train":
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss, hit


def epoch_controler(mode: str, df: pd.DataFrame, net, criterion, optimizer):
    loss_epoch = []
    accuracy_epoch = 0

    for _, line in df.iterrows():
        loss, hit = forward_step(
            mode, line["text_pt"], line["sentiment"], net, criterion, optimizer
        )

        loss_epoch.append(loss.detach().cpu().numpy())
        accuracy_epoch += hit

    loss_epoch = np.asarray(loss_epoch).ravel()
    accuracy_epoch = accuracy_epoch / float(len(df))

    print(f"Mode: {mode}\nLoss: {loss_epoch.mean()}\nAccuracy: {accuracy_epoch}")
    print("-" * 80)

    return loss_epoch.mean(), accuracy_epoch

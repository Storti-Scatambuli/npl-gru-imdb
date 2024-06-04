import timeit

import matplotlib.pyplot as plt

import pandas as pd

from sklearn.model_selection import train_test_split

from torch import nn
import torch

from utils import *

ARGS = {
    "num_epochs": 500,
    "lr": 7e-6,
    "weight_decay": 7e-3,
    "dataset_size": 10000
}

print("importando database")

df = pd.read_csv("./data/imdb-reviews-pt-br.csv")
df = df.sample(n=ARGS['dataset_size']).reset_index(drop=True)
df.drop(columns=["text_en"], inplace=True)
df["sentiment"].replace("neg", 0, inplace=True)
df["sentiment"].replace("pos", 1, inplace=True)
train_df, test_df = train_test_split(df, test_size=0.3, random_state=23)

print("database importada e configurada")


print("Inicializando rede neural")
net = NLP(input_size=300, hidden_size=150, output_size=2)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    net.parameters(), lr=ARGS["lr"], weight_decay=ARGS["weight_decay"]
)

loss_train, loss_test = [], []
accuracy_train, accuracy_test = [], []
print("Iniciando treino")
for epoch in range(ARGS["num_epochs"]):
    print("=" * 80)
    print(f"Epoch: {epoch}")
    print("-" * 80)

    start = timeit.default_timer()

    #TRAIN
    loss, accuracy = epoch_controler("train", train_df, net, criterion, optimizer, 4000)
    loss_train.append(loss)
    accuracy_train.append(accuracy)
    
    #TEST
    loss, accuracy = epoch_controler("test", test_df, net, criterion, optimizer, 2000)
    loss_test.append(loss)
    accuracy_test.append(accuracy)

    end = timeit.default_timer()

    print(f"Epoch Time: {(end-start)/60:.2f}min")
    torch.save(net.state_dict(), "./data/model-state-dict.pt")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3))

    ax1.plot(loss_train[1:], label="Train")
    ax1.plot(loss_test[1:], label="Test")
    ax1.set_title("Model Convergence - Loss")
    ax1.set_xlabel("epochs")
    ax1.set_ylabel("Loss")
    ax1.legend()

    ax2.plot(accuracy_train, label="Train")
    ax2.plot(accuracy_test, label="Test")
    ax2.set_title("Model Convergence - Accuracy")
    ax2.set_xlabel("epochs")
    ax2.set_ylabel("Accuracy")
    ax2.legend()

    plt.savefig(f"./plots/plot_epoch{epoch}.png")

    plt.close(fig)

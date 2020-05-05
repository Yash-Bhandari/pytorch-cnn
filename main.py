import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from network import Network

import torchvision
import torchvision.transforms as transforms

train_set = torchvision.datasets.FashionMNIST(
    root='./data/FashionMNIST',
    train=True,
    download=True,
    transform=transforms.Compose([
        transforms.ToTensor()
    ])
)

def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()


train_loader = DataLoader(train_set, batch_size=100)

#initializing network
network = Network()
# optimizer updates weights
optimizer = optim.Adam(network.parameters(), lr=0.01)

# Training with a single batch
def train_one_batch(images, labels):
    preds = network(images)
    loss = F.cross_entropy(preds, labels)

    optimizer.zero_grad()  # Clears Gradient
    loss.backward()  # Calculate Gradients
    optimizer.step()  # Update Weights

    # returns batch statistics
    num_correct = get_num_correct(preds, labels)
    loss = loss.item()
    return num_correct, loss


def train_epoch():
    total_loss = 0
    total_correct = 0

    for images, labels in train_loader:
        batch_correct, batch_loss = train_one_batch(images, labels)
        total_correct += batch_correct
        total_loss += batch_loss
    return total_correct, total_loss


def summarize(epoch_stats, epoch_num=0):
    total_correct, total_loss = epoch_stats
    print(f'epoch #{epoch_num} num_correct: {total_correct}/{len(train_set)} = {total_correct/len(train_set)} loss: {total_loss} ')

# trains 5 epochs
for epoch in range(5):
    summarize(train_epoch(), epoch_num=epoch)


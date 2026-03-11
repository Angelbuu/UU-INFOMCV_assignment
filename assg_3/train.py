import torch
import torch.nn as nn


def train(model: nn.Module):
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

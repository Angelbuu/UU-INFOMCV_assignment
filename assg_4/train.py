import os
import torch
import torch.nn as nn
from torch.optim import Adam

from model import Model
from preprocessing import prepare_datasets

WEIGHT_COORD = 5
WEIGHT_NOOBJ = 0.5
GRID_SIZE = 7


def minibatch_forward_pass(model, minibatch):
    images, _, _, targets = minibatch
    outputs = model(images)
    outputs = outputs.view(-1, GRID_SIZE, GRID_SIZE, 7)

    obj_mask = targets[..., 4] == 1  # shape (B,7,7)
    noobj_mask = targets[..., 4] == 0

    coord_loss = WEIGHT_COORD * ((outputs[..., 0:2] - targets[..., 0:2]) ** 2 * obj_mask.unsqueeze(-1)).sum()

    pred_wh_sqrt = torch.sign(outputs[..., 2:4]) * torch.sqrt(torch.abs(outputs[..., 2:4]) + 1e-6)
    target_wh_sqrt = torch.sqrt(targets[..., 2:4])
    size_loss = WEIGHT_COORD * ((pred_wh_sqrt - target_wh_sqrt) ** 2 * obj_mask.unsqueeze(-1)).sum()

    obj_loss = ((outputs[..., 4] - targets[..., 4]) ** 2 * obj_mask).sum()

    noobj_loss = WEIGHT_NOOBJ * ((outputs[..., 4] - targets[..., 4]) ** 2 * noobj_mask).sum()

    class_loss = ((outputs[..., 5:] - targets[..., 5:]) ** 2 * obj_mask.unsqueeze(-1)).sum()

    loss = coord_loss + size_loss + obj_loss + noobj_loss + class_loss

    return loss, loss.item()


def validate(model, val_data):
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for minibatch in val_data:
            _, batch_loss = minibatch_forward_pass(model, minibatch)
            val_loss += batch_loss

    return val_loss


def train(model, train_data, val_data, optim=Adam, lr=0.001, max_epochs=50, patience=5):
    opt = optim(model.parameters(), lr=lr)

    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0

        for batch in train_data:
            opt.zero_grad()
            loss, bl = minibatch_forward_pass(model, batch)
            loss.backward()
            opt.step()
            epoch_loss += bl

        val_loss = validate(model, val_data)

        print(f'Epoch {epoch + 1}, train loss: {epoch_loss:.2f}, val loss: {val_loss:.2f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f'Early stop at epoch {epoch + 1}')
                break

    return model


def main():
    train_data, val_data, test_data = prepare_datasets(batch_size=32)
    model = Model()
    model = train(model, train_data, val_data, max_epochs=3)


if __name__ == '__main__':
    main()

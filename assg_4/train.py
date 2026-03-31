import os
import torch
import torch.nn as nn
from torch.optim import Adam

from model import Model
from preprocessing import prepare_datasets, GRID_SIZE
from graphs import plot_losses

WEIGHT_COORD = 5
WEIGHT_NOOBJ = 0.5


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

    return loss, {
        'total': loss.detach().item(),
        'coord': coord_loss.detach().item(),
        'size': size_loss.detach().item(),
        'obj': obj_loss.detach().item(),
        'noobj': noobj_loss.detach().item(),
        'class': class_loss.detach().item(),
    }


def validate(model, val_data):
    model.eval()
    val_loss = 0
    num_batches = 0

    with torch.no_grad():
        for minibatch in val_data:
            _, bl_components = minibatch_forward_pass(model, minibatch)
            val_loss += bl_components['total']
            num_batches += 1

    return val_loss / num_batches


def train(model, train_data, val_data, optim=Adam, lr=0.001, max_epochs=50, patience=5):
    opt = optim(model.parameters(), lr=lr)

    best_val_loss = float('inf')
    epochs_no_improve = 0

    history = {
        'train': {'total': [], 'coord': [], 'size': [], 'obj': [], 'noobj': [], 'class': []},
        'val': {'total': []}
    }

    for epoch in range(max_epochs):
        model.train()
        epoch_sums = {'total': 0, 'coord': 0, 'size': 0, 'obj': 0, 'noobj': 0, 'class': 0}
        num_batches = 0

        for batch in train_data:
            opt.zero_grad()
            loss, bl_components = minibatch_forward_pass(model, batch)
            loss.backward()
            opt.step()

            for k in epoch_sums:
                epoch_sums[k] += bl_components[k]
            num_batches += 1

        epoch_avg = {k: v / num_batches for k, v in epoch_sums.items()}
        val_loss = validate(model, val_data)

        for k in epoch_avg:
            history['train'][k].append(epoch_avg[k])
        history['val']['total'].append(val_loss)

        print(
            f'Epoch {epoch + 1} | '
            f'Train Loss: {epoch_avg["total"]:.2f} | '
            f'Val Loss: {val_loss:.2f}'
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f'Early stop at epoch {epoch + 1}')
                break

    return model, history


def main():
    os.makedirs('checkpoints', exist_ok=True)

    train_data, val_data, _ = prepare_datasets(batch_size=32)
    model = Model()
    model, history = train(model, train_data, val_data, max_epochs=30, patience=3)
    torch.save(model.state_dict(), 'checkpoints/yolo.pt')

    plot_losses(history)
    plot_losses(history, title='Train Loss Components', validation=False)


if __name__ == '__main__':
    main()

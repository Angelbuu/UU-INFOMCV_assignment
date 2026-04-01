import os
import torch
from torch.optim import Adam

from model import Model
from preprocessing import prepare_datasets, GRID_SIZE, ENTRIES_PER_GRID
from graphs import plot_losses

WEIGHT_COORD = 5
WEIGHT_NOOBJ = 0.5


def minibatch_forward_pass(model, minibatch):
    """Computes and returns yolo loss and its individual components from samples of a minibatch."""
    images, _, _, targets = minibatch
    outputs = model(images)
    outputs = outputs.view(-1, GRID_SIZE, GRID_SIZE, ENTRIES_PER_GRID)

    obj_mask = targets[..., 4] == 1
    noobj_mask = targets[..., 4] == 0

    coord_loss = WEIGHT_COORD * ((outputs[..., 0:2] - targets[..., 0:2]) ** 2 * obj_mask.unsqueeze(-1)).sum()

    pred_wh_sqrt = torch.sqrt(outputs[..., 2:4])
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
    """Computes batch-average loss and its components of a (validation) set."""
    model.eval()
    epoch_sums = {'total': 0, 'coord': 0, 'size': 0, 'obj': 0, 'noobj': 0, 'class': 0}
    num_batches = 0

    with torch.no_grad():
        for minibatch in val_data:
            _, bl_components = minibatch_forward_pass(model, minibatch)
            for k in epoch_sums:
                epoch_sums[k] += bl_components[k]
            num_batches += 1

    epoch_avg = {k: v / num_batches for k, v in epoch_sums.items()}
    return epoch_avg


def train(model, train_data, val_data, optim=Adam, lr=0.001, max_epochs=50, patience=5):
    """
    Trains the model with early stopping on validation loss, returns batch-average train and validation losses
    (and their individual components) per epoch.
    """
    opt = optim(model.parameters(), lr=lr)

    best_val_loss = float('inf')
    epochs_no_improve = 0

    history = {
        'train': {'total': [], 'coord': [], 'size': [], 'obj': [], 'noobj': [], 'class': []},
        'val': {'total': [], 'coord': [], 'size': [], 'obj': [], 'noobj': [], 'class': []}
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
        val_components = validate(model, val_data)

        for k in epoch_avg:
            history['train'][k].append(epoch_avg[k])
            history['val'][k].append(val_components[k])

        val_loss = val_components["total"]

        print(f'Epoch {epoch + 1} | Train Loss: {epoch_avg["total"]:.2f} | Val Loss: {val_loss:.2f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f'Early stop at epoch {epoch + 1}')
                break

    return model, history


def main(augment_data=False):
    """Trains the model, saves parameters and plots train and validation losses over epochs."""
    os.makedirs('checkpoints', exist_ok=True)

    train_data, val_data, _ = prepare_datasets(batch_size=32, augment_train=augment_data)
    model = Model()
    model, history = train(model, train_data, val_data)
    torch.save(model.state_dict(), 'checkpoints/yolo.pt')

    for key in history['train'].keys():
        plot_losses(history, key)


if __name__ == '__main__':
    main()

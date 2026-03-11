import torch
import torch.nn as nn
from models import LeNet, init_model
from import_data import load_data


def train(model: nn.Module, train_data, val_data, epochs):
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    print('Training', type(model))
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    for epoch in range(epochs):
        print(f'Epoch {epoch+1}')
        epoch_train_loss = 0

        correct = 0
        total = 0
        for minibatch in train_data:
            images, labels = minibatch
            optimizer.zero_grad()

            outputs = model(images)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()

            _, predictions = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predictions == labels).sum().item()

        print(f'Train loss: {epoch_train_loss}, accuracy {100 * correct / total}')
        epoch_val_loss, val_acc = validate(model, val_data)
        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        train_accuracies.append(100 * correct / total)
        val_accuracies.append(val_acc)

    return model, train_losses, train_accuracies, val_losses, val_accuracies


def validate(model: nn.Module, val_data):
    loss_function = nn.CrossEntropyLoss()
    correct = 0
    total = 0
    val_loss = 0
    with torch.no_grad():
        for minibatch in val_data:
            images, labels = minibatch
            outputs = model(images)
            _, predictions = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predictions == labels).sum().item()
            val_loss += loss_function(outputs, labels).item()

    print(f'Validation loss: {val_loss}, accuracy {100 * correct / total}')
    return val_loss, 100 * correct / total


def main():
    train_data, val_data, _ = load_data(val_set_ratio=0.15)
    baseline = init_model(LeNet)
    train(baseline, train_data, val_data, 3)


if __name__ == '__main__':
    main()

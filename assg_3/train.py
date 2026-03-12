import torch
import torch.nn as nn
from torch.optim import Adam

from models import init_model, LeNet, LeNetVariant1, LeNetVariant2, CIFAR100Model
from import_data import load_data


def minibatch_forward_pass(model, minibatch, loss_function):
    images, labels = minibatch
    outputs = model(images)

    loss = loss_function(outputs, labels)
    predictions = outputs.argmax(dim=1)

    batch_loss = loss.item()
    batch_size = labels.size(0)
    batch_correct = (predictions == labels).sum().item()

    return loss, batch_loss, batch_size, batch_correct


def validate(model: nn.Module, val_data):
    model.eval()
    loss_function = nn.CrossEntropyLoss()

    val_loss = 0
    num_labels = 0
    correct_predictions = 0

    with torch.no_grad():
        for minibatch in val_data:
            _, batch_loss, batch_size, batch_correct = minibatch_forward_pass(model, minibatch, loss_function)

            val_loss += batch_loss
            num_labels += batch_size
            correct_predictions += batch_correct

    accuracy = 100 * correct_predictions / num_labels
    return val_loss, accuracy


def train(model: nn.Module, train_data, val_data, epochs, optim=Adam, lr=0.001):
    print('\nTraining', type(model).__name__)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim(model.parameters(), lr=lr)

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(epochs):
        print(f'Epoch {epoch+1}')
        model.train()

        epoch_train_loss = 0
        num_labels = 0
        correct_predictions = 0

        for minibatch in train_data:
            optimizer.zero_grad()

            loss, batch_loss, batch_size, batch_correct = minibatch_forward_pass(model, minibatch, loss_function)

            loss.backward()
            optimizer.step()

            epoch_train_loss += batch_loss
            num_labels += batch_size
            correct_predictions += batch_correct

        train_acc = 100 * correct_predictions / num_labels
        epoch_val_loss, val_acc = validate(model, val_data)

        print(f'Train loss: {epoch_train_loss}, accuracy {train_acc}')
        print(f'Validation loss: {epoch_val_loss}, accuracy {val_acc}')

        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

    return model, train_losses, train_accuracies, val_losses, val_accuracies


def main():
    cif_ten_train_data, cif_ten_val_data, test_data = load_data(val_set_ratio=0.15)
    cif_hundred_train_data, cif_hundred_val_data, _ = load_data(dataset='CIF_HUNDRED', val_set_ratio=0.15)

    lenet = init_model(LeNet)
    lenet_v1 = init_model(LeNetVariant1)
    lenet_v2 = init_model(LeNetVariant2)

    # TODO: choose better hyperparameters
    train(lenet, cif_ten_train_data, cif_ten_val_data, 3)
    train(lenet_v1, cif_ten_train_data, cif_ten_val_data, 3)
    train(lenet_v2, cif_ten_train_data, cif_ten_val_data, 3)

    cif_100_model = init_model(CIFAR100Model)
    # SAME HYPERPARAMS AS FOR THE BEST MODEL ON VAL SET, TRAIN UNTIL CONVERGENCE
    train(cif_100_model, cif_hundred_train_data, cif_hundred_val_data, 3)

    cif_10_pretrained = cif_100_model
    cif_10_pretrained.fc_2 = nn.Linear(84, 10)
    # Fine-tuning, LEARNING RATE SHOULD BE HALF OF THE PREVIOUS ONE USED
    train(cif_10_pretrained, cif_ten_train_data, cif_ten_val_data, 3, lr=0.001/2)


if __name__ == '__main__':
    main()

import numpy as np
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold

from import_data import load_data
from models import init_model, LeNet
from train import train


def main():
    """
    Trains LeNet on 5-fold cross-validation, determines the optimal number of epochs and re-trains on
    full training set. Report accuracy for test set.
    """
    train_data, test_data = load_data(cv=True)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    kf = KFold(n_splits=5, shuffle=True)

    all_val_accuracies = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(train_data)):
        print(f"Fold {fold + 1}")

        train_subset = Subset(train_data, train_idx)
        val_subset = Subset(train_data, val_idx)

        train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)

        lenet = init_model(LeNet)
        model, tl, ta, vl, va, _ = train(lenet, train_loader, val_loader, max_epochs=20, patience=100)

        all_val_accuracies.append(va)

    all_val_accuracies = np.array(all_val_accuracies)
    avg_val_accuracy = np.mean(all_val_accuracies, axis=0)
    best_epoch = np.argmax(avg_val_accuracy) + 1

    print(f"Best epoch on average: {best_epoch}, Avg val accuracy: {avg_val_accuracy[best_epoch - 1]:.4f}")

    full_train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    final_model = init_model(LeNet)

    final_model, _, _, _, test_acc, _ = train(final_model, full_train_loader, test_loader, max_epochs=best_epoch,
                                              patience=100)

    print(f"Test accuracy: {test_acc[-1]:.4f}")


if __name__ == '__main__':
    main()

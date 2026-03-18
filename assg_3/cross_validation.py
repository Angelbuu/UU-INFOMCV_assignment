import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold

from import_data import load_data
from models import init_model, LeNet
from train import train


def main():
    train_data, test_data = load_data(cv=True)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    kf = KFold(n_splits=5, shuffle=True)

    for fold, (train_idx, val_idx) in enumerate(kf.split(train_data)):
        print(f"Fold {fold + 1}")

        train_subset = Subset(train_data, train_idx)
        val_subset = Subset(train_data, val_idx)

        train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)

        # Now train your model with train_loader, evaluate with val_loader
        lenet, tl, ta, vl, va, _ = train(init_model(LeNet), train_loader, val_loader)


if __name__ == '__main__':
    main()

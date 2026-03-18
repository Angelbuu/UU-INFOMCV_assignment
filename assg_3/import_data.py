import os
import pickle
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10 as CIF_TEN, CIFAR100 as CIF_HUNDRED
from torchvision.transforms import ToTensor
from sklearn.model_selection import train_test_split

CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]


class CIFAR100Coarse(CIF_HUNDRED):
    """CIFAR 100 dataset adapted to use 20 superclass labels instead of 100 subclass labels."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        file = self.train_list[0][0] if self.train else self.test_list[0][0]
        path = os.path.join(self.root, self.base_folder, file)

        with open(path, "rb") as f:
            entry = pickle.load(f, encoding="latin1")

        self.targets = entry["coarse_labels"]


def train_validation_split(dataset, val_set_ratio):
    labels = dataset.targets
    indices = list(range(len(dataset)))
    train_idx, val_idx = train_test_split(indices, test_size=val_set_ratio, stratify=labels)

    train_data = Subset(dataset, train_idx)
    validation_data = Subset(dataset, val_idx)
    return train_data, validation_data


def load_data(dataset='CIF_TEN', val_set_ratio=0.15, batch_size=32, cv=False):
    if dataset == 'CIF_TEN':
        dataset = CIF_TEN
    elif dataset == 'CIF_HUNDRED':
        dataset = CIFAR100Coarse
    else:
        raise ValueError('Unknown dataset')

    all_train_data = dataset(root='data', download=True, transform=ToTensor())
    test_data = dataset(root='data', download=True, transform=ToTensor(), train=False)

    if cv:
        return all_train_data, test_data

    print('Loading dataset', type(all_train_data))
    print('All train samples', len(all_train_data))
    print('Test samples', len(test_data))

    train_data, validation_data = train_validation_split(all_train_data, val_set_ratio)
    print('Train samples', len(train_data))
    print('Validation samples', len(validation_data))

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    validation_dataloader = DataLoader(validation_data, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_dataloader, validation_dataloader, test_dataloader


if __name__ == '__main__':
    load_data(val_set_ratio=0.15)
    load_data('CIF_HUNDRED', val_set_ratio=0.15)

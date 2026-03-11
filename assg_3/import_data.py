from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10 as CIF_TEN, CIFAR100 as CIF_HUNDRED
from torchvision.transforms import ToTensor
from sklearn.model_selection import train_test_split


def train_validation_split(dataset, val_set_ratio):
    labels = dataset.targets
    indices = list(range(len(dataset)))
    train_idx, val_idx = train_test_split(indices, test_size=val_set_ratio, stratify=labels)

    train_data = Subset(dataset, train_idx)
    validation_data = Subset(dataset, val_idx)
    return train_data, validation_data


def load_data(dataset='CIF_TEN', val_set_ratio=0.1, batch_size=32):
    if dataset == 'CIF_TEN':
        dataset = CIF_TEN
    elif dataset == 'CIF_HUNDRED':
        dataset = CIF_HUNDRED
    else:
        raise ValueError('Unknown dataset')

    all_train_data = dataset(root='data', download=True, transform=ToTensor())
    test_data = dataset(root='data', download=True, transform=ToTensor(), train=False)
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

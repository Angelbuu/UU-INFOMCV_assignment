import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from models import init_model, LeNet
from import_data import load_data, CIFAR10_CLASSES


def get_embeddings(model, test_data):
    """Collects the embeddings and labels of test data from the second the last dense layer."""
    model.eval()
    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in test_data:
            embeddings = model.get_embeddings(imgs)
            all_embeddings.append(embeddings.cpu())
            all_labels.append(labels)

    all_embeddings = torch.cat(all_embeddings).numpy()
    all_labels = torch.cat(all_labels).numpy()

    return all_embeddings, all_labels


def encode_tsne(embeddings):
    """Performs t-SNE."""
    tsne = TSNE(2)
    return tsne.fit_transform(embeddings)


def plot_tsne(embeddings_2d, labels):
    """Plots scatterplot of the t-SNE with corresponding labels."""
    plt.figure(figsize=(8, 6))

    for class_idx, class_name in enumerate(CIFAR10_CLASSES):
        indices = labels == class_idx
        plt.scatter(
            embeddings_2d[indices, 0],
            embeddings_2d[indices, 1],
            label=class_name,
            s=10
        )

    plt.legend(markerscale=2, fontsize=8)
    plt.title("t-SNE of FC Layer (Test Set)")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.tight_layout()
    plt.savefig("tsne.png")


def main():
    """Loads a trained LeNet model visualizes the t-SNE embeddings of test data."""
    _, _, test_data = load_data(val_set_ratio=0.15)

    model = init_model(LeNet)
    model.load_state_dict(torch.load('checkpoints/CIFAR10_lenet.pt'))
    model.eval()

    embeddings, labels = get_embeddings(model, test_data)
    embeddings_2d = encode_tsne(embeddings)
    plot_tsne(embeddings_2d, labels)


if __name__ == '__main__':
    main()

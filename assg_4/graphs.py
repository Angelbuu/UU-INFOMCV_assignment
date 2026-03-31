import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.metrics import confusion_matrix

CLASS_NAMES = ["cat", "dog"]


def plot_losses(history, title='Training vs Validation', validation=True):
    """Plots losses over epochs."""
    epochs = range(1, len(history['train']['total']) + 1)
    plt.figure()

    if validation:
        plt.plot(epochs, history['train']['total'], label='Train')
        plt.plot(epochs, history['val']['total'], label='Validation')
    else:
        for k in ['coord', 'size', 'obj', 'noobj', 'class']:
            plt.plot(epochs, history['train'][k], label=k)

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.grid(True)

    plt.savefig(f'{title}.png')


def plot_confusion_matrix(confusion, class_names=None, title="Confusion Matrix"):
    """
    confusion: tensor of shape (num_classes+1, num_classes+1)
    """
    confusion = confusion.numpy()

    if class_names is None:
        class_names = ["Cat", "Dog"]

    # Add background label
    labels = class_names + ["Background"]

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(confusion)

    # Show values inside cells
    for i in range(confusion.shape[0]):
        for j in range(confusion.shape[1]):
            ax.text(
                j, i, int(confusion[i, j]),
                ha="center", va="center",
                color="white" if confusion[i, j] > confusion.max() / 2 else "black"
            )

    # Axis labels
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))

    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)

    plt.colorbar(im)
    plt.tight_layout()
    plt.show()

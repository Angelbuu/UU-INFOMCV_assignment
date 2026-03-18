import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from import_data import CIFAR10_CLASSES


def plot_metrics(train_values, val_values, ylabel="Loss", title="Training vs Validation"):
    epochs = range(1, len(train_values) + 1)

    plt.figure()
    plt.plot(epochs, train_values, label=f"Train {ylabel}")
    plt.plot(epochs, val_values, label=f"Validation {ylabel}")

    plt.xlabel("Epochs")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)

    plt.show()


def plot_confusion_matrix(labels, preds, title="Confusion Matrix"):
    cm = confusion_matrix(labels, preds)

    plt.figure()
    plt.imshow(cm)
    plt.title(title)
    plt.colorbar()

    ticks = np.arange(len(CIFAR10_CLASSES))
    plt.xticks(ticks, CIFAR10_CLASSES, rotation=45)
    plt.yticks(ticks, CIFAR10_CLASSES)

    plt.xlabel("Predicted label")
    plt.ylabel("True label")

    # Add numbers inside the matrix
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j],
                     ha="center", va="center")

    plt.tight_layout()
    plt.show()

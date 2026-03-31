import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


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

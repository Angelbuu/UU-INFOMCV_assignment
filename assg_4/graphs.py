import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

CLASS_NAMES = ['cat', 'dog']


def plot_losses(history, component: str, y_label='Loss', title='Training vs Validation'):
    """Plots losses over epochs."""
    epochs = range(1, len(history['train']['total']) + 1)
    plt.figure()

    plt.plot(epochs, history['train'][component], label='Train')
    plt.plot(epochs, history['val'][component], label='Validation')

    y_label = component.capitalize() + ' ' + y_label
    title = title + ' ' + y_label
    plt.xlabel('Epochs')
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.grid(True)

    plt.savefig(f'{title}.png')


def plot_confusion_matrix(confusion, title="Confusion Matrix"):
    """Plots a heatmap confusion matrix for cats and dogs detection."""
    confusion = confusion.numpy()

    labels = CLASS_NAMES + ["Background"]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(confusion)

    for i in range(confusion.shape[0]):
        for j in range(confusion.shape[1]):
            ax.text(
                j, i, int(confusion[i, j]),
                ha="center", va="center",
                color="black" if confusion[i, j] > confusion.max() / 2 else "white"
            )

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))

    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)

    plt.tight_layout()
    plt.show()


def visualize_predictions(model, dataloader, threshold=0.5):
    """Displays a batch of images with their ground truth and predicted labels and bounding boxes."""
    model.eval()
    images, bboxes_list, labels_list, _ = next(iter(dataloader))

    with torch.no_grad():
        preds = model.predict(images, threshold)

    fig, axes = plt.subplots(1, len(images), figsize=(15, 5))

    for i in range(len(images)):
        img = images[i].cpu().permute(1, 2, 0).numpy()
        axes[i].imshow(img)

        for box, lbl in zip(bboxes_list[i], labels_list[i]):
            xmin, ymin, xmax, ymax = box.tolist()

            rect = patches.Rectangle(
                (xmin, ymin), xmax - xmin, ymax - ymin,
                linewidth=2, edgecolor='red', facecolor='none'
            )
            axes[i].add_patch(rect)

            axes[i].text(
                xmin, ymin - 5,
                f"GT: {CLASS_NAMES[lbl.item()]}",
                color='red',
                fontsize=10,
                bbox=dict(facecolor='white', alpha=0.5)
            )

        pred = preds[i]

        for box, score, lbl in zip(pred["boxes"], pred["scores"], pred["labels"]):
            h, w, _ = img.shape
            xmin, ymin, xmax, ymax = box.tolist()

            xmin *= w
            xmax *= w
            ymin *= h
            ymax *= h

            rect = patches.Rectangle(
                (xmin, ymin), xmax - xmin, ymax - ymin,
                linewidth=2, edgecolor='green', facecolor='none'
            )
            axes[i].add_patch(rect)

            axes[i].text(
                xmin, ymax + 5,
                f"Pred: {CLASS_NAMES[lbl.item()]} ({score:.2f})",
                color='green',
                fontsize=10,
                bbox=dict(facecolor='white', alpha=0.5)
            )

        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

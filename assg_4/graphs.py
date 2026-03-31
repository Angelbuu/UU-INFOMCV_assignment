import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

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
                color="black" if confusion[i, j] > confusion.max() / 2 else "white"
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


def visualize_predictions(model, dataloader, threshold=0.5, device="cpu"):
    model.eval()

    images, bboxes_list, labels_list, _ = next(iter(dataloader))
    images = images.to(device)

    with torch.no_grad():
        preds = model.predict(images, threshold)

    fig, axes = plt.subplots(1, len(images), figsize=(15, 5))

    if len(images) == 1:
        axes = [axes]

    for i in range(len(images)):
        img = images[i].cpu().permute(1, 2, 0).numpy()
        axes[i].imshow(img)

        # --- Ground truth (RED) ---
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

        # --- Predictions (GREEN) ---
        pred = preds[i]

        for box, score, lbl in zip(pred["boxes"], pred["scores"], pred["labels"]):
            h, w, _ = img.shape
            xmin, ymin, xmax, ymax = box.tolist()

            # scale to image size
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

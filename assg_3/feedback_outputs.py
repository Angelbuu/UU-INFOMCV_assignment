import torch
import matplotlib.pyplot as plt

from models import init_model, LeNet
from train import train, evaluate
from import_data import load_data, CIFAR10_CLASSES


def visualize_outputs(model, dataloader, class_names, num_images=4):
    model.eval()

    images, labels = next(iter(dataloader))
    images = images[:num_images]
    labels = labels[:num_images]

    with torch.no_grad():
        final_out, fb1, fb2 = model(images, use_feedback=True)

    final_preds = final_out.argmax(dim=1)
    fb1_preds = fb1.argmax(dim=1)
    fb2_preds = fb2.argmax(dim=1)

    fig, axes = plt.subplots(1, num_images, figsize=(15, 3))

    for i in range(num_images):
        img = images[i].permute(1, 2, 0).cpu().numpy()

        axes[i].imshow(img)
        axes[i].axis('off')

        axes[i].set_title(
            f"GT: {class_names[labels[i]]}\n"
            f"FB1: {class_names[fb1_preds[i]]}\n"
            f"FB2: {class_names[fb2_preds[i]]}\n"
            f"Final: {class_names[final_preds[i]]}"
        )

    plt.tight_layout()
    plt.savefig(f"fb.png")


def main():
    cif_ten_train, cif_ten_val, test_data = load_data(val_set_ratio=0.15)
    lenet, tl, ta, vl, va, _ = train(init_model(LeNet), cif_ten_train, cif_ten_val, use_feedback=True)
    print(f'Test accuracy: {evaluate(lenet, test_data)[0]}')
    visualize_outputs(lenet, test_data, CIFAR10_CLASSES)


if __name__ == '__main__':
    main()

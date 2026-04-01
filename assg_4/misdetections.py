"""
CHOICE 7: save a few examples of cat<->dog swaps, false positives, false negatives on val.
Run after training: python misdetections.py --threshold 0.35
"""
import os
import argparse
import torch
import matplotlib.pyplot as plt

from model import Model
from preprocessing import prepare_datasets, INPUT_IMG_SIZE
from compute_metrics import match_predictions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--threshold', type=float, default=0.35, help='match evaluate.py best F1 threshold if you can')
    args = ap.parse_args()
    thr = args.threshold

    os.makedirs('misdetection_samples', exist_ok=True)
    _, val_data, _ = prepare_datasets(batch_size=1)
    model = Model()
    model.load_state_dict(torch.load('checkpoints/yolo.pt', map_location='cpu'))
    model.eval()

    # true cat -> pred dog, true dog -> pred cat
    cat_to_dog = []
    dog_to_cat = []
    false_pos = []
    false_neg = []

    with torch.no_grad():
        for images, bboxes_list, labels_list, _ in val_data:
            pred = model.predict(images, thr, apply_nms=True)[0]
            gt_boxes = bboxes_list[0]
            gt_labels = labels_list[0]

            if len(gt_boxes) == 0:
                continue
            boxes_gt = gt_boxes / INPUT_IMG_SIZE
            target = {
                'boxes': boxes_gt,
                'labels': gt_labels,
            }

            matches, unmatched_preds, unmatched_gts = match_predictions(pred, target)

            for p_lbl, g_lbl in matches:
                if g_lbl == 0 and p_lbl == 1:
                    cat_to_dog.append(images[0].clone())
                elif g_lbl == 1 and p_lbl == 0:
                    dog_to_cat.append(images[0].clone())

            for _ in unmatched_preds:
                false_pos.append(images[0].clone())
            for _ in unmatched_gts:
                false_neg.append(images[0].clone())

            if (
                len(cat_to_dog) >= 2
                and len(dog_to_cat) >= 2
                and len(false_pos) >= 2
                and len(false_neg) >= 2
            ):
                break

    def save_grid(tensors, title, fname):
        if not tensors:
            return
        n = min(4, len(tensors))
        fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
        if n == 1:
            axes = [axes]
        for i in range(n):
            img = tensors[i].cpu().permute(1, 2, 0).numpy()
            img = img.clip(0, 1)
            axes[i].imshow(img)
            axes[i].set_title(title)
            axes[i].axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join('misdetection_samples', fname), dpi=120)
        plt.close()
        print('wrote', fname)

    save_grid(cat_to_dog, 'GT cat, pred dog', 'cat_to_dog.png')
    save_grid(dog_to_cat, 'GT dog, pred cat', 'dog_to_cat.png')
    save_grid(false_pos, 'unmatched pred (FP-ish)', 'false_positives.png')
    save_grid(false_neg, 'missed GT (FN-ish)', 'false_negatives.png')
    print('done — check assg_4/misdetection_samples/')


if __name__ == '__main__':
    main()

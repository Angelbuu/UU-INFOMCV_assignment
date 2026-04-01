import os
import torch

from model import Model
from preprocessing import prepare_datasets, INPUT_IMG_SIZE
from compute_metrics import match_predictions
from graphs import visualize_single_prediction

MAX_SAMPLES = 4


def main():
    """Saves visualisations of misclassifications and misdetections of the model."""
    os.makedirs('misdetection_samples', exist_ok=True)

    _, _, test_data = prepare_datasets(batch_size=1)

    model = Model()
    model.load_state_dict(torch.load('checkpoints/yolo.pt'))
    model.eval()

    counters = {
        "cat_to_dog": 0,
        "dog_to_cat": 0,
        "false_pos": 0,
        "false_neg": 0
    }

    with torch.no_grad():
        for images, bboxes_list, labels_list, _ in test_data:
            image = images[0]

            pred = model.predict(images)[0]

            gt_boxes = bboxes_list[0]
            gt_labels = labels_list[0]

            if len(gt_boxes) == 0:
                continue

            boxes_gt = gt_boxes / INPUT_IMG_SIZE
            target = {
                "boxes": boxes_gt,
                "labels": gt_labels,
            }

            matches, unmatched_preds, unmatched_gts = match_predictions(pred, target)

            for p_lbl, g_lbl in matches:
                if g_lbl == 0 and p_lbl == 1 and counters["cat_to_dog"] < MAX_SAMPLES:
                    visualize_single_prediction(
                        image, gt_boxes, gt_labels, pred,
                        f"misdetection_samples/cat_to_dog_{counters['cat_to_dog']}.png"
                    )
                    counters["cat_to_dog"] += 1

                elif g_lbl == 1 and p_lbl == 0 and counters["dog_to_cat"] < MAX_SAMPLES:
                    visualize_single_prediction(
                        image, gt_boxes, gt_labels, pred,
                        f"misdetection_samples/dog_to_cat_{counters['dog_to_cat']}.png"
                    )
                    counters["dog_to_cat"] += 1

            if unmatched_preds and counters["false_pos"] < MAX_SAMPLES:
                visualize_single_prediction(
                    image, gt_boxes, gt_labels, pred,
                    f"misdetection_samples/false_pos_{counters['false_pos']}.png"
                )
                counters["false_pos"] += 1

            if unmatched_gts and counters["false_neg"] < MAX_SAMPLES:
                visualize_single_prediction(
                    image, gt_boxes, gt_labels, pred,
                    f"misdetection_samples/false_neg_{counters['false_neg']}.png"
                )
                counters["false_neg"] += 1

            if all(v >= MAX_SAMPLES for v in counters.values()):
                break


if __name__ == '__main__':
    main()

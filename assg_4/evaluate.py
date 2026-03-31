import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from model import Model
from preprocessing import prepare_datasets, GRID_SIZE, INPUT_IMG_SIZE
from graphs import plot_confusion_matrix

CLASS_NAMES = ["cat", "dog"]


def decode_predictions(output, threshold, grid_size=GRID_SIZE):
    """
    output: (7, 7, 7) tensor for ONE image
    returns: dict with boxes, scores, labels
    """

    boxes = []
    scores = []
    labels = []

    for i in range(grid_size):
        for j in range(grid_size):
            cell = output[i, j]

            conf = cell[4].item()

            # Apply objectness threshold
            if conf < threshold:
                continue

            x, y, w, h = cell[0:4]
            class_probs = cell[5:]

            cls = torch.argmax(class_probs).item()
            cls_score = class_probs[cls].item()

            score = conf * cls_score  # YOLO confidence

            # Convert to image-relative coordinates
            x_center = (j + x.item()) / grid_size
            y_center = (i + y.item()) / grid_size

            xmin = x_center - w.item() / 2
            ymin = y_center - h.item() / 2
            xmax = x_center + w.item() / 2
            ymax = y_center + h.item() / 2

            # Clamp to [0, 1]
            xmin = max(0, xmin)
            ymin = max(0, ymin)
            xmax = min(1, xmax)
            ymax = min(1, ymax)

            boxes.append([xmin, ymin, xmax, ymax])
            scores.append(score)
            labels.append(cls)

    if len(boxes) == 0:
        return {
            "boxes": torch.zeros((0, 4)),
            "scores": torch.zeros((0,)),
            "labels": torch.zeros((0,), dtype=torch.int64),
        }

    return {
        "boxes": torch.tensor(boxes, dtype=torch.float32),
        "scores": torch.tensor(scores, dtype=torch.float32),
        "labels": torch.tensor(labels, dtype=torch.int64),
    }


def get_predictions(model, dataloader, threshold):
    model.eval()

    preds = []
    targets = []

    with torch.no_grad():
        for images, bboxes_list, labels_list, target_tensor in dataloader:
            outputs = model(images)
            outputs = outputs.view(-1, GRID_SIZE, GRID_SIZE, 7)

            for i in range(images.shape[0]):
                output = outputs[i].cpu()

                # --- Predictions ---
                pred_dict = decode_predictions(output, threshold)
                preds.append(pred_dict)

                # --- Targets ---
                bboxes = bboxes_list[i]  # (N, 4)
                labels = labels_list[i]  # (N,)

                # Normalize GT boxes to [0,1]
                img_size = INPUT_IMG_SIZE

                if len(bboxes) == 0:
                    target_dict = {
                        "boxes": torch.zeros((0, 4)),
                        "labels": torch.zeros((0,), dtype=torch.int64),
                    }
                else:
                    boxes = bboxes / img_size  # normalize
                    target_dict = {
                        "boxes": boxes,
                        "labels": labels,
                    }

                targets.append(target_dict)

    return preds, targets


def compute_iou(box1, box2):
    """
    box: [xmin, ymin, xmax, ymax]
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = area1 + area2 - inter + 1e-6

    return inter / union


def compute_map(preds, targets):
    metric = MeanAveragePrecision()

    metric.update(preds, targets)
    result = metric.compute()

    return result["map"].item()


def compute_f1(preds, targets, iou_threshold=0.5, num_classes=2):
    TP = [0] * num_classes
    FP = [0] * num_classes
    FN = [0] * num_classes

    for pred, target in zip(preds, targets):
        pred_boxes = pred["boxes"]
        pred_labels = pred["labels"]

        gt_boxes = target["boxes"]
        gt_labels = target["labels"]

        matched_gt = set()

        # --- Match predictions ---
        for p_box, p_label in zip(pred_boxes, pred_labels):
            best_iou = 0
            best_gt_idx = -1

            for i, (g_box, g_label) in enumerate(zip(gt_boxes, gt_labels)):
                if i in matched_gt:
                    continue
                if p_label != g_label:
                    continue

                iou = compute_iou(p_box.tolist(), g_box.tolist())

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = i

            if best_iou >= iou_threshold:
                TP[p_label] += 1
                matched_gt.add(best_gt_idx)
            else:
                FP[p_label] += 1

        # --- Count FN ---
        for i, g_label in enumerate(gt_labels):
            if i not in matched_gt:
                FN[g_label] += 1

    # --- Compute F1 per class ---
    f1_scores = []

    for c in range(num_classes):
        precision = TP[c] / (TP[c] + FP[c] + 1e-6)
        recall = TP[c] / (TP[c] + FN[c] + 1e-6)

        f1 = 2 * precision * recall / (precision + recall + 1e-6)
        f1_scores.append(f1)

    avg_f1 = sum(f1_scores) / num_classes

    return avg_f1, f1_scores


def compute_confusion_matrix(preds, targets, iou_threshold=0.5, num_classes=2):
    # +1 for background
    confusion = torch.zeros((num_classes + 1, num_classes + 1), dtype=torch.int32)

    for pred, target in zip(preds, targets):
        pred_boxes = pred["boxes"]
        pred_labels = pred["labels"]

        gt_boxes = target["boxes"]
        gt_labels = target["labels"]

        matched_gt = set()

        # --- Match predictions ---
        for p_box, p_label in zip(pred_boxes, pred_labels):
            best_iou = 0
            best_gt_idx = -1

            for i, (g_box, g_label) in enumerate(zip(gt_boxes, gt_labels)):
                if i in matched_gt:
                    continue

                iou = compute_iou(p_box.tolist(), g_box.tolist())

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = i

            if best_iou >= iou_threshold:
                true_label = gt_labels[best_gt_idx].item()
                pred_label = p_label.item()

                confusion[true_label, pred_label] += 1
                matched_gt.add(best_gt_idx)
            else:
                # False Positive → predicted something that doesn't exist
                confusion[num_classes, p_label.item()] += 1

        # --- Count False Negatives ---
        for i, g_label in enumerate(gt_labels):
            if i not in matched_gt:
                confusion[g_label.item(), num_classes] += 1

    return confusion


def evaluate_thresholds(model, dataloader, thresholds):
    results = []

    for t in thresholds:
        preds, targets = get_predictions(model, dataloader, t)

        mAP = compute_map(preds, targets)
        f1_avg, f1_per_class = compute_f1(preds, targets)

        results.append({
            "threshold": t,
            "mAP": mAP,
            "F1": f1_avg,
            "F1_per_class": f1_per_class
        })

    return results


def visualize_predictions(model, dataloader, threshold=0.5, device="cpu"):
    model.eval()

    images, bboxes_list, labels_list, _ = next(iter(dataloader))
    images = images.to(device)

    with torch.no_grad():
        outputs = model(images)
        outputs = outputs.view(-1, 7, 7, 7)

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
        pred = decode_predictions(outputs[i].cpu(), threshold)

        for box, score, lbl in zip(pred["boxes"], pred["scores"], pred["labels"]):
            # convert from normalized [0,1] → image pixels
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


def small_test(model, val_data):
    preds, targets = get_predictions(model, val_data, threshold=0.5)

    print(preds[0])
    print(targets[0])


if __name__ == '__main__':
    train_data, val_data, test_data = prepare_datasets(batch_size=4)
    model = Model()
    model.load_state_dict(torch.load('checkpoints/yolo.pt'))
    small_test(model, val_data)

    results = evaluate_thresholds(model, val_data, torch.linspace(0, 1, 20))

    best = max(results, key=lambda x: x["F1"])
    best_threshold = best["threshold"]

    preds, targets = get_predictions(model, val_data, best_threshold)
    conf_matrix = compute_confusion_matrix(preds, targets)

    print('mAP:')
    for result in results:
        print('Threshold:', result['threshold'], 'mAP:', result['mAP'])
    plot_confusion_matrix(conf_matrix)

    visualize_predictions(model, val_data)

import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision


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

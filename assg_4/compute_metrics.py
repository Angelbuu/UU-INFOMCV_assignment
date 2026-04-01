import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision

IOU_THRESHOLD = 0.5


def compute_iou(box1, box2):
    """Computes intersection over union."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union


def compute_map(preds, targets):
    """Computes mAP@0.50."""
    metric = MeanAveragePrecision(iou_thresholds=[IOU_THRESHOLD])
    metric.update(preds, targets)
    result = metric.compute()
    return result['map'].item()


def match_predictions(pred, target):
    """
    For an image predictions and ground truths, finds matched pairs, unmatched predictions and unmatched ground truths.
    """
    pred_boxes = pred['boxes']
    pred_labels = pred['labels']

    gt_boxes = target['boxes']
    gt_labels = target['labels']

    matched_gt = set()

    matches = []
    unmatched_preds = []
    unmatched_gts = []

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

        if best_iou >= IOU_THRESHOLD:
            matches.append((p_label.item(), gt_labels[best_gt_idx].item()))
            matched_gt.add(best_gt_idx)
        else:
            unmatched_preds.append(p_label.item())

    for i, g_label in enumerate(gt_labels):
        if i not in matched_gt:
            unmatched_gts.append(g_label.item())

    return matches, unmatched_preds, unmatched_gts


def compute_f1(preds, targets, num_classes=2):
    """Computes average F1 over classes."""
    true_pos = [0] * num_classes
    false_pos = [0] * num_classes
    false_neg = [0] * num_classes

    for pred, target in zip(preds, targets):
        matches, unmatched_preds, unmatched_gts = match_predictions(pred, target)

        for p_label, g_label in matches:
            if p_label == g_label:
                true_pos[p_label] += 1
            else:
                false_pos[p_label] += 1
                false_neg[g_label] += 1

        for p_label in unmatched_preds:
            false_pos[p_label] += 1

        for g_label in unmatched_gts:
            false_neg[g_label] += 1

    f1_scores = []

    for c in range(num_classes):
        precision = 0 if true_pos[c] == 0 else true_pos[c] / (true_pos[c] + false_pos[c])
        recall = 0 if true_pos[c] == 0 else true_pos[c] / (true_pos[c] + false_neg[c])

        f1 = 0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        f1_scores.append(f1)

    return sum(f1_scores) / num_classes


def compute_confusion_matrix(preds, targets, num_classes=2):
    """Computes a confusion matrix."""
    confusion = torch.zeros((num_classes + 1, num_classes + 1), dtype=torch.int32)

    for pred, target in zip(preds, targets):
        matches, unmatched_preds, unmatched_gts = match_predictions(pred, target)

        for p_label, g_label in matches:
            confusion[g_label, p_label] += 1

        for p_label in unmatched_preds:
            confusion[num_classes, p_label] += 1

        for g_label in unmatched_gts:
            confusion[g_label, num_classes] += 1

    return confusion

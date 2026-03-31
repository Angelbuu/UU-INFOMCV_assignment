import torch

from model import Model
from preprocessing import prepare_datasets, INPUT_IMG_SIZE
from graphs import plot_confusion_matrix, visualize_predictions
from compute_metrics import compute_map, compute_f1, compute_confusion_matrix


def get_predictions(model, dataloader, threshold):
    """
    Returns a list of target dictionaries (bounding boxes, labels) and prediction dictionaries (bounding boxes,
    labels and confidence scores) for the entire dataset.
    """
    model.eval()
    preds = []
    targets = []

    with torch.no_grad():
        for images, bboxes_list, labels_list, _ in dataloader:
            batch_preds = model.predict(images, threshold)

            for i in range(len(images)):
                bboxes = bboxes_list[i]
                labels = labels_list[i]

                if len(bboxes) == 0:
                    target_dict = {
                        "boxes": torch.zeros((0, 4)),
                        "labels": torch.zeros((0,), dtype=torch.int64),
                    }
                else:
                    boxes = bboxes / INPUT_IMG_SIZE  # also normalize target bboxes to [0,1]
                    target_dict = {
                        "boxes": boxes,
                        "labels": labels,
                    }

                targets.append(target_dict)
                preds.append(batch_preds[i])

    return preds, targets


def evaluate_thresholds(model, dataloader, thresholds):
    """Performs dataset evaluation sweep over thresholds. Metrics include mAP and F1 score."""
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


def main():
    datasets = prepare_datasets(batch_size=32)
    model = Model()
    model.load_state_dict(torch.load('checkpoints/yolo.pt'))

    for dataset in datasets:
        thresholds = torch.linspace(0, 1, 3)
        results = evaluate_thresholds(model, dataset, thresholds)

        best = max(results, key=lambda x: x["F1"])
        best_threshold = best["threshold"]

        preds, targets = get_predictions(model, dataset, best_threshold)
        conf_matrix = compute_confusion_matrix(preds, targets)

        print('mAP:')
        for result in results:
            print('Threshold:', result['threshold'], 'mAP:', result['mAP'])
        plot_confusion_matrix(conf_matrix)

        visualize_predictions(model, dataset)


if __name__ == '__main__':
    main()

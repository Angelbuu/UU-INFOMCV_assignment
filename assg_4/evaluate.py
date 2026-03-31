import torch

from model import Model
from preprocessing import prepare_datasets, INPUT_IMG_SIZE
from graphs import plot_confusion_matrix, visualize_predictions
from compute_metrics import compute_map, compute_f1, compute_confusion_matrix


def get_predictions(model, dataloader, threshold, device="cpu"):
    model.eval()

    preds = []
    targets = []

    with torch.no_grad():
        for images, bboxes_list, labels_list, _ in dataloader:
            images = images.to(device)

            # --- Predictions (already decoded!) ---
            batch_preds = model.predict(images, threshold)

            for i in range(len(images)):
                preds.append(batch_preds[i])

                # --- Targets ---
                bboxes = bboxes_list[i]
                labels = labels_list[i]

                if len(bboxes) == 0:
                    target_dict = {
                        "boxes": torch.zeros((0, 4)),
                        "labels": torch.zeros((0,), dtype=torch.int64),
                    }
                else:
                    # normalize to [0,1]
                    boxes = bboxes / INPUT_IMG_SIZE
                    target_dict = {
                        "boxes": boxes,
                        "labels": labels,
                    }

                targets.append(target_dict)

    return preds, targets


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


def small_test(model, val_data):
    preds, targets = get_predictions(model, val_data, threshold=0.5)

    print(preds[0])
    print(targets[0])


if __name__ == '__main__':
    datasets = prepare_datasets(batch_size=32)
    model = Model()
    model.load_state_dict(torch.load('checkpoints/yolo.pt'))

    for dataset in datasets:
        small_test(model, dataset)

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

import torch

from model import Model
from preprocessing import prepare_datasets, INPUT_IMG_SIZE
from graphs import plot_confusion_matrix, visualize_predictions
from compute_metrics import compute_map, compute_f1, compute_confusion_matrix


def get_predictions(model, dataloader, threshold, apply_nms=False):
    """
    Returns a list of target dictionaries (bounding boxes, labels) and prediction dictionaries (bounding boxes,
    labels and confidence scores) for the entire dataset.
    """
    model.eval()
    preds = []
    targets = []

    with torch.no_grad():
        for images, bboxes_list, labels_list, _ in dataloader:
            batch_preds = model.predict(images, threshold, apply_nms=apply_nms)

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

    for threshold in thresholds:
        preds, targets = get_predictions(model, dataloader, threshold)

        m_ap = compute_map(preds, targets)
        f1 = compute_f1(preds, targets)

        results.append({
            "threshold": threshold,
            "mAP": m_ap,
            "F1": f1,
        })

    return results


def main():
    """
    Evaluates each dataset at different thresholds and reports threshold for best F1 and corresponding mAP.
    For test set, plots confusion matrix and some test samples with predictions.
    """
    dataset_names = ['Train set', 'Val set', 'Test set']
    datasets = prepare_datasets(batch_size=4)
    model = Model()
    model.load_state_dict(torch.load('checkpoints/yolo.pt'))

    for idx, dataset in enumerate(datasets):
        print('Evaluating:', dataset_names[idx])
        thresholds = torch.linspace(0, 1, 20)
        results = evaluate_thresholds(model, dataset, thresholds)

        best = max(results, key=lambda x: x['F1'])
        best_threshold = best['threshold']
        print('Best F1 at threshold:', best_threshold)
        print('mAP at that threshold:', best['mAP'])

        if dataset_names[idx] == 'Test set':
            preds, targets = get_predictions(model, dataset, best_threshold)
            conf_matrix = compute_confusion_matrix(preds, targets)
            plot_confusion_matrix(conf_matrix)
            visualize_predictions(model, dataset, threshold=best_threshold)

            # choice task 8:
            preds_nms, _ = get_predictions(model, dataset, best_threshold, apply_nms=True)
            map_no = compute_map(preds, targets)
            map_yes = compute_map(preds_nms, targets)
            print('Same objectness threshold, mAP without NMS:', map_no, '| with NMS:', map_yes)
            plot_confusion_matrix(compute_confusion_matrix(preds, targets), title='Confusion matrix no NMS')
            plot_confusion_matrix(compute_confusion_matrix(preds_nms, targets), title='Confusion matrix with NMS')

        print()


if __name__ == '__main__':
    main()

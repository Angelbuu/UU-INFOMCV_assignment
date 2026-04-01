import torch
import torch.nn as nn
import torch.nn.functional as tf
from torchinfo import summary
from torchvision.ops import nms as tv_nms

from preprocessing import GRID_SIZE, ENTRIES_PER_GRID


def nms_single_image(boxes, scores, labels, iou_threshold=0.45):
    """Greedy per-class NMS; boxes xyxy in [0,1]."""
    if boxes.numel() == 0:
        return boxes, scores, labels
    keep_all = []
    for c in (0, 1):
        m = labels == c
        if not m.any():
            continue
        idx = torch.where(m)[0]
        b = boxes[idx]
        s = scores[idx]
        k = tv_nms(b, s, iou_threshold)
        keep_all.append(idx[k])
    if not keep_all:
        return boxes[:0], scores[:0], labels[:0]
    keep = torch.cat(keep_all)
    order = torch.argsort(scores[keep], descending=True)
    keep = keep[order]
    return boxes[keep], scores[keep], labels[keep]


class Model(nn.Module):
    """Mini-yolo model class."""
    def __init__(self, grid_size=GRID_SIZE, entries_per_grid=ENTRIES_PER_GRID):
        """Initializes the architecture for mini-yolo."""
        super().__init__()
        self.grid_size = grid_size
        self.entries_per_grid = entries_per_grid
        # out: 112x112x3
        self.conv_1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.batch_norm_1 = nn.BatchNorm2d(16)
        self.pool_1 = nn.MaxPool2d(2, 2)
        # out: 56x56x16
        self.conv_2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.batch_norm_2 = nn.BatchNorm2d(32)
        self.pool_2 = nn.MaxPool2d(2, 2)
        # out: 28x28x32
        self.conv_3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.batch_norm_3 = nn.BatchNorm2d(64)
        self.pool_3 = nn.MaxPool2d(2, 2)
        # out: 14x14x64
        self.conv_4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.batch_norm_4 = nn.BatchNorm2d(64)
        self.pool_4 = nn.MaxPool2d(2, 2)
        # out: 7x7x64
        self.conv_5 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.batch_norm_5 = nn.BatchNorm2d(32)
        # out: 7x7x32
        self.flatten = nn.Flatten()
        # out: 1568
        self.fc = nn.Linear(1568, 512)
        self.dropout = nn.Dropout(p=0.5)
        self.out = nn.Linear(512, self.grid_size * self.grid_size * self.entries_per_grid)
        # out: 343

    def forward(self, imgs):
        """Returns raw outputs from a batch of images."""
        out = self.pool_1(tf.relu(self.batch_norm_1(self.conv_1(imgs))))
        out = self.pool_2(tf.relu(self.batch_norm_2(self.conv_2(out))))
        out = self.pool_3(tf.relu(self.batch_norm_3(self.conv_3(out))))
        out = self.pool_4(tf.relu(self.batch_norm_4(self.conv_4(out))))
        out = self.flatten(tf.relu(self.batch_norm_5(self.conv_5(out))))
        out = self.dropout(tf.relu(self.fc(out)))
        out = tf.sigmoid(self.out(out))
        return out

    def decode_predictions(self, outputs, threshold, apply_nms=False):
        """
        Transforms a batch of raw model outputs into a batch (list) of dictionary style predictions
        based on a confidence/objectness threshold. Optionally applies nms.
        """
        outputs = outputs.view(-1, self.grid_size, self.grid_size, self.entries_per_grid)
        batch_preds = []

        for output in outputs:
            boxes = []
            confidences = []
            labels = []

            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    cell = output[i, j]

                    conf = cell[4].item()
                    if conf < threshold:
                        continue

                    x, y, w, h = cell[0:4]
                    class_probs = cell[5:]
                    cls = torch.argmax(class_probs).item()

                    # convert to image-level bounding boxes in [0, 1]
                    x_center = (j + x.item()) / self.grid_size
                    y_center = (i + y.item()) / self.grid_size

                    xmin = max(0, x_center - w.item() / 2)
                    ymin = max(0, y_center - h.item() / 2)
                    xmax = min(1, x_center + w.item() / 2)
                    ymax = min(1, y_center + h.item() / 2)

                    boxes.append([xmin, ymin, xmax, ymax])
                    confidences.append(conf)
                    labels.append(cls)

            if len(confidences) == 0:  # no predictions :(
                pred = {
                    "boxes": torch.zeros((0, 4)),
                    "scores": torch.zeros((0,)),
                    "labels": torch.zeros((0,), dtype=torch.int64),
                }
            else:
                bx = torch.tensor(boxes, dtype=torch.float32)
                sc = torch.tensor(confidences, dtype=torch.float32)
                lb = torch.tensor(labels, dtype=torch.int64)
                if apply_nms:
                    bx, sc, lb = nms_single_image(bx, sc, lb)
                pred = {
                    "boxes": bx,
                    "scores": sc,
                    "labels": lb,
                }
            batch_preds.append(pred)

        return batch_preds

    def predict(self, images, threshold=0.5, apply_nms=False):
        """
        Gets the raw outputs from a batch of images and transforms them into dictionary-style predictions.
        Essentially a wrapper for decode_predictions method. Optionally applies non-maximum suppression.
        """
        self.eval()
        with torch.no_grad():
            outputs = self(images)
            return self.decode_predictions(outputs, threshold, apply_nms=apply_nms)


if __name__ == '__main__':
    """Summarizes model architecture and shows output shape."""
    summary(Model(), input_size=(1, 3, 112, 112), verbose=1)

    test_tensor = torch.randn(1, 3, 112, 112)
    print(Model()(test_tensor).shape)

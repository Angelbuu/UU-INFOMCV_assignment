import kagglehub
import os
import glob
import torch
import xml.etree.ElementTree as ET
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.patches as patches


PATH = kagglehub.dataset_download("andrewmvd/dog-and-cat-detection")
IMG_DIR = os.path.join(PATH, "images")
ANNOTATION_DIR = os.path.join(PATH, 'annotations')
INPUT_IMG_SIZE = 112
GRID_SIZE = 7
ENTRIES_PER_GRID = 7


class CatDogDataset(Dataset):
    def __init__(self, img_dir, ann_dir, input_img_size, transform=None):
        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.transform = transform
        self.img_files = sorted(glob.glob(os.path.join(img_dir, "*.png")))
        self.ann_files = sorted(glob.glob(os.path.join(ann_dir, "*.xml")))
        self.label_map = {"cat": 0, "dog": 1}  # Label mapping
        self.input_img_size = input_img_size

    def parse_annotation(self, ann_path):
        tree = ET.parse(ann_path)
        root = tree.getroot()
        width = int(root.find("size/width").text)
        height = int(root.find("size/height").text)
        objects = []

        for obj in root.findall("object"):
            name = obj.find("name").text
            xmin = int(obj.find("bndbox/xmin").text)
            ymin = int(obj.find("bndbox/ymin").text)
            xmax = int(obj.find("bndbox/xmax").text)
            ymax = int(obj.find("bndbox/ymax").text)

            label = self.label_map.get(name, -1)  # Default to -1 if unknown label
            objects.append({"label": label, "bbox": [xmin, ymin, xmax, ymax]})

        return width, height, objects

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        ann_path = self.ann_files[idx]

        image = Image.open(img_path).convert("RGB")
        width, height, objects = self.parse_annotation(ann_path)

        scaler_x = width / self.input_img_size
        scaler_y = height / self.input_img_size

        target = torch.zeros((GRID_SIZE, GRID_SIZE, ENTRIES_PER_GRID))

        bboxes = []
        for obj in objects:
            xmin = obj['bbox'][0] / scaler_x
            ymin = obj['bbox'][1] / scaler_y
            xmax = obj['bbox'][2] / scaler_x
            ymax = obj['bbox'][3] / scaler_y
            bboxes.append([xmin, ymin, xmax, ymax])

            xmin, ymin, xmax, ymax = obj['bbox']
            label = obj['label']

            x_center = (xmin + xmax) / 2 / width
            y_center = (ymin + ymax) / 2 / height
            w = (xmax - xmin) / width
            h = (ymax - ymin) / height

            grid_x = int(x_center * GRID_SIZE)
            grid_y = int(y_center * GRID_SIZE)

            target[grid_y, grid_x, 0] = x_center * GRID_SIZE - grid_x
            target[grid_y, grid_x, 1] = y_center * GRID_SIZE - grid_y
            target[grid_y, grid_x, 2] = w
            target[grid_y, grid_x, 3] = h
            target[grid_y, grid_x, 4] = 1
            target[grid_y, grid_x, 5 + label] = 1

        bboxes = torch.tensor(bboxes, dtype=torch.float32)
        labels = torch.tensor([obj["label"] for obj in objects], dtype=torch.int64)

        if self.transform:
            image = self.transform(image)

        return image, bboxes, labels, target


def split_data(dataset, test_ratio, val_ratio):
    """Split the dataset into train, validation and test subsets."""
    labels = [dataset.parse_annotation(dataset.ann_files[i])[2][0]['label'] for i in range(len(dataset))]
    indices = list(range(len(dataset)))
    train_val_idx, test_idx = train_test_split(indices, test_size=test_ratio, stratify=labels)

    train_val_labels = [labels[i] for i in train_val_idx]
    train_idx, val_idx = train_test_split(train_val_idx, test_size=val_ratio, stratify=train_val_labels)

    train_data = Subset(dataset, train_idx)
    val_data = Subset(dataset, val_idx)
    test_data = Subset(dataset, test_idx)
    return train_data, val_data, test_data


def visualize_batch(dataloader):
    images, bboxes, labels, targets = next(iter(dataloader))
    fig, axes = plt.subplots(1, len(images), figsize=(15, 5))

    if len(images) == 1:
        axes = [axes]

    for i, (img, bbox, label) in enumerate(zip(images, bboxes, labels)):
        img = img.permute(1, 2, 0).numpy()
        axes[i].imshow(img)

        for box, lbl in zip(bbox, label):
            xmin, ymin, xmax, ymax = box.tolist()
            rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                     linewidth=2, edgecolor='r', facecolor='none')
            axes[i].add_patch(rect)
            axes[i].text(xmin, ymin - 5, f'Label: {lbl.item()}', color='red', fontsize=10,
                         bbox=dict(facecolor='white', alpha=0.5))
        axes[i].axis('off')

    plt.show()


def yolo_collate_fn(batch):
    images = torch.stack([item[0] for item in batch])  # (B, C, H, W)
    bboxes = [item[1] for item in batch]  # keep list for visualization
    labels = [item[2] for item in batch]  # keep list for visualization
    targets = torch.stack([item[3] for item in batch])  # (B, 7, 7, 7)
    return images, bboxes, labels, targets


def prepare_datasets(batch_size=4):
    transform = T.Compose([
        T.Resize((INPUT_IMG_SIZE, INPUT_IMG_SIZE)),
        T.ToTensor()
    ])

    dataset = CatDogDataset(img_dir=IMG_DIR,
                            ann_dir=ANNOTATION_DIR,
                            input_img_size=INPUT_IMG_SIZE,
                            transform=transform)
    train_set, val_set, test_set = split_data(dataset, 0.2, 0.2)

    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=yolo_collate_fn)
    val_dataloader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=yolo_collate_fn)
    test_dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=False, collate_fn=yolo_collate_fn)

    return train_dataloader, val_dataloader, test_dataloader


if __name__ == '__main__':
    train_loader, val_loader, test_loader = prepare_datasets()
    visualize_batch(train_loader)

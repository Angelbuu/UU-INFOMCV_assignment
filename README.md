# UU-INFOMCV_assignment

Four group assignments from Utrecht University's Computer Vision course (INFOMCV), covering camera geometry, video foreground extraction, image classification, and object detection. The datasets in this repository are general-purpose computer vision datasets; these assignments were not medical imaging projects.

## What is in the repository

### [Assignment 1: Camera calibration and 3D overlays](assg_1/)

- Detected and refined chessboard corners with OpenCV to estimate camera parameters.
- Compared calibration runs using reprojection error and projected 3D axes and a cube onto images.
- [Code](assg_1/calibrate.py) · [Report](assg_1/assignment_1_report_group98.pdf)

### [Assignment 2: Foreground masks and 3D reconstruction](assg_2/)

- Built foreground masks from multi-camera video using HSV background subtraction and morphological cleanup.
- Reconstructed visible voxels from camera views and converted a voxel volume into a mesh using marching cubes.
- [Foreground extraction](assg_2/background_subtraction.py) · [Voxel reconstruction](assg_2/voxel_reconstruction.py) · [Report](assg_2/assignment_2_report_group98.pdf)

### [Assignment 3: Image classification and transfer learning](assg_3/)

- Trained and compared LeNet-based PyTorch classifiers on CIFAR-10 using training and validation accuracy, loss curves, and test-set confusion matrices.
- Trained on CIFAR-100 coarse classes, transferred model weights to a CIFAR-10 classifier, and fine-tuned the model for comparison.
- [Models](assg_3/models.py) · [Training and evaluation](assg_3/train.py) · [Cross-validation](assg_3/cross_validation.py)

![LeNet training and validation accuracy](assg_3/LeNet%20Accuracy%20over%20Epochs.png)

### [Assignment 4: Object detection and error analysis](assg_4/)

- Built a small YOLO-style PyTorch detector for cat and dog images, with stratified training, validation, and test splits.
- Implemented optional training augmentation, including colour jitter, Gaussian blur, and autocontrast.
- Evaluated predictions using mAP and F1 across confidence thresholds; compared non-maximum suppression and inspected false positives, false negatives, and class errors.
- [Model](assg_4/model.py) · [Data preparation](assg_4/preprocessing.py) · [Evaluation](assg_4/evaluate.py) · [Report](assg_4/assg4_report_group98.pdf)

![Object detector training and validation loss](assg_4/Training%20vs%20Validation%20Total%20Loss.png)

## Running selected experiments

The code uses Python with PyTorch, torchvision, OpenCV, NumPy, scikit-learn, Matplotlib, Pillow, torchinfo, and KaggleHub. Assignment 2 also uses scikit-image and Open3D. The repository does not currently pin package versions, so install versions compatible with your Python environment.

From the relevant assignment directory:

```bash
cd assg_3
python train.py

cd ../assg_4
python train.py
python evaluate.py
```

Assignment 3 downloads [CIFAR-10 and CIFAR-100](https://cave.cs.toronto.edu/kriz/cifar.html) through torchvision. Assignment 4 downloads the [Dog and Cat Detection dataset](https://www.kaggle.com/datasets/andrewmvd/dog-and-cat-detection) through KaggleHub. Training may take time. Assignments 1 and 2 use their supplied image or video assets and include scripts that write outputs or open display windows; review their code and reports before running them.

## Attribution

This is group coursework. Repository contributors include [@Angelbuu](https://github.com/Angelbuu) and [@MajoBednar](https://github.com/MajoBednar). Some assignment files build on course starter code; the linked reports and source files give more detail. The repository's MIT license is in [LICENSE](LICENSE).

# uc-merced-cnn-comparison
This project compares the performance of advanced Convolutional Neural Networks (CNNs) — AlexNet, VGG19, ResNet50, and LeNet-5 — on the UC Merced Land Use Dataset for image classification. The goal is to evaluate their training behavior, validation accuracy, and generalization ability using standardized metrics and visualizations.
# 🌍 CNN Model Comparison on UC Merced Land Use Dataset

This project implements and compares several Convolutional Neural Network (CNN) architectures for classifying aerial images in the UC Merced Land Use Dataset.

---

## 📁 Dataset

**UC Merced Land Use Dataset**  
- 21 land use categories (e.g., agricultural, forest, harbor, etc.)  
- 100 RGB images per class  
- Image resolution: 256x256 pixels  
- Download link: [https://weegee.vision.ucmerced.edu/datasets/landuse.html](https://weegee.vision.ucmerced.edu/datasets/landuse.html)

---

## 🧠 Models Implemented

The following CNN architectures were trained and evaluated:

- **LeNet-5** – Classic architecture, trained from scratch  
- **AlexNet** – Custom PyTorch implementation with pretrained weights  
- **VGG19** – Deep architecture, pretrained on ImageNet  
- **ResNet50** – Residual network using skip connections, pretrained on ImageNet  

---

## 🏋️‍♂️ Training Details

- **Framework:** PyTorch  
- **Loss Function:** CrossEntropyLoss  
- **Optimizer:** Adam  
- **Batch Size:** 32  
- **Epochs:** 10  
- **Augmentations:** RandomCrop, RandomHorizontalFlip, Normalization  

Training logs and accuracy plots are saved in the `logs/` and `plots/` folders respectively.

---

## 📊 Results

| Model     | Best Validation Accuracy |
|-----------|--------------------------|
| LeNet-5   | 33.02%                   |
| AlexNet   | 94.60%                   |
| VGG19     | 93.65%                   |
| ResNet50  | **98.10%**               |

📈 Training curves and confusion matrices are saved in the `plots/` folder.

---

## 📂 Project Structure

```
uc-merced-cnn-comparison/
├── data/                 # Dataset or link to download
├── models/               # All model definitions (LeNet5, AlexNet, etc.)
├── utils/                # Training, evaluation, Grad-CAM code
├── logs/                 # Training logs
├── plots/                # Graphs and confusion matrices
├── notebooks/            # Jupyter notebooks for exploration
├── main.py               # Entry point for training models
├── requirements.txt      # Required Python packages
└── README.md             # This file
```

---

## 🚀 How to Run

1. Clone the repository:

```bash
git clone https://github.com/yourusername/uc-merced-cnn-comparison.git
cd uc-merced-cnn-comparison
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Train a model:

```bash
python main.py --model resnet50
```

You can change `resnet50` to `alexnet`, `vgg19`, or `lenet5`.

---

## 🔍 Grad-CAM Visualization

Class activation maps (Grad-CAM) are implemented for visualizing what parts of the image the model focuses on during classification. Check `utils/gradcam.py`.

---

## ✨ Acknowledgements

- UC Merced Vision Group for the dataset  
- PyTorch team for the deep learning framework  
- Inspired by classic CNN research papers

---

> Made with ❤️ by Rishi Italiya

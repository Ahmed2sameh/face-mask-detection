# Face Mask Detection with FastAI & Transfer Learning

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Ahmed2sameh/face-mask-detection/blob/main/face_mask_detection.ipynb)

> Binary image classifier that detects whether a person is wearing a face mask — built with FastAI and ResNet18 transfer learning.

## 📌 Overview
This project uses transfer learning with the FastAI library to classify images as **with mask** or **without mask**. The model fine-tunes a pretrained ResNet18 on a face mask dataset using the one-cycle learning rate policy.

## 🧰 Technologies Used
- **Python 3**
- **FastAI v2** — High-level deep learning library
- **ResNet18** — Pretrained ImageNet backbone
- **Google Colab** — GPU training environment

## 🏗️ Approach
1. Load dataset from Google Drive using `ImageDataLoaders`
2. Apply data augmentation (`aug_transforms`) and ImageNet normalization
3. Fine-tune ResNet18 with `cnn_learner` using `fit_one_cycle`
4. Unfreeze model and fine-tune all layers for additional epochs
5. Evaluate with confusion matrix and top losses visualization

## 📊 Results
| Stage | Epochs | LR Range |
|-------|--------|----------|
| Frozen backbone | 4 | `1e-3 → 1e-1` |
| Unfrozen (full fine-tune) | 10 | `1e-4 → 1e-3` |

## 📂 Dataset
- Face mask image dataset with two classes: `with_mask` / `without_mask`
- Loaded from Google Drive (not included in repo)
- Source: [Kaggle Face Mask Dataset](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset)

## 🚀 How to Run
1. Click **Open in Colab** badge above
2. Upload your dataset to Google Drive under `face mask data/train/`
3. Run all cells in order

## 📂 Project Structure
```
├── face_mask_detection.ipynb   # Full pipeline: data → fine-tuning → evaluation
└── README.md
```

## 👤 Author
Ahmed Sameh — [GitHub](https://github.com/Ahmed2sameh)

# RealWaste Classification

## Overview

Built an end-to-end waste classification system using deep learning to classify real-world waste images into 9 categories.
The model achieves a peak **F1-score of 95.57%**, outperforming baseline approaches and improving classification reliability under real-world conditions.

---

## Demo

![Demo](demo/demo.png)

---

## Problem

Improper waste sorting leads to contamination in recycling streams, reducing efficiency and increasing environmental impact.
This project aims to automate waste classification using computer vision to improve sorting accuracy and reduce human error.

---

## Approach

### Models

* ConvNeXtV2 (best performing)
* ResNet50
* DenseNet121
* EfficientNetV2

### Key Techniques

* Transfer Learning (ImageNet pre-trained models)
* Partial Fine-Tuning (last convolution block + classifier)
* Data Augmentation (flip, rotation, cropping)
* Class Imbalance Handling (weighted CrossEntropyLoss)

---

## Dataset

* 4,752 real-world waste images
* 9 classes:

  * Cardboard, Food Organics, Glass, Metal, Trash, Paper, Plastic, Textile, Vegetation
* Imbalanced distribution handled via weighted loss and stratified sampling

---

## Results

| Model          | Accuracy | F1 Score   |
| -------------- | -------- | ---------- |
| ConvNeXtV2     | 90.95%   | **90.98%** |
| DenseNet121    | 90.11%   | 90.09%     |
| ResNet50       | 89.68%   | 89.69%     |
| EfficientNetV2 | 73.68%   | 73.15%     |

* Achieved **95.57% peak F1-score during training**
* Outperformed baseline InceptionV3 model (F1: 90.25%)

---

## Key Contributions

* Improved minority class performance using weighted loss and stratified sampling
* Reduced overfitting via augmentation and learning rate scheduling
* Designed a modular training pipeline for evaluating multiple architectures

---

## Project Structure

```
realwaste-classification/
├── notebooks/        # training experiments
├── src/              # training & inference scripts
├── demo/             # demo images
├── docs/             # report (optional)
└── requirements.txt
```

---

## Tech Stack

* Python
* PyTorch
* NumPy, pandas
* Matplotlib

---

## How to Run

```bash
pip install -r requirements.txt
python src/train.py
```

---

## Future Work

* Deploy as a web application for real-time classification
* Improve generalization with larger datasets
* Explore transformer-based models (e.g., ViT)

---

# Effective Brain Tumor Detection using Vision Transformer

This project focuses on detecting and classifying brain tumours from MRI scans using deep learning techniques such as Convolutional Neural Networks (CNNs), ResNet-based models, and Vision Transformers (ViT). The system classifies MRI images into different tumour categories or normal (healthy).

---

## Project Overview

- Detects brain tumours from MRI images.
- Classifies images into four classes:
  - Glioma
  - Meningioma
  - Pituitary
  - Normal (No Tumour)
- Implements CNN, ResNet, and Vision Transformer–based models.
- Includes training, evaluation, prediction, and visualization scripts.

---

## Dataset

- **Dataset Name:** Brain Tumor MRI Dataset  
- **Classes:** 4 (glioma, meningioma, pituitary, normal)  
- **Training Images:** 5712  
- **Testing Images:** 1311  
- **Source:** Kaggle  
  https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri  

> ⚠️ Dataset is not included in this repository due to size constraints.  
> Please download it manually from Kaggle.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Priyanka1846/Brain-Tumour.git
   cd Brain-Tumour

2. Install required dependencies:

   ```pip install -r requirements.txt```


3. Download the dataset and update the dataset path inside the training and prediction scripts.

---

## Usage

1. Train the model
   ```bash
   python train.py

2. Predict tumour from an MRI image
   ```bash
   python predict.py
   
   OR (CNN-specific prediction)
   
   python predict_cnn.py

3. Evaluate the model
   ```bash
   python evaluate.py
---
## Model Architecture

- CNN and ResNet-based architectures

- Vision Transformer (ViT) implementation

- Data augmentation for better generalization

## Evaluation metrics:

1. Accuracy

2. Precision

3. Recall

4. F1-score

5. Confusion Matrix

---

## Results

1. Achieves high accuracy in classifying brain tumour types.

2. Generates confusion matrices, loss curves, and performance comparison graphs.

3. Grad-CAM visualizations are used for model interpretability.

---

## Repository Structure

- train.py – Model training

- predict.py / predict_cnn.py – Tumour prediction

- evaluate.py – Model evaluation

- gradcam.py – Visualization using Grad-CAM

- requirements.txt – Required Python libraries

---

## Acknowledgments

- Dataset provided by Kaggle

- Deep learning framework: PyTorch

- Inspiration: [GitHub Brain Tumor Projects](https://github.com/topics/brain-tumor)

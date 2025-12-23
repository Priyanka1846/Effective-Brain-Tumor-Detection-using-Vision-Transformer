# Brain Tumor Detection Using Deep Learning

This project aims to detect and classify brain tumors in MRI scans using deep learning models such as CNNs or Vision Transformers. The system is trained on a dataset of brain MRI images and can classify tumors into different types (e.g., glioma, meningioma, pituitary, and healthy).

***

## Project Overview

- Detects brain tumors in MRI images.
- Classifies tumors into four categories: glioma, meningioma, pituitary, and normal (no tumor).
- Uses a pretrained model (e.g., ResNet-50, CNN, or Vision Transformer) for high accuracy.

***

## Dataset

- Dataset: Brain Tumor MRI Dataset
- Classes: 4 (glioma, meningioma, pituitary, normal)
- Training images: 5712
- Testing images: 1311
- Source: [Kaggle Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri)

***

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/brain-tumor-detection.git
   cd brain-tumor-detection
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Download the dataset and place it in the `data/` folder.

***

## Usage

- Run the training script:
  ```
  python train.py
  ```

- Run the prediction script:
  ```
  python predict.py --image_path path/to/image.jpg
  ```

***

## Model Architecture

- Pretrained ResNet-50 or custom CNN/Vision Transformer.
- Data augmentation for improved generalization.
- Metrics: Accuracy, precision, recall, F1-score.

***

## Results

- Achieves high accuracy in classifying brain tumor types.
- Confusion matrix and classification report are generated after training.

***

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

---

## Acknowledgments

- Dataset source: Kaggle
- Inspiration: [GitHub Brain Tumor Projects](https://github.com/topics/brain-tumor)

***

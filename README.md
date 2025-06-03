# 🧠 Brain Tumor Segmentation Using U-Net in TensorFlow

This project uses a U-Net convolutional neural network to perform semantic segmentation on MRI brain scans to detect and localize brain tumors. The model is built using TensorFlow and Keras, with comprehensive preprocessing, training, and evaluation workflows.

---

## 📌 Key Features

- ✅ U-Net implementation from scratch in TensorFlow/Keras
- ✅ Binary segmentation of brain tumor regions from MRI images
- ✅ Uses Dice coefficient and binary crossentropy loss
- ✅ Dataset loading, preprocessing, and visualization
- ✅ Model training with callbacks and evaluation metrics

---

## 🧪 Architecture: U-Net

U-Net is a CNN architecture used for biomedical image segmentation. It has a **contracting path** to capture context and a **symmetric expanding path** to enable precise localization, with skip connections between layers.

---

## 🗂️ Dataset Structure

You should organize your dataset as follows:

- /data
- ├── images
- │ ├── image1.png
- │ └── ...
- └── masks/
- ├── mask1.png
- └── ...

Each `image` should have a corresponding `mask`.

> 💡 Dataset Source: *https://www.kaggle.com/datasets/atikaakter11/brain-tumor-segmentation-dataset/data*

---

## 📊 Evaluation Metrics

- Dice Coefficient

- Binary Crossentropy

- Pixel Accuracy

---

## 📚 Citations & References

- U-Net: Convolutional Networks for Biomedical Image Segmentation

- TensorFlow and Keras documentation

---


## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/RishitSaxena55/brain-tumor-segmentation-using-unet.git
cd brain-tumor-segmentation-using-unet

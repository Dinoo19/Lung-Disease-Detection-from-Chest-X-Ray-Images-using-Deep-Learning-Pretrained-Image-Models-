# Lung Disease Detection from Chest X-Ray Images using Deep-Learning Pretrained Image Models
This project uses deep learning models (ResNet-50, VGG-16) to detect lung diseases such as COVID-19, viral pneumonia, and normal conditions from chest X-ray images. It applies image preprocessing with CLAHE to enhance contrast and achieve accurate, automated diagnosis.

 🩺 Lung Disease Detection from Chest X-Ray Images

## 📖 Overview
Lung diseases are among the most widespread health issues worldwide. Early detection plays a crucial role in effective treatment. This project uses **deep learning** and **image processing** to automatically identify and classify chest X-rays into:
- **COVID-19**
- **Non-COVID (Flu & RSV)**
- **Normal**

By leveraging **ResNet-50** and **VGG-16** pretrained models, this system can distinguish between healthy and diseased lungs from X-ray images.

---

## 🎯 Objectives
- Build an automated lung disease classification system using X-ray images.
- Compare performance between **ResNet-50** and **VGG-16** models.
- Improve diagnostic efficiency in medical imaging.

---

## 🧩 Dataset
The dataset contains labeled chest X-ray images divided into:
- **COVID-19**: 3,616 images  
- **Normal**: 10,192 images  
- **Lung Opacity (Non-COVID)**: 6,012 images  
- **Viral Pneumonia**: 1,345 images  

**Dataset Link:** [Google Drive Dataset](https://drive.google.com/file/d/1bum9Sehb3AzUMHLhBMuowPKyr_PCrB3a/view)

For training and testing, a subset of **21165 images** was used and later reduced for model efficiency:
- **Train Set:** 3728  
- **Test Set:** 1116  
- **Validation Set:** 730  

---

## ⚙️ Preprocessing
The preprocessing pipeline includes:
1. **Renaming images** for consistency (COVID, NonCOVID, Normal).
2. **Contrast Limited Adaptive Histogram Equalization (CLAHE)** to enhance image contrast.
3. **Resizing** images to `100x100` pixels.
4. **Color conversion** (BGR → LAB → BGR).
5. **Label encoding** using one-hot encoding.

---

## 🧠 Models Used
### 1. **ResNet-50**
- 50-layer residual neural network.
- Utilizes skip connections for deep feature learning.
- Pretrained on ImageNet.
- Achieved **82.5% accuracy** on test data.

### 2. **VGG-16**
- Deep CNN with 16 layers.
- Captures fine-grained image details.
- Achieved **76.4% accuracy** on test data.

---

## 📈 Evaluation Metrics
Both models were evaluated using:
- **Confusion Matrix**
- **Classification Report** (Precision, Recall, F1-score)
- **Accuracy**

| Model | Test Accuracy |
|--------|----------------|
| ResNet-50 | 82.5% |
| VGG-16 | 76.4% |

---

## 🧰 Tech Stack
- Python  
- TensorFlow / Keras  
- NumPy, Pandas  
- OpenCV  
- Matplotlib / Seaborn  
- Scikit-learn  

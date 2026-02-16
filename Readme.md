# 😷 Face Mask Detection using CNN and TensorFlow

A real-time face mask detection system built using Convolutional Neural Networks (CNN) with TensorFlow/Keras and OpenCV. The model detects whether a person is wearing a face mask or not using webcam input.

This project demonstrates deep learning, computer vision, model optimization, and real-time inference.

---

## 🚀 Features

- Real-time face mask detection using webcam
- CNN model built using TensorFlow/Keras
- Data augmentation for improved generalization
- Binary classification: Mask / No Mask
- Batch Normalization and Dropout for improved training stability
- Clean and modular implementation
- Easy to train and deploy

---

## 🧠 Model Architecture

The CNN model consists of:

- Convolutional Layers for feature extraction
- MaxPooling Layers for dimensionality reduction
- Batch Normalization for stable and faster training
- Dropout Layers to reduce overfitting
- Fully Connected Dense Layers for classification
- Sigmoid output activation for binary classification

**Input shape:** `(224, 224, 3)`  
**Output:** Mask / No Mask  

---

## 📊 Model Development and Improvements

### Baseline Model

A baseline CNN model was developed with:

- 2 Convolutional layers  
- 1 Dense layer  

Performance:

- Training Accuracy: **93%**
- Validation Accuracy: **91%**

Increasing model complexity without improving data diversity resulted in overfitting. Training accuracy increased, but validation accuracy did not improve significantly, indicating poor generalization.

---

### Improved Model with Data Augmentation and Increased Capacity

To improve performance, both model capacity and training data diversity were enhanced.

**Model improvements:**

- Increased number of convolutional layers  
- Added Batch Normalization layers  
- Added Dropout layers  

**Data augmentation techniques used:**

- Random horizontal flip
- Random rotation
- Random zoom

These techniques increased training data diversity and helped prevent overfitting.

Performance after improvements:

- Training Accuracy: **95%**
- Validation Accuracy: **94.7%**

This demonstrates that combining increased model capacity with data augmentation significantly improves generalization and model performance.

---

## 📁 Project Structure

```
face-mask-detection/
│
├── model/
│   └── face_mask_model.keras      # Trained CNN model
│
├── webcam.py                      # Real-time detection script
├── model.ipynb                    # Training notebook
├── requirements.txt               # Python dependencies
├── README.md                      # Project documentation
└── .gitignore                     # Files ignored by Git
```


## 🧪 Training Pipeline

Training includes:

- Data preprocessing
- Data augmentation
- CNN model training
- Model evaluation
- Model saving

---

## ▶️ Run Real-Time Detection

Run the webcam detection script:

```python
python3.10 webcam.py
```

press q to exit

## 🛠 Technologies Used

- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Matplotlib

---

## 🎯 Key Concepts Demonstrated

- Convolutional Neural Networks (CNN)
- Image classification
- Data augmentation
- Overfitting reduction
- Model optimization
- Real-time inference
- Computer vision deployment

---
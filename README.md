# Breast Cancer Decision-Support System using PyTorch

## Overview
This project implements an end-to-end data science workflow to develop a neural network–based decision-support system for classifying breast tumors as malignant or benign. Using the Breast Cancer Wisconsin dataset, the system demonstrates data exploration, preprocessing, model training, evaluation, visualization, and interactive inference using PyTorch.

## ⚠️ Disclaimer
This project is intended **strictly for academic and research demonstration purposes**.  
It is **not a medical diagnostic tool**.

---

## Dataset
- **Name:** Breast Cancer Wisconsin Dataset  
- **Source:** UCI Machine Learning Repository (via scikit-learn)  
- **Samples:** 569  
- **Features:** 30 numerical features describing tumor characteristics  
- **Target Classes:**  
  - 0 – Malignant  
  - 1 – Benign  

---

## Methodology
The project follows a structured data science pipeline:

- Exploratory Data Analysis (class distribution, feature distributions)
- Data preprocessing (train/validation/test split, feature standardization)
- Model development using a Multi-Layer Perceptron (PyTorch)
- Regularization with dropout and early stopping
- Performance evaluation using accuracy, precision, recall, F1-score, confusion matrix, and ROC-AUC
- Visualization of training dynamics and evaluation results

---

## Model Architecture
- Fully connected neural network (MLP)
- ReLU activation functions
- Dropout for overfitting prevention
- Cross-Entropy Loss
- Adam optimizer

---

## Results
The trained model demonstrates strong performance on unseen test data:
- Accuracy: ~95%
- ROC-AUC: ~0.99
- Low false-negative rate
- Stable convergence with early stopping

These results indicate effective generalization and reliable class discrimination.

---

## Decision-Support Demo
An interactive inference module is included to simulate real-world usage. Users can input realistic tumor feature values within dataset ranges, and the system outputs:
- Predicted class (malignant or benign)
- Class probabilities
- Interpretable decision-support message

This demonstrates how trained machine learning models can be integrated into practical analytical systems.

---

## Technologies Used
- Python
- PyTorch
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn

---

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/breast-cancer-decision-support-pytorch.git

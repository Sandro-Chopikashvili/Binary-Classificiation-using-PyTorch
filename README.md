# Binary Classification Pipeline with PyTorch and Scikit-Learn

This repository contains a streamlined and modular pipeline for a binary classification task using PyTorch for modeling and scikit-learn for data preprocessing. The project demonstrates end-to-end handling of data cleaning, feature engineering, model training, and evaluation — ideal for practical machine learning workflows with structured tabular data.

---

## Table of Contents

- [Project Overview](#project-overview)  
- [Dataset](#dataset)  
- [Preprocessing](#preprocessing)  
- [Model Architecture](#model-architecture)  
- [Training Procedure](#training-procedure)  
- [Evaluation Metrics](#evaluation-metrics)  
- [Usage](#usage)  
- [Requirements](#requirements)  
- [Future Work](#future-work)  
- [Contact](#contact)  

---

## Project Overview

This project implements a simple yet effective binary classification pipeline leveraging:

- **Pandas & NumPy** for data manipulation  
- **Scikit-learn** for handling missing values, encoding categorical variables, and splitting data  
- **PyTorch** for defining and training a neural network classifier  
- Emphasis on reproducibility with fixed random seeds and train/test splits  

The goal is to classify samples into two classes accurately while showcasing best practices in data preprocessing and model evaluation.

---

## Dataset

- The data is loaded from `classification_dataset_clean.csv`.  
- Features consist of both numerical and categorical variables.  
- The target variable is a binary label named `Class`.  
- The dataset may contain missing values that are handled during preprocessing.

---

## Preprocessing

- **Categorical Features:** Missing values are imputed using the most frequent value and then one-hot encoded.  
- **Numerical Features:** Missing values are imputed with the median value.  
- **Pipeline:** Uses `ColumnTransformer` and `Pipeline` from scikit-learn to apply transformations in a clean and modular way.  
- The preprocessor is fit on the entire dataset before splitting for training/testing.

---

## Model Architecture

- A simple feedforward neural network with two fully connected layers:  
  - Input layer with neurons equal to number of features after preprocessing.  
  - Hidden layer with 5 neurons and ReLU activation.  
  - Output layer with 1 neuron and Sigmoid activation for binary classification probability output.

---

## Training Procedure

- Uses Binary Cross-Entropy Loss (`BCELoss`).  
- Optimizer: Adam with learning rate 0.01.  
- Trained for 100 epochs on the training split.  
- Training loop includes forward pass, loss computation, backward pass, and optimizer step.

---

## Evaluation Metrics

- **Accuracy:** Percentage of correctly predicted labels on the test set.  
- **Confusion Matrix:** Breakdown of True Positives, True Negatives, False Positives, and False Negatives.  
- **Precision, Recall, F1-Score:** To assess classification quality beyond accuracy, especially for imbalanced data.

---

## Usage

1. Clone this repo:
   ```bash
   git clone https://github.com/Sandro-Chopikashvili/binary-classification-pytorch.git
   cd binary-classification-pytorch

# Intel Image Classification

## Overview
This project focuses on image classification using machine learning and deep learning techniques.

The goal is to classify natural scene images into one of six categories:

- Buildings
- Forest
- Glacier
- Mountain
- Sea
- Street

---

## Dataset

Dataset used:
Intel Image Classification Dataset

The data was divided into:

- Training set
- Validation set
- Test set

Images were resized to:

150 × 150 pixels

---

## Models Implemented

### Linear Model (Baseline)
A simple baseline model that flattens image pixels and applies a dense output layer.

### Baseline CNN
A convolutional neural network designed to capture spatial image features.

### Improved CNN
An enhanced CNN architecture with additional layers and regularization techniques to improve generalization.

---

## Evaluation Metrics

The models were evaluated using:

- Training Accuracy
- Validation Accuracy
- Test Accuracy
- Loss curves

---

## Results Summary

- Linear model showed limited performance
- Baseline CNN improved feature extraction but suffered from overfitting
- Improved CNN achieved the best balance between learning and generalization

---

## Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- Jupyter Notebook

---

## Project Structure

```text
Intel-Image-Classification/
│
├── Intel_Image_Classification.ipynb
├── train_linear_model.py
├── train_baseline_cnn.py
├── train_improved_cnn.py
├── predict_unlabeled.py
├── saved_models/
```

---

## Author

Ido Cohen
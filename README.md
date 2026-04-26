# Ethiopian Genuine and Counterfeit Banknote Classification

A master's thesis research project for detecting counterfeit Ethiopian banknotes (100 and 200 ETB) using deep learning and TensorFlow Lite models.

## Overview

This project implements a machine learning-based classification system to identify counterfeit Ethiopian currency, specifically focusing on the 100 and 200 ETB banknotes. The research aims to contribute to financial security by providing an automated detection mechanism.

## Models

The project includes multiple pre-trained TensorFlow Lite models:

- `Dense121_best_weights_mixed-III.tflite` - Best performing Dense121 model
- `dense121.tflite` - DenseNet121 architecture
- `Vgg16.tflite` - VGG16 architecture
- `Vgg19.tflite` - VGG19 architecture
- `Vgg19_mixed.tflite` - Mixed VGG19 variant

## Requirements

See `requirements.txt` for Python dependencies.

## Usage

```python
python counterfeit_etb_classification.py
```

## Research Context

This project was developed as part of a master's thesis research focusing on computer vision techniques for currency authentication in the Ethiopian financial system.

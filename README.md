# Baldness Classification Project

This project aims to classify different stages of baldness using deep learning techniques implemented with PyTorch, OpenCV, and Scikit-learn. The dataset used for this project consists of approximately 2100 images obtained from Roboflow Universe. These images are annotated with five possible classes of baldness: bald, normal, stage1, stage2, and stage3.

## Overview

Baldness classification is a significant task in the field of computer vision and healthcare. By accurately classifying the stages of baldness, it can assist in early detection and intervention strategies for hair loss problems.

## Technologies Used

- PyTorch: For implementing deep learning models
- OpenCV: For image processing tasks such as resizing and grayscale conversion
- Scikit-learn: For evaluation metrics and model evaluation
- Roboflow: For obtaining and preprocessing the dataset

## Dataset Details
- **Source**: [Roboflow Universe](https://universe.roboflow.com/search?q=class%3Abald)
- **Number of Images**: Approximately 2100
- **Classes**:
    - bald
    - normal
    - stage1
    - stage2
    - stage3
- **Image Preprocessing**: Images were resized to various dimensions (28x28, 32x32, 50x50, 100x100) using OpenCV. Grayscale conversion was applied when necessary.

The dataset was split into:
- 1500 images for training
- 300 images for validation
- 300 images for testing

## Process

1. **Data Collection and Preprocessing**: The dataset was obtained from Roboflow Universe and preprocessed by resizing images and converting them to grayscale.
2. **Model Development**: Various convolutional neural network (CNN) architectures were developed using PyTorch.<br>
   Different configurations were explored, ranging from simple architectures with two convolutional layers to more complex architectures with multiple convolutional and linear layers with dropout.<br>
   Pretrained models such as ResNet50 and VGG16 were also utilized.
4. **Model Training**: Models were trained using a batch size of 32.
5. **Evaluation**: The performance of models was evaluated using metrics including accuracy, precision, recall, F1 score, area under the ROC curve (AUC) and confusion matrix.

## Modelling

Several CNN architectures were experimented with, ranging from simple to complex. The performance of each model varied based on the complexity of the architecture and the use of pretrained models. For example:
- The simplest CNN architecture achieved 0.9 in all metrics on the test set, while obtaining 0.4 on the training set.
- The VGG16 model achieved a precision of 0.84 on the training, validation, and test sets. However, other metrics were lower, around 0.2.

## Evaluation

Due to the limited number of images and slight class imbalance, achieving optimal performance on all metrics was challenging. <br> Certain metrics were prioritized over others based on the nature of the problem and the goals of the classification task.

For further details and implementation code, please refer to the project files.

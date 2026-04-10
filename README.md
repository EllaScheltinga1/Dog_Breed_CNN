# Dog Breed Identification (Udacity Project)

## Project Description
This project builds an end‑to‑end dog breed identification algorithm using Convolutional Neural Networks (CNNs). Given an image, the system detects whether it contains a human, a dog, or neither, and predicts the most likely dog breed using deep learning.

---
## Project Motivation
The goal of this project is to explore practical applications of convolutional neural networks by building an end‑to‑end image classification pipeline. By combining human detection, dog detection, and dog breed classification, this project demonstrates how multiple deep learning models can work together to solve a real‑world computer vision problem.

---

## Project Overview

The notebook is structured in steps:

1. **Import datasets**
   - Dog images (train/valid/test) + labels for 133 breeds
   - Human images (LFW subset)

2. **Human detection**
   - Uses OpenCV Haar Cascades face detector to check for a visible face

3. **Dog detection**
   - Uses a pre-trained **ResNet-50** model trained on ImageNet  
   - Dog classes correspond to ImageNet indices **151–268**

4. **Breed classification (CNN from scratch)**
   - Custom CNN with multiple Conv + MaxPool blocks, Global Average Pooling, Dense + Dropout
   - Trained with checkpointing and learning-rate reduction

5. **Breed classification (Transfer Learning)**
   - Uses precomputed bottleneck features from a pre-trained CNN
   - Includes baseline VGG16 bottleneck model and a custom transfer learning head

6. **Final algorithm**
   - Combines human detector + dog detector + transfer learning breed classifier into one pipeline

---

## Libraries Used
- Python
- TensorFlow / Keras
- OpenCV
- NumPy, scikit‑learn
- Matplotlib

---
## Files in Repository

- dog_app.ipynb (main file, containing all code and results)
- 6 files used for testing algorithm in step 7 in notebook
     - Albert_Einstein.webp
     - Beagle_600.jpg
     - Bulldog.jpg
     - German_Shepherd.jpg
     - Rottweiler.jpg
     - Taylor_Swift.jpg

---

## Summary of Results
The final algorithm reliably distinguishes between human and dog images using face detection and a pre‑trained ResNet‑50 dog detector. Dog breed classification performs well when using transfer learning, achieving over **80% test accuracy**. Most misclassifications occur between visually similar breeds. Overall, the system demonstrates a robust end‑to‑end CNN pipeline to detect dog breeds from dog images.



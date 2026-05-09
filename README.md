# Coconut Leaf Disease Classification Using Deep Learning

A Deep Learning based web application for detecting and classifying coconut leaf diseases using PyTorch and Flask.

This project uses a pretrained CNN model (ResNet18) with Transfer Learning to classify coconut leaf images into multiple disease categories with confidence scores.

---

# Project Overview

Coconut trees are affected by several diseases that reduce crop quality and productivity. Early disease detection can help farmers take preventive measures quickly.

This project aims to automate coconut leaf disease detection using Deep Learning and Computer Vision techniques.

Users can upload an image of a coconut leaf through the web application, and the model predicts the disease class along with confidence score.

---

# Results

The model achieved:

- Training Accuracy: 96.26%
- Validation Accuracy: 97.71%
- Validation Loss: 0.0832

# Home Page
<img width="1918" height="1015" alt="Screenshot 2026-05-09 144118" src="https://github.com/user-attachments/assets/88775908-de08-4fab-87d6-bdc70365edc1" />

# Prediction
<img width="1918" height="1015" alt="Screenshot 2026-05-09 144206" src="https://github.com/user-attachments/assets/dfeaec6d-2bdb-4ae9-b3ed-982741d3ece2" />
<img width="1918" height="1008" alt="Screenshot 2026-05-09 144336" src="https://github.com/user-attachments/assets/eb83fe22-2a5f-49a1-9dac-cfb9e6db0398" />


# Features

- Deep Learning based disease classification
- Image upload through web interface
- PyTorch model integration
- Modern responsive UI
- Confidence score display
- Transfer Learning using ResNet18
- Flask web application
- Multiple disease classification

---

# Disease Classes

The model is trained to classify the following classes:

- Caterpillar Damage
- CCI Leaflet Disease
- Drying of Leaflets
- Flaccidity Disease
- Healthy Leaves
- Yellowing Disease

---

# Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming Language |
| PyTorch | Deep Learning Framework |
| Torchvision | Pretrained Models & Image Processing |
| Flask | Web Framework |
| PIL | Image Handling |
| HTML/CSS | Frontend UI |
| Gunicorn | Deployment Server |

---

# Model Architecture

The project uses:

## ResNet18 Transfer Learning

Pretrained ResNet18 model is used and fine-tuned for coconut leaf disease classification.

### Workflow

Input Image  
↓  
Image Preprocessing  
↓  
ResNet18 Model  
↓  
Disease Prediction  
↓  
Confidence Score

---

# Dataset

Dataset used:

https://www.kaggle.com/datasets/samitha96/coconutdiseases/data

---

# Data Preprocessing

The following preprocessing techniques were used:

- Image resizing
- Data augmentation
- Horizontal flipping
- Rotation
- Tensor conversion
- Dataset balancing using augmentation

---

# Project Structure

```text
CoconutDiseaseProject/
│
├── dataset/
├── models/
│   ├── best_model.pth
│   ├── final_model.pth
│   ├── class_names.json
│   └── history.json
│
├── static/
│   └── uploads/
│
├── templates/
│   └── index.html
│
├── app.py
├── train.py
├── requirements.txt
├── Procfile
└── README.md

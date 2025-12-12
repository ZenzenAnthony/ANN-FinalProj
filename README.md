# ANN Final Project – Public Unrest Classification from Text

This repository contains the final project for **CCS 248 – Artificial Neural Networks**.

## Project Title
**Classifying Levels of Public Unrest from Social Media Text Using a Neural Network**

---

## 🧠 Project Overview

The goal of this project is to train a **deep neural network from scratch** to classify
levels of public unrest expressed in short online text (e.g., Reddit-style comments).

The task is formulated as a **multi-class classification problem**, where each text
sample is assigned to one of three unrest categories derived from emotion labels.

The project complies with CCS-248 requirements:
- No pretrained models
- No transfer learning
- Neural network trained from random initialization

---

## 📊 Dataset

The project uses the **GoEmotions dataset**, a publicly available dataset released by
Google Research containing approximately 58,000 English Reddit comments annotated
with emotion labels.

### Preprocessing Summary
- Emotion labels are mapped into **three unrest classes**
- Text is cleaned and normalized
- TF-IDF is used to convert text into numeric feature vectors
- No external embeddings are used

### Bias & Privacy
- The dataset contains publicly available text
- No personally identifiable information is included
- Class imbalance is handled using class weighting during training

---

## 🧩 Neural Network Architecture

The final model is a **feedforward neural network (MLP)** trained from scratch.

**Architecture:**
- Input Layer: TF-IDF features (5000 dimensions)
- Hidden Layer: Dense (128 units, ReLU)
- Dropout: 0.3
- Output Layer: Dense (3 units, Softmax)

**Training Details:**
- Optimizer: Adam
- Loss Function: Sparse Categorical Cross-Entropy
- Early Stopping enabled

---

## 🔧 Hyperparameter Tuning

Multiple hyperparameter configurations were tested by varying:
- Learning rate
- Hidden layer size
- Dropout rate

The best-performing configuration achieved a **validation accuracy of 65.39%**, which
exceeds the minimum requirement of 50–60%.

---

## 📈 Evaluation

Model performance was evaluated using **classification accuracy** on a held-out
validation set.

**Best Validation Accuracy:** **65.39%**

---

## 📁 Repository Structure

```text
ANN-FinalProj/
│
├── Goemotion/          # Dataset reference and exploratory scripts
│
├── PublicUnrest/
│   ├── data/           # Raw and processed datasets
│   ├── src/            # Preprocessing, training, and evaluation scripts
│   ├── plots/          # Evaluation visualizations
│   └── documentation/  # Final project report and diagrams
│
└── README.md

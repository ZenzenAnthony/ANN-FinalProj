# ANN Final Project – Public Unrest Classification from Text

This repository contains the final project for **CCS 248 – Artificial Neural Networks**.

## Project Title
**Classifying Levels of Public Unrest from Social Media Text Using a Neural Network**

---

## 🧠 Project Overview

The objective of this project is to design and train an **artificial neural network from scratch**
to classify levels of public unrest expressed in short social-media-style text
(e.g., Reddit comments).

The task is formulated as a **multi-class classification problem**, where each text sample
is assigned to one of **three unrest levels**: **Low**, **Medium**, or **High**.
These classes are derived from emotion annotations in the GoEmotions dataset.

This project strictly complies with **CCS 248 requirements**:
- No pretrained models
- No transfer learning
- Neural network initialized from random weights
- Fully student-implemented pipeline

---

## 📊 Dataset

The project uses the **GoEmotions dataset**, a publicly available dataset released by
Google Research containing approximately **58,000 English Reddit comments** annotated
with fine-grained emotion labels.

### Preprocessing Summary
- Emotion labels are mapped into **three public unrest classes**
- Text is cleaned and normalized (lowercasing, punctuation removal, URL removal)
- TF-IDF is used to convert text into numeric feature vectors
- No pretrained word embeddings are used

### Bias & Privacy
- The dataset contains publicly available text
- No personally identifiable information is included
- Potential class imbalance is addressed during model training

---

## 🧩 Neural Network Architecture

The final model is a **feedforward neural network (MLP)** trained from scratch.

**Architecture:**
- Input Layer: TF-IDF features (up to 5000 dimensions)
- Hidden Layer: Dense (128 units, ReLU)
- Dropout: 0.3
- Output Layer: Dense (3 units, Softmax)

**Training Details:**
- Optimizer: Adam
- Loss Function: Sparse Categorical Cross-Entropy
- Early stopping based on validation loss

---

## 📈 Evaluation

Model performance is evaluated using **classification accuracy** on a held-out validation
and test set.

**Best Validation Accuracy:** **~65%**, exceeding the minimum requirement of 50–60%.

Additional evaluation artifacts such as confusion matrices and accuracy curves
are generated and saved during experimentation.

---

## 📓 Notebooks (Primary Submission Artifacts)

The main implementation of the project is provided through Jupyter notebooks:

- `01_data_preprocessing.ipynb` – dataset preparation and label mapping
- `02_model_training.ipynb` – TF-IDF vectorization and ANN training
- `03_evaluation.ipynb` – model evaluation and analysis

These notebooks represent the **official implementation** used for grading.

---

## 📁 Repository Structure

```text
ANN-FinalProj/
│
├── Goemotion/           # Dataset references and exploratory scripts
│
├── Notebooks/           # Primary implementation (Jupyter notebooks)
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_evaluation.ipynb
│
├── PublicUnrest/
│   ├── data/            # Raw and processed datasets
│   ├── models/          # Saved trained models
│   ├── plots/           # Evaluation visualizations
│   ├── src/             # Reference Python scripts (non-primary)
│   └── documentation/   # Final project report and diagrams
│
└── README.md
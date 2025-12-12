```markdown
# PublicUnrest – ANN Project Implementation

This directory contains the implementation artifacts for the **CCS 248 – Artificial Neural Networks**
final project on **public unrest classification from text**.

All neural networks in this project are **trained from scratch**, without pretrained models
or external embeddings.

---

## 📓 Primary Implementation

The main project workflow is implemented in **Jupyter notebooks**, located in the root
`Notebooks/` directory:

1. **Data preprocessing and label mapping**
2. **Model training using TF-IDF + ANN**
3. **Model evaluation and analysis**

These notebooks constitute the **official submission code** for the project.

---

## 📁 Folder Structure

```text
PublicUnrest/
│
├── data/
│   ├── raw/            # Reference copies of GoEmotions data
│   └── processed/      # Preprocessed datasets for training and evaluation
│
├── models/             # Saved trained ANN models
│
├── plots/              # Evaluation plots (accuracy curves, confusion matrices, etc.)
│
├── src/                # Reference Python scripts (legacy / non-primary)
│
└── documentation/      # Final report, diagrams, and supporting documents
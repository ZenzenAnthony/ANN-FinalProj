
---

## 📝 2) PublicUnrest README – `PublicUnrest/README.md` (Temporary)

```markdown
# PublicUnrest – ANN Project Implementation (Temporary README)

This folder contains the WORKING implementation files for the project:

**Predicting Levels of Public Unrest from Social Media Text:  
A Neural Network-Based Solution**

⚠️ *This is a temporary README for setup and reference.  
It will be updated after preprocessing and model training are completed.*

---

## 📁 Folder Structure (Temporary)

```text
PublicUnrest/
│
├── data/
│   ├── raw/          # Raw GoEmotions dataset (tsv/csv referenced or copied here)
│   └── processed/    # Processed CSV files created after preprocessing
│
├── src/              # Python scripts (preprocessing, training, evaluation)
│
├── models/           # Trained models (.h5), vectorizers (.pkl)
│
└── documentation/    # Writeups, methodology notes, diagrams, final report

🧠 Project Description

This project aims to estimate the intensity of public unrest expressed in social-media-style text using an artificial neural network trained on the GoEmotions dataset.

Instead of predicting only 3 discrete classes (calm / mild / high), we:

Assign each emotion a weight between 0.0 and 1.0 that reflects how unrest-related it is.

Combine active emotions in a comment into a single unrest_percent value (0%–100%).

The neural network is then trained to predict this percentage from text.


🔢 Label Mapping to Unrest Percentage (Temporary Summary)

Each GoEmotions emotion is mapped to an unrest weight in 
0
,
1
0,1.
Examples (subject to refinement):

Calm / positive / neutral emotions → weight ≈ 0.0

admiration, approval, amusement, curiosity, joy, gratitude, optimism, neutral, etc.

Mild negative / concern emotions → weight ≈ 0.4–0.5

annoyance, confusion, disappointment, embarrassment, sadness, nervousness, remorse

Strong unrest-related emotions → weight ≈ 0.8–1.0

anger, disgust, disapproval, fear, grief

For each comment:

Collect all active emotion labels (those marked 1).

Map each to its weight.

Compute the mean of these weights.

Convert to a percentage:

unrest_percent = mean(weighted_emotions) * 100


This value (0–100) is used as the target for the neural network.

Later, when the model predicts a value in 
0
,
1
0,1 (via a sigmoid output), we scale it back to a percentage and can interpret it as:

“The model estimates X% unrest in this post.”


🔧 Planned Scripts (Not Yet Added)

The following Python scripts will eventually go inside src/:

1. prepare_goemotions_unrest.py

Loads raw GoEmotions TSV files (from data/raw/ or ../Goemotion/data/)

Merges train/dev/test splits

Applies the emotion weight mapping

Computes unrest_percent for each comment

Cleans text (basic preprocessing)

Splits into train/validation/test sets

Saves processed CSVs into data/processed/

2. train_unrest_model.py

Loads processed train/val/test CSVs

Vectorizes text using TF-IDF

Builds and trains a regression ANN:

Input: TF-IDF features

Hidden layers: Dense(256 → 64, ReLU)

Output: Dense(1, Sigmoid) → value in 
0
,
1
0,1 scaled to 0–100%

Evaluates the model with MAE, MSE, and R²

Saves the trained model and TF-IDF vectorizer into models/

3. evaluate_model.py (optional)

Loads the saved model and vectorizer

Runs predictions on new text or test data

Prints example posts + predicted unrest %

Generates plots or basic analysis


▶️ Execution Plan (Temporary)

Once scripts are added, the intended flow will be:

Preprocessing
python src/prepare_goemotions_unrest.py

Training
python src/train_unrest_model.py

Evaluation / Demo
python src/evaluate_model.py


📚 Documentation Notes (Temporary)

All documentation for the project will be placed in:

PublicUnrest/documentation/


This includes:

Dataset description

Emotion-to-unrest weighting table

Privacy & bias analysis

ANN architecture and training details

Evaluation results and figures

Final project report (PDF or DOCX)
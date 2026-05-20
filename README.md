# Credit Card Fraud Detection

## Business Problem

Credit card fraud costs financial institutions billions annually. This project builds a production-ready ML pipeline to detect fraudulent transactions in real time. The dataset is highly imbalanced (0.172% fraud), so standard accuracy is misleading — we optimize for recall and AUC.

## Dataset

Kaggle Credit Card Fraud Detection dataset. 284,807 transactions, 492 frauds. Features V1-V28 are PCA-transformed for anonymity. Time and Amount are raw.

Download from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Place the file at `data/creditcard.csv`.

## Project Structure

```
fraud-detection/
├── notebooks/
│   ├── 01_eda.ipynb          # Exploratory Data Analysis
│   └── 02_modeling.ipynb     # Model Training & Evaluation
├── src/
│   ├── __init__.py
│   ├── preprocess.py         # Data loading, SMOTE, scaling pipeline
│   ├── train.py              # Model training utilities
│   └── evaluate.py           # Metrics, ROC curves, confusion matrices
├── models/
│   └── .gitkeep              # Trained models saved here (gitignored binaries)
├── api/
│   ├── __init__.py
│   ├── main.py               # FastAPI application
│   └── schemas.py            # Pydantic v2 request/response schemas
├── data/
│   └── .gitkeep              # Place creditcard.csv here
├── requirements.txt
└── README.md
```

## Approach

1. **EDA** — Understand class imbalance, feature distributions, and correlations
2. **Preprocessing** — StandardScaler + SMOTE oversampling on training set only (no data leakage)
3. **Modeling** — Logistic Regression (baseline), Random Forest, XGBoost
4. **Evaluation** — Precision, Recall, F1, AUC-ROC with visual comparisons
5. **API** — FastAPI service for real-time inference with a single transaction

## Results (Example — run notebooks to get actual numbers)

| Model | Precision | Recall | F1 | AUC |
|---|---|---|---|---|
| Logistic Regression | ~0.88 | ~0.92 | ~0.90 | ~0.97 |
| Random Forest | ~0.94 | ~0.82 | ~0.87 | ~0.97 |
| XGBoost | ~0.93 | ~0.86 | ~0.89 | ~0.98 |

**Key insight:** XGBoost achieves the best AUC. For fraud detection, high recall is critical to minimize missed fraud cases (false negatives are more costly than false positives).

## Setup & Installation

```bash
git clone <repo>
cd fraud-detection
pip install -r requirements.txt
```

## Running the Notebooks

```bash
jupyter notebook notebooks/
```

Run `01_eda.ipynb` first for exploratory analysis, then `02_modeling.ipynb` to train and evaluate models.

## Running the API

First train and save models by running notebook `02_modeling.ipynb` end-to-end. Then:

```bash
uvicorn api.main:app --reload
```

API available at http://localhost:8000. Interactive docs at http://localhost:8000/docs.

## API Usage

Health check:

```bash
curl http://localhost:8000/health
```

Predict fraud probability for a transaction:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Time": 0.0,
    "V1": 0.0, "V2": 0.0, "V3": 0.0, "V4": 0.0, "V5": 0.0,
    "V6": 0.0, "V7": 0.0, "V8": 0.0, "V9": 0.0, "V10": 0.0,
    "V11": 0.0, "V12": 0.0, "V13": 0.0, "V14": 0.0, "V15": 0.0,
    "V16": 0.0, "V17": 0.0, "V18": 0.0, "V19": 0.0, "V20": 0.0,
    "V21": 0.0, "V22": 0.0, "V23": 0.0, "V24": 0.0, "V25": 0.0,
    "V26": 0.0, "V27": 0.0, "V28": 0.0,
    "Amount": 100.0
  }'
```

Example response:

```json
{
  "fraud_probability": 0.0123,
  "is_fraud": false,
  "model_used": "xgboost"
}
```

## Key Design Decisions

- **SMOTE applied to training set only** (never test set) to prevent data leakage
- **StandardScaler fitted on training set only**, then applied to test set
- **XGBoost deployed in API** (best AUC across all models)
- **Threshold 0.5 by default**; tune lower for higher recall (fewer missed frauds) at the cost of more false positives
- **Stratified train/test split** preserves the natural class imbalance in the test set for realistic evaluation

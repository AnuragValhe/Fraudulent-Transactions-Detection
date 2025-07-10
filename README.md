# Fraudulent Transaction Detection

Detecting rare Credit Card Frauds from highly imbalanced data using ML techniques like Decision Trees, Random Forests, and SMOTE oversampling.

---

## 📌 Dataset Summary
- **Source**: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- **Rows**: 284,807
- **Frauds**: Only 0.17% (492)
- **Features**: V1–V28 (anonymized), Time, Amount, Class (target)

---

## 📌 ML Approaches & Results

### 1. Baseline (Raw Data)
| Model               | Accuracy | Precision | Recall | F1 Score |
| ------------------- | -------- | --------- | ------ | -------- |
| Logistic Regression | 0.72     | 0.60      | 0.50   | 0.54     |
| Random Forest       | 0.80     | 0.65      | 0.55   | 0.59     |
| Decision Tree       | 0.75     | 0.62      | 0.51   | 0.56     |


→ Accuracy was misleading due to imbalance — models ignored frauds.

---

### 2. UnderSampling
- Legit class reduced to match fraud count → 473 + 473 = 946 records
- Boosted metrics, but lost data diversity

| Model         | F1 Score |
|---------------|----------|
| Decision Tree | 0.92     |

---

### 3. SMOTE Oversampling
- Minority class synthetically boosted via SMOTE  
- Final dataset: 550,380 rows  
- **Best model: Decision Tree** (F1: **0.98**, Recall: **0.99**)

---

## 📌 Final Model
- **Chosen**: Decision Tree on SMOTE-balanced data  
- **Saved using**: `pickle` for easy future inference

---

## 📌 Tech Stack

| Category         | Libraries                                 |
|------------------|-------------------------------------------|
| Data Handling     | `pandas`, `numpy`                         |
| Modeling          | `scikit-learn`, `imbalanced-learn`, `pickle` |
| Evaluation        | `confusion_matrix`, `f1_score`, `roc_auc_score` |
| Visualization     | `matplotlib`                             |


---

## 📌 Run Locally

#### 1. Clone the Repository

```bash
git clone https://github.com/AnuragValhe/Fraudulent-Transactions-Detection
cd Fraudulent-Transactions-Detection
```

#### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

#### 3. Run the Script / Notebook

```bash
python Fraud_Detection.py  # OR open Fraud Detection.ipynb
```

---
![Static Badge](https://img.shields.io/badge/Machine-Learning-blue)
![Static Badge](https://img.shields.io/badge/Python-green)
![Static Badge](https://img.shields.io/badge/Classification-Model-blue)
![Static Badge](https://img.shields.io/badge/Decision-Trees-green)
![Static Badge](https://img.shields.io/badge/SMOTE-blue)

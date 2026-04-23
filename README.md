---

# 📦 Retail Demand Forecasting & Inventory Optimization

## 📌 Overview

Demand uncertainty in retail often leads to overstocking or stockouts. This project presents an end-to-end machine learning system to **forecast product demand** and support **inventory optimization**.

The solution combines traditional machine learning, deep learning, and time-series forecasting to generate accurate, product-level predictions and actionable inventory metrics.

---
![Project Thumbnail](Retail.jpg)

---
## 🎯 Objectives

* Predict product demand using multiple modeling approaches
* Compare baseline, deep learning, and transformer-based models
* Incorporate time-series forecasting for improved accuracy
* Generate inventory planning metrics such as safety stock and reorder levels

---

## 🗂️ Dataset

* Retail dataset with product-level demand information
* Includes categorical, textual, and numerical features
* Preprocessing steps:

  * Missing value handling
  * Data cleaning and transformation
  * Correlation and variance analysis
  * Dimensionality reduction using PCA

---

## ⚙️ Methodology

### 🔍 1. Exploratory Data Analysis (EDA)

* Data profiling and statistical analysis
* Correlation heatmap and feature relationships
* Missing value analysis

---

### 🧠 2. Feature Engineering

* Categorical Encoding
* TF-IDF Vectorization
* Word2Vec Embeddings
* GloVe Embeddings
* PCA for dimensionality reduction

---

### 📊 3. Baseline Models

* Classification: Logistic Regression
* Regression: Random Forest

These models establish baseline performance across feature representations.

---

### 🤖 4. Advanced Models

#### Custom Deep Learning Model

* Designed a hybrid neural architecture for:

  * Classification
  * Regression

#### Transformer-Based Models

* Classification: DistilBERT + XGBoost
* Time-Series Forecasting: Chronos + XGBoost

---

### 🚀 Key Innovation

* Adapted Chronos for **product-level demand forecasting**
* Combined temporal modeling with gradient boosting
* Enabled forecasting that captures both:

  * Time-series patterns
  * Feature interactions

---

### 📦 5. Inventory Optimization

Based on forecasted demand:

* Safety Stock
* Lead Time Demand
* Reorder Quantity
* Stockout Risk

---

### 🌐 6. Deployment

* Model inference integrated with a REST API using FastAPI (team collaboration)
* Supports real-time prediction

---

## 📊 Model Performance & Comparison

```text
Model                     | Task                  | Technique / Features        | Metric (R² / Accuracy)
--------------------------|----------------------|-----------------------------|----------------------
Logistic Regression       | Classification        | Categorical Encoding        | 0.94
Random Forest             | Regression            | TF-IDF Features             | 0.999
Custom Deep Learning      | Classification        | Categorical Encoding        | 0.91
Custom Deep Learning      | Regression            | TF-IDF Features             | 0.98
DistilBERT + XGBoost      | NLP Classification    | Transformer + Boosting      | 0.92
Chronos + XGBoost         | Time-Series Forecast  | Temporal + Tabular Hybrid   | 0.89
```

### 📈 Key Insight

Traditional ML models performed strongly on static tabular features, while deep learning models provided stable performance across tasks. However, time-series forecasting using Chronos combined with XGBoost enabled capturing temporal demand patterns, making it more suitable for real-world inventory planning despite slightly lower R².

---

## 🖼️ Results & Visualizations

Key outputs included in `/assets`:

* EDA insights (correlation, distributions)
* Model performance comparisons
* Demand forecast visualization (Actual vs Predicted)
* Inventory planning outputs

---

## 🏗️ Project Structure

```text
├── notebooks/
│   └── retail_demand_forecasting.ipynb
├── assets/
│   ├── eda.png
│   ├── forecast.png
│   └── inventory.png
├── reports/
│   └── retail_demand_forecasting_report.pdf
├── api/
│   └── main.py
├── requirements.txt
└── README.md
```

---

## 👤 My Contributions

* Designed and implemented custom deep learning model
* Developed Chronos + XGBoost forecasting pipeline
* Performed EDA, feature engineering, and model evaluation
* Led project documentation and presentation
* Collaborated on API integration using FastAPI

---

## 🛠️ Tech Stack

* Python
* Scikit-learn
* TensorFlow / PyTorch
* XGBoost
* NLP Techniques (TF-IDF, Word2Vec, GloVe)
* Transformer Models (DistilBERT, Chronos)
* FastAPI

---

## 🔮 Future Improvements

* Full-scale deployment pipeline
* Real-time data ingestion
* Hyperparameter optimization
* Integration with dashboards for business users

---

## 📌 Conclusion

This project demonstrates how combining machine learning, deep learning, and time-series forecasting can address real-world retail challenges and support data-driven inventory planning decisions.

---

## 🤝 Connect

- [LinkedIn](https://www.linkedin.com/in/varsha-shekhar)
- [Gmail](varshaiyer96@gmail.com)



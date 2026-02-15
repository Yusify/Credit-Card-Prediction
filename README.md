# Credit Card Fraud Detection – Advanced Feature Optimization

![Credit Card Fraud Banner](https://github.com/yourusername/your-repo/raw/main/images/fraud_banner.jpg)  
*(Optional: Add a nice banner image here – you can find free ones on Unsplash or create one with Canva)*

**Revisiting the famous Kaggle Credit Card Fraud dataset — this time with deeper focus on feature engineering and optimization to beat my previous results.**

## 📌 Project Overview

This project uses the well-known [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) from Kaggle.  

The dataset is **extremely imbalanced** (~284,807 transactions, only **492 frauds** ≈ 0.172%). Features V1–V28 are PCA-transformed (anonymized), with raw 'Time' and 'Amount' also included.

In my earlier attempts, I achieved decent results with standard models and basic imbalance handling.  
**This time** I concentrated heavily on **feature optimization** — exploring transformations, selection methods, domain-inspired features, and their real impact on performance (especially PR-AUC, F1-score, and recall for the minority class).

Goal: Demonstrate how thoughtful feature work can meaningfully improve fraud detection without just throwing more complex models at the problem.

## 🔑 Key Focus: Feature Optimization Techniques Used

- **Exploratory analysis** of 'Time' (cyclic patterns → hour/day features) and 'Amount' (log / Box-Cox / standardization)
- **Scaling strategies** comparison (StandardScaler vs RobustScaler vs MinMax — especially important post-PCA)
- **Feature importance** analysis (tree-based models, permutation importance)
- **Feature selection** methods (e.g. SelectKBest, Recursive Feature Elimination, Boruta, variance threshold)
- **Custom / domain-inspired features** (e.g. transaction frequency proxies from Time, Amount clustering/binning)
- **Dimensionality reduction** experiments (further PCA tuning, t-SNE/UMAP visualization for intuition)
- Imbalance handling integrated with features: SMOTE / ADASYN / Borderline-SMOTE variants, undersampling, class weights

## 🛠️ Technologies & Libraries

- Python 3.9+
- pandas, numpy
- scikit-learn
- imbalanced-learn
- XGBoost / LightGBM / CatBoost
- matplotlib, seaborn, plotly (for visualizations)
- jupyter / colab

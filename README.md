# Employee Attrition Prediction  

This project builds a machine learning system to predict **employee attrition** (whether an employee is likely to leave the company) using the **IBM HR Analytics Employee Attrition Dataset**. The model helps HR teams understand key factors influencing attrition and take proactive measures to improve employee retention.  

## 🔍 Project Overview  
- **Goal**: Predict employee attrition and identify the most important factors that drive employees to leave.  
- **Dataset**: IBM HR Analytics Attrition Dataset (`WA_Fn-UseC_-HR-Employee-Attrition.csv`).  
- **Models Used**:  
  - Logistic Regression  
  - Random Forest Classifier  
  - XGBoost Classifier  
- **Explainability**: SHAP values used to explain model predictions at both global and individual employee levels.  

## ⚙️ Features  
- Data preprocessing (handling categorical features, scaling, train/test split).  
- Model training and evaluation with accuracy, precision, recall, and F1-score.  
- Feature importance visualization to highlight key attrition factors (e.g., OverTime, Job Satisfaction, Monthly Income).  
- Local explanations using **SHAP force plots** to interpret individual predictions.  

## 📊 Results  
- Best performance achieved with **XGBoost**.  
- Key drivers of attrition:  
  - OverTime (Yes)  
  - Low Job Satisfaction  
  - Low Monthly Income  
- Explainable AI techniques provide transparency into decision-making.  

## 🛠️ Tools & Libraries  
- Python  
- Pandas, NumPy  
- Scikit-learn  
- XGBoost  
- SHAP  
- Matplotlib, Seaborn  

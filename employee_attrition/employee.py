# Employee Attrition Prediction
# Complete, runnable Python script / notebook cells.
# Dataset: IBM HR Analytics Employee Attrition (WA_Fn-UseC_-HR-Employee-Attrition.csv)
# This script includes code to download from Kaggle (requires Kaggle API token) and full ML pipeline:
# - preprocessing
# - Logistic Regression, Random Forest, XGBoost
# - evaluation (accuracy, precision, recall, f1, roc_auc)
# - SHAP explainability (global + local)
# - business impact calculation

# --- Install required packages (run once in your environment) ---
#!pip install pandas numpy scikit-learn xgboost shap matplotlib seaborn kaggle joblib

# ------------------------------------------------------------------
# NOTE about Kaggle download:
# 1) Create an account on Kaggle and go to "Account" -> "API" -> "Create New API Token".
#    This downloads a kaggle.json file with your credentials.
# 2) Either place kaggle.json in ~/.kaggle/kaggle.json OR set environment variables KAGGLE_USERNAME and KAGGLE_KEY.
# 3) The code below will try to download the dataset automatically. If it fails, download it manually and place
#    the CSV in the same folder as this script/notebook, named: WA_Fn-UseC_-HR-Employee-Attrition.csv
# ------------------------------------------------------------------

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib

# ------------------ Helper: download from Kaggle ------------------


# ------------------ Load data ------------------

CSV_FILENAME = 'WA_Fn-UseC_-HR-Employee-Attrition.csv'



# Load
df = pd.read_csv(CSV_FILENAME)
print('Dataset loaded. Shape:', df.shape)
print('Columns:', df.columns.tolist())

# ------------------ Quick EDA ------------------
print('\nTarget distribution:')
print(df['Attrition'].value_counts())

# Convert target to binary
df['Attrition_flag'] = df['Attrition'].map({'Yes':1, 'No':0})

# Drop columns that are identifiers or not useful for modeling
# Here, EmployeeNumber is an ID. Overly redundant columns can be dropped later if needed.
if 'EmployeeNumber' in df.columns:
    df = df.drop(columns=['EmployeeNumber'])

# ------------------ Feature Engineering & Preprocessing ------------------
# We'll use pandas.get_dummies for categorical variables for simplicity.

# Identify categorical and numerical features
cat_cols = df.select_dtypes(include=['object']).columns.tolist()
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# Remove target and Attrition (string) from lists
if 'Attrition' in cat_cols:
    cat_cols.remove('Attrition')
if 'Attrition_flag' in num_cols:
    num_cols.remove('Attrition_flag')

print('\nCategorical cols:', cat_cols)
print('Numerical cols:', num_cols)

# For simplicity: convert categorical variables using get_dummies (drop first to avoid multicollinearity)
df_model = pd.get_dummies(df.drop(columns=['Attrition']), columns=cat_cols, drop_first=True)

# Separate X and y
X = df_model.drop(columns=['Attrition_flag'])
y = df_model['Attrition_flag']

print('\nAfter get_dummies, feature count:', X.shape[1])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Scale numeric features
scaler = StandardScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

# ------------------ Model Training ------------------

# We'll train three models: Logistic Regression, Random Forest, XGBoost
models = {}

print('\nTraining Logistic Regression...')
lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
lr.fit(X_train, y_train)
models['LogisticRegression'] = lr

print('Training Random Forest...')
rf = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
rf.fit(X_train, y_train)
models['RandomForest'] = rf

print('Training XGBoost...')
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb.fit(X_train, y_train)
models['XGBoost'] = xgb

# Save scalers and models
joblib.dump(scaler, 'scaler.joblib')
joblib.dump(lr, 'logistic_model.joblib')
joblib.dump(rf, 'rf_model.joblib')
joblib.dump(xgb, 'xgb_model.joblib')

# ------------------ Evaluation ------------------

def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = None
    try:
        y_proba = model.predict_proba(X_test)[:,1]
    except Exception:
        # Some models (or wrappers) might not have predict_proba
        y_proba = model.decision_function(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_proba)

    print(f"\nModel: {name}")
    print('Accuracy:', round(acc,4))
    print('Precision:', round(prec,4))
    print('Recall:', round(rec,4))
    print('F1:', round(f1,4))
    print('ROC-AUC:', round(roc,4))
    print('\nClassification Report:')
    print(classification_report(y_test, y_pred))

    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title(f'Confusion Matrix: {name}')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    plt.tight_layout()
    plt.savefig(f'confusion_{name}.png')
    plt.close()

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0,1],[0,1],'--')
    plt.title(f'ROC Curve: {name}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.savefig(f'roc_{name}.png')
    plt.close()

for name, model in models.items():
    evaluate_model(name, model, X_test, y_test)

# ------------------ Feature Importance & SHAP ------------------
print('\nComputing feature importances...')

# --- Logistic Regression coefficients ---
coef_df = pd.DataFrame({
    'feature': X.columns,
    'coef': lr.coef_[0]
}).sort_values(by='coef', ascending=False)

print('\nTop 10 positive coefficients (logistic regression):')
print(coef_df.head(10))
print('\nTop 10 negative coefficients (logistic regression):')
print(coef_df.tail(10))

# --- Random Forest feature importances ---
rf_imp = pd.DataFrame({
    'feature': X.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print('\nTop 10 Random Forest features:')
print(rf_imp.head(10))

# --- XGBoost feature importances ---
xgb_imp = pd.DataFrame({
    'feature': X.columns,
    'importance': xgb.feature_importances_
}).sort_values('importance', ascending=False)

print('\nTop 10 XGBoost features:')
print(xgb_imp.head(10))

# --- Ensure all X_test columns are numeric for SHAP ---
X_train_numeric = X_train.astype(float)
X_test_numeric = X_test.astype(float)

# --- SHAP for XGBoost ---
print('\nRunning SHAP... (this may take a moment)')
explainer = shap.TreeExplainer(xgb)
# sample for speed
X_sample = X_train_numeric.sample(n=min(200, X_train_numeric.shape[0]), random_state=42)
shap_values = explainer.shap_values(X_sample)

# Global summary plot (saved to PNG)
shap.summary_plot(shap_values, X_sample, show=False)
plt.tight_layout()
plt.savefig('shap_summary.png')
plt.close()

# Local explanation: highest-risk test employee
probas = xgb.predict_proba(X_test_numeric)[:,1]
at_risk_idx = np.argmax(probas)
print(f'Highest-risk test index (local explain): {at_risk_idx}, prob={probas[at_risk_idx]:.4f}')

# SHAP force plot for the single employee
force_html = shap.force_plot(
    explainer.expected_value,
    explainer.shap_values(X_test_numeric.iloc[[at_risk_idx]]),  # double brackets ensure 2D
    X_test_numeric.iloc[[at_risk_idx]],
    matplotlib=False
)

shap.save_html('shap_force_highest_risk.html', force_html)
print('Saved shap_summary.png and shap_force_highest_risk.html')

# ------------------ Business Impact Calculation ------------------
print('\nBusiness impact calculation example:')

num_employees = df.shape[0]
current_attrition_rate = y.mean()
current_attritions = int(round(num_employees * current_attrition_rate))
avg_replacement_cost = 50000  # example; change as needed

total_cost = current_attritions * avg_replacement_cost
savings_10pct = 0.10 * current_attritions * avg_replacement_cost

print(f"Dataset employees: {num_employees}")
print(f"Current attrition rate: {current_attrition_rate:.2%} -> approx {current_attritions} leavers")
print(f"Average replacement cost (assumed): ${avg_replacement_cost:,}")
print(f"Total annual cost due to attrition (approx): ${total_cost:,}")
print(f"If we reduce attrition by 10%, estimated savings: ${savings_10pct:,.0f}")

# Save top feature lists to CSV for reporting
coef_df.to_csv('logistic_coefficients.csv', index=False)
rf_imp.to_csv('rf_feature_importances.csv', index=False)
xgb_imp.to_csv('xgb_feature_importances.csv', index=False)

print('\nAll outputs saved: model files, plots (confusion, roc), shap summary, shap force html, csv exports.')

# ------------------ Example: Predict single new employee ------------------
# To predict for a new employee, prepare a single-row dataframe with the same columns as X
# Example (build a dummy from mean values):
new_emp = X.mean().to_frame().T
pred_prob = xgb.predict_proba(new_emp)[:,1][0]
print(f'Example new-employee predicted attrition probability: {pred_prob:.4f}')

print('\nScript complete. Check the working directory for saved artifacts (models, plots, csvs).')

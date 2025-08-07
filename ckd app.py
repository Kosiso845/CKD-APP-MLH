import pandas as pd
import numpy as np
df=pd.read_csv('Chronic_Kidney_Dsease_data.csv')
df.describe
df.info
df.head
df.replace("?", pd.NA, inplace=True)
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()
print(df.columns.tolist())

df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

print(df.columns.tolist())

import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(x='diagnosis', data=df)
plt.title("Target Variable Distribution")
plt.xticks(rotation=45)  # Optional: Rotate x-axis labels if needed
plt.show()

import matplotlib.pyplot as plt
numeric_cols = df.select_dtypes(include='number').columns
df[numeric_cols].hist(figsize=(20, 15), bins=30, edgecolor='black')
plt.suptitle("Histograms of Numeric Features", fontsize=16)
plt.tight_layout()
plt.show()
important_features = [
    'serumcreatinine', 'bunlevels', 'gfr', 'acr', 'proteininurine',
    'hemoglobinlevels', 'systolicbp', 'diastolicbp', 'hba1c'
]


for col in important_features:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=df[col])
    plt.title(f'Boxplot of {col}')
    plt.tight_layout()
    plt.show()
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['diagnosis'] = le.fit_transform(df['diagnosis']) 
bins = [0, 15, 30, 60, 90, 120]
labels = ['Stage 5', 'Stage 4', 'Stage 3', 'Stage 2', 'Stage 1']
df['gfr_stage'] = pd.cut(df['gfr'], bins=bins, labels=labels)
df['gfr_stage'] = df['gfr_stage'].astype(str)
df['gfr_stage'] = LabelEncoder().fit_transform(df['gfr_stage'])
X = df.drop(['patientid', 'doctorincharge', 'diagnosis'], axis=1)
y = df['diagnosis']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print(" Random Forest Classification Report:")
print(classification_report(y_test, y_pred_rf))
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Precision:", precision_score(y_test, y_pred_rf))
print("Recall:", recall_score(y_test, y_pred_rf))
print("F1 Score:", f1_score(y_test, y_pred_rf))
rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred_rf))

from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print("Before SMOTE:", y_train.value_counts())
print("After SMOTE:", y_train_resampled.value_counts())
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_resampled, y_train_resampled)


y_pred_rf = rf.predict(X_test)
print("Random Forest After SMOTE:")
print(classification_report(y_test, y_pred_rf))
from imblearn.ensemble import BalancedRandomForestClassifier
brf = BalancedRandomForestClassifier(n_estimators=100, random_state=42)
brf.fit(X_train, y_train)
y_pred_brf = brf.predict(X_test)
print(" Balanced Random Forest Results:")
print(classification_report(y_test, y_pred_brf))
from sklearn.metrics import classification_report, confusion_matrix
cm = confusion_matrix(y_test, y_pred_brf)
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples')
plt.title("Confusion Matrix - Balanced Random Forest")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}
brf = BalancedRandomForestClassifier(random_state=42)
grid_search = GridSearchCV(
    estimator=brf,
    param_grid=param_grid,
    cv=3,  
    scoring='f1_macro', 
    n_jobs=-1,  
    verbose=1
)
grid_search.fit(X_train, y_train)
print(" Best Parameters:", grid_search.best_params_)
print(" Best F1 Score (macro):", grid_search.best_score_)
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)
print(" Evaluation of Tuned Model:")
print(classification_report(y_test, y_pred_best))
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
log_reg.fit(X_train, y_train)
y_pred_log = log_reg.predict(X_test)
print("Logistic Regression (class_weight='balanced') Results:")
print(classification_report(y_test, y_pred_log))
cm = confusion_matrix(y_test, y_pred_log)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - Logistic Regression")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
param_grid = {
    'penalty': ['l1', 'l2'],
    'C': [0.01, 0.1, 1, 10, 100],  
    'solver': ['liblinear'],     
    'class_weight': ['balanced']
}
logreg = LogisticRegression(max_iter=1000, random_state=42)

grid_search = GridSearchCV(
    estimator=logreg,
    param_grid=param_grid,
    scoring='f1_macro',
    cv=3,
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)
print("Best Parameters:", grid_search.best_params_)
print(" Best F1 Macro Score:", grid_search.best_score_)
best_logreg = grid_search.best_estimator_
y_pred_best_logreg = best_logreg.predict(X_test)
print(" Fine-Tuned Logistic Regression:")
print(classification_report(y_test, y_pred_best_logreg))
import joblib

joblib.dump(best_model, "tuned_balanced_rf_ckd.pkl")
import os
if os.path.exists("tuned_balanced_rf_ckd.pkl"):
    print("✅ Model file saved successfully!")
else:
    print("❌ Model file not found.")

X = df[['aceinhibitors', 'alcoholconsumption', 'antidiabeticmedications', 'cholesterolhdl', 'cholesteroltotal', ...]]

print(df.columns.tolist())

feature_columns = [
    'age', 'gender', 'ethnicity', 'socioeconomicstatus', 'educationlevel',
    'bmi', 'smoking', 'alcoholconsumption', 'physicalactivity', 'dietquality',
    'sleepquality', 'familyhistorykidneydisease', 'familyhistoryhypertension',
    'familyhistorydiabetes', 'previousacutekidneyinjury', 'urinarytractinfections',
    'systolicbp', 'diastolicbp', 'fastingbloodsugar', 'hba1c', 'serumcreatinine',
    'bunlevels', 'gfr', 'proteininurine', 'acr', 'serumelectrolytessodium',
    'serumelectrolytespotassium', 'serumelectrolytescalcium',
    'serumelectrolytesphosphorus', 'hemoglobinlevels', 'cholesteroltotal',
    'cholesterolldl', 'cholesterolhdl', 'cholesteroltriglycerides',
    'aceinhibitors', 'diuretics', 'nsaidsuse', 'statins', 'antidiabeticmedications',
    'edema', 'fatiguelevels', 'nauseavomiting', 'musclecramps', 'itching',
    'qualityoflifescore', 'heavymetalsexposure', 'occupationalexposurechemicals',
    'waterquality', 'medicalcheckupsfrequency', 'medicationadherence',
    'healthliteracy'
]

X = df[feature_columns]
y = df['gfr_stage']  # Or 'diagnosis', whichever is your label

model.fit(X, y)

from sklearn.ensemble import RandomForestClassifier

# Assuming df is your cleaned DataFrame
feature_columns = [
    'age', 'gender', 'ethnicity', 'socioeconomicstatus', 'educationlevel',
    'bmi', 'smoking', 'alcoholconsumption', 'physicalactivity', 'dietquality',
    'sleepquality', 'familyhistorykidneydisease', 'familyhistoryhypertension',
    'familyhistorydiabetes', 'previousacutekidneyinjury', 'urinarytractinfections',
    'systolicbp', 'diastolicbp', 'fastingbloodsugar', 'hba1c', 'serumcreatinine',
    'bunlevels', 'gfr', 'proteininurine', 'acr', 'serumelectrolytessodium',
    'serumelectrolytespotassium', 'serumelectrolytescalcium',
    'serumelectrolytesphosphorus', 'hemoglobinlevels', 'cholesteroltotal',
    'cholesterolldl', 'cholesterolhdl', 'cholesteroltriglycerides',
    'aceinhibitors', 'diuretics', 'nsaidsuse', 'statins', 'antidiabeticmedications',
    'edema', 'fatiguelevels', 'nauseavomiting', 'musclecramps', 'itching',
    'qualityoflifescore', 'heavymetalsexposure', 'occupationalexposurechemicals',
    'waterquality', 'medicalcheckupsfrequency', 'medicationadherence',
    'healthliteracy'
]

X = df[feature_columns]
y = df['gfr_stage']  # or 'diagnosis'

model = RandomForestClassifier(random_state=42)
model.fit(X, y)

joblib.dump((model, feature_columns), 'ckd_model.pkl')
print(" Model saved successfully as 'ckd_model.pkl'")

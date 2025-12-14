# ============================================
# Python Implementation - Heart Disease Analysis
# Models: Logistic Regression, Random Forest, Gradient Boosting
# ============================================

import pandas as pd
import numpy as np
import random
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import (
    roc_auc_score, accuracy_score,
    precision_score, recall_score, f1_score,
    confusion_matrix
)

df = pd.read_csv("heart_disease.csv")

print(df.head())
print(df.info())
print(df.describe())

# Drop missing values & duplicates
df = df.dropna().drop_duplicates()

# BMI Winsorization (1–99%)
bmi_low, bmi_high = df["BMI"].quantile([0.01, 0.99])
df["BMI_capped"] = df["BMI"].clip(bmi_low, bmi_high)

# GenHlth factor
genhlth_map = {1: "Excellent", 2: "Very Good", 3: "Good", 4: "Fair", 5: "Poor"}
df["GenHlth_Factor"] = df["GenHlth"].map(genhlth_map)

# Filter invalid MentHlth & PhysHlth
df = df[(df["MentHlth"] <= 30) & (df["PhysHlth"] <= 30)]

# BMI Category
df["BMI_Category"] = pd.cut(
    df["BMI"],
    bins=[0, 18.5, 25, 30, np.inf],
    labels=["Underweight", "Normal", "Overweight", "Obese"]
)

# AgeGroup
age_labels = [
    "18-24","25-29","30-34","35-39","40-44","45-49",
    "50-54","55-59","60-64","65-69","70-74","75-79","80+"
]
df["AgeGroup"] = df["Age"].astype(int).apply(lambda x: age_labels[x-1])

# AgeBand
df["AgeBand"] = pd.cut(
    df["Age"],
    bins=[0,3,6,9,11,14],
    labels=["18-34","35-49","50-64","65-74","75+"]
)

# RiskScore
df["RiskScore"] = (
    df["Smoker"] +
    df["HvyAlcoholConsump"] +
    (1 - df["PhysActivity"]) +
    (1 - df["Fruits"]) +
    (1 - df["Veggies"])
)

# DiseaseCount
df["DiseaseCount"] = (
    df["HighBP"] + df["HighChol"] +
    df["Diabetes"] + df["Stroke"]
)

# HealthStressIndex
df["HealthStressIndex"] = df["MentHlth"] + df["PhysHlth"]

# HealthcareScore
df["HealthcareScore"] = df["AnyHealthcare"] + (1 - df["NoDocbcCost"])

# ObeseFlag
df["ObeseFlag"] = (df["BMI"] >= 30).astype(int)

# LifestyleProfile
df["LifestyleProfile"] = pd.cut(
    df["RiskScore"],
    bins=[-1,1,3,10],
    labels=["Healthy","ModerateRisk","HighRisk"]
)

target_col = "HeartDiseaseorAttack"

numeric_features = [
    "Age","PhysHlth","MentHlth","HealthStressIndex",
    "DiseaseCount","ObeseFlag","RiskScore","BMI",
    "Sex","HighBP","HighChol","Diabetes","Stroke",
    "Smoker","PhysActivity"
]

categorical_features = ["AgeGroup","AgeBand","LifestyleProfile"]

X = df[numeric_features + categorical_features]
y = df[target_col]

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=123
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=123
)

def evaluate_model(y_true, y_pred, y_proba, name="Model"):
    print(f"\n==== {name} ====")
    print("AUC       :", roc_auc_score(y_true, y_proba))
    print("Accuracy  :", accuracy_score(y_true, y_pred))
    print("Precision :", precision_score(y_true, y_pred))
    print("Recall    :", recall_score(y_true, y_pred))
    print("F1        :", f1_score(y_true, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))


lr_preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

lr_model = Pipeline(steps=[
    ("preprocess", lr_preprocess),
    ("model", LogisticRegression(
        max_iter=100,
        C=1.0,
        penalty="elasticnet",
        l1_ratio=0.5,
        solver="saga"
    ))
])

lr_model.fit(X_train, y_train)

y_pred = lr_model.predict(X_test)
y_proba = lr_model.predict_proba(X_test)[:,1]

evaluate_model(y_test, y_pred, y_proba, "Logistic Regression")

rf_preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ],
    remainder="passthrough"
)

rf_model = Pipeline(steps=[
    ("preprocess", rf_preprocess),
    ("model", RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_leaf=5,
        random_state=123
    ))
])

rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)
y_proba = rf_model.predict_proba(X_test)[:,1]

evaluate_model(y_test, y_pred, y_proba, "Random Forest")

gbt_model = Pipeline(steps=[
    ("preprocess", rf_preprocess),
    ("model", GradientBoostingClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=123
    ))
])

gbt_model.fit(X_train, y_train)

y_pred = gbt_model.predict(X_test)
y_proba = gbt_model.predict_proba(X_test)[:,1]

evaluate_model(y_test, y_pred, y_proba, "Gradient Boosting")



# Save model
joblib.dump(gbt_model, "gbt_model.pkl")

print("GBT model saved successfully as gbt_model.pkl")

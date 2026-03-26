# ============================================================
# LOGISTIC REGRESSION MODEL - DIABETES PREDICTION DATASET
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# --- MEMBER 2: DATA CLEANING --- (Praveen)


# --- MEMBER 3: VISUALIZATION / EDA --- (Harshvardhini)


# --- MEMBER 4: FEATURE PREPARATION --- (Rohal)

# Separate features and target
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("=== FEATURE PREPARATION DONE ===")
print("Features shape:", X_scaled.shape)
print("Target shape:", y.shape)
print("Sample scaled values:\n", X_scaled[:3])

# --- MEMBER 5: SPLITTING & TRAINING --- (Mathdevru)


# --- MEMBER 6: EVALUATION --- (Hemanth)
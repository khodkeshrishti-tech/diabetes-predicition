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

df = pd.read_csv("diabetes.csv")

print("=== BEFORE CLEANING ===")
print(df.isnull().sum())
print("Duplicate rows:", df.duplicated().sum())

# Drop duplicates
df = df.drop_duplicates()

# Replace 0s with column mean for these columns
cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in cols_with_zeros:
    df[col] = df[col].replace(0, df[col].mean())

print("\n=== AFTER CLEANING ===")
print(df.isnull().sum())
print("Shape:", df.shape)

# --- MEMBER 3: VISUALIZATION / EDA --- (Harshvardhini)


# --- MEMBER 4: FEATURE PREPARATION --- (Rohal)


# --- MEMBER 5: SPLITTING & TRAINING --- (Mathdevru)


# --- MEMBER 6: EVALUATION --- (Hemanth)
code for @46665440481483.                     # Make predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("=== MODEL EVALUATION ===")
print(f"Accuracy Score: {accuracy * 100:.2f}%")

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Not Diabetic', 'Diabetic'],
            yticklabels=['Not Diabetic', 'Diabetic'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('chart_confusion_matrix.png')
plt.show()
print("Confusion Matrix saved!")
import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score
)


# ---------------------------------------------------------
# 1. Load Dataset
# ---------------------------------------------------------

DATA_PATH = "data/dataset.csv"

df = pd.read_csv(DATA_PATH)

print("Dataset:")
print(df)

print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())


# ---------------------------------------------------------
# 2. Separate Features and Target
# ---------------------------------------------------------

X = df[["feature_1", "feature_2"]]
y = df["target"]


# ---------------------------------------------------------
# 3. Train-Test Split
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)


# ---------------------------------------------------------
# 4. Feature Scaling
# ---------------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ---------------------------------------------------------
# 5. Model Selection and Training
# ---------------------------------------------------------

model = LogisticRegression(random_state=42)

model.fit(X_train_scaled, y_train)


# ---------------------------------------------------------
# 6. Predictions
# ---------------------------------------------------------

y_pred = model.predict(X_test_scaled)

y_probability = model.predict_proba(X_test_scaled)[:, 1]


# ---------------------------------------------------------
# 7. Evaluation Metrics
# ---------------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

auc = roc_auc_score(y_test, y_probability)

print("\nModel Evaluation")
print("---------------------------")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-Score  : {f1:.4f}")
print(f"ROC-AUC   : {auc:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# ---------------------------------------------------------
# 8. Create Visualization Directory
# ---------------------------------------------------------

os.makedirs("visualizations", exist_ok=True)


# ---------------------------------------------------------
# 9. Confusion Matrix
# ---------------------------------------------------------

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Class 0", "Class 1"],
    yticklabels=["Class 0", "Class 1"]
)

plt.title("Confusion Matrix - Logistic Regression")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

plt.tight_layout()

plt.savefig(
    "visualizations/confusion_matrix.png",
    dpi=300
)

plt.close()


# ---------------------------------------------------------
# 10. ROC Curve
# ---------------------------------------------------------

fpr, tpr, thresholds = roc_curve(
    y_test,
    y_probability
)

plt.figure(figsize=(7, 5))

plt.plot(
    fpr,
    tpr,
    color="blue",
    linewidth=2,
    label=f"Logistic Regression (AUC = {auc:.2f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    color="gray",
    linestyle="--",
    label="Random Classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Logistic Regression")

plt.legend(loc="lower right")

plt.tight_layout()

plt.savefig(
    "visualizations/roc_curve.png",
    dpi=300
)

plt.close()


print("\nVisualizations saved successfully.")
print("1. visualizations/confusion_matrix.png")
print("2. visualizations/roc_curve.png")

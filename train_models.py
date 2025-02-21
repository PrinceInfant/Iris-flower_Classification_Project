import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
import os

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models
models = {
    "SVM": SVC(kernel="linear").fit(X_train, y_train),
    "Random Forest": RandomForestClassifier(n_estimators=100).fit(X_train, y_train),
    "Decision Tree": DecisionTreeClassifier().fit(X_train, y_train),
    "KNN": KNeighborsClassifier(n_neighbors=3).fit(X_train, y_train),
}

# Create the 'models' directory if it doesn't exist
if not os.path.exists("models"):
    os.makedirs("models")

# Save models
for name, model in models.items():
    filename = f"models/{name.lower().replace(' ', '_')}_model.pkl"
    joblib.dump(model, filename)

print("✅ Models trained and saved successfully!")

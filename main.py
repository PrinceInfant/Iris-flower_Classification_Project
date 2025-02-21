from src.data_preprocessing import load_and_preprocess_data
from src.model_training import train_models
from src.model_evaluation import evaluate_models

DATA_PATH = "data/Iris.csv"

print("Starting the Iris Flower Classification Process...\n")

# Load and preprocess data
print("Loading and preprocessing data...")
X_train, X_test, y_train, y_test = load_and_preprocess_data(DATA_PATH)
print("Data loading and preprocessing complete!\n")

# Train models
print("Training models...")
models = train_models(X_train, y_train)  
print("Model training completed!\n")

# Evaluate models
print("Evaluating models...")
evaluate_models(models, X_test, y_test)  # Now correctly passing three arguments
print("Model evaluation completed!\n")

print("Process completed successfully!")

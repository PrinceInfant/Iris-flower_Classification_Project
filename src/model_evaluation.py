from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def evaluate_models(models, X_test, y_test):  # Ensure function expects three arguments
    print("Starting model evaluation...\n")

    for name, model in models.items():
        print(f" Evaluating {name} model...")
        y_pred = model.predict(X_test)
        
        print(f"{name} Model Accuracy: {accuracy_score(y_test, y_pred):.2f}")
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
        print("Classification Report:\n", classification_report(y_test, y_pred))
        print("-" * 50 + "\n")

    print("Model evaluation completed!\n")

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

def train_models(X_train, y_train):  # Ensure function expects two arguments
    print("Initializing models...")
    
    models = {
        "SVM": SVC(kernel='linear'),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "DecisionTree": DecisionTreeClassifier(),
        "KNN": KNeighborsClassifier(n_neighbors=5)
    }

    trained_models = {}

    for name, model in models.items():
        print(f"Training {name} model...")
        model.fit(X_train, y_train)
        trained_models[name] = model
        print(f"{name} model training complete!\n")

    print("All models trained successfully!\n")
    return trained_models

from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load all trained models
models = {
    "SVM": joblib.load("models/svm_model.pkl"),
    "RandomForest": joblib.load("models/random_forest_model.pkl"),
    "DecisionTree": joblib.load("models/decision_tree_model.pkl"),
    "KNN": joblib.load("models/knn_model.pkl")
}

# Map class labels to actual flower names
flower_names = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", models=models.keys())

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get form input values
        sepal_length = float(request.form["sepal_length"])
        sepal_width = float(request.form["sepal_width"])
        petal_length = float(request.form["petal_length"])
        petal_width = float(request.form["petal_width"])
        selected_model = request.form["model"]

        # Prepare input data
        input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

        # Predict flower class
        predicted_class = models[selected_model].predict(input_data)[0]
        prediction = flower_names.get(predicted_class, "Unknown")

        return jsonify({"prediction": prediction})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)

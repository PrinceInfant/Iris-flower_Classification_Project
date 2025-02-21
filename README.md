# Iris Flower Classification


##  Project Overview
The **Iris Flower Classification** project is a machine learning model that predicts the species of an iris flower (Setosa, Versicolor, or Virginica) based on its measurements (sepal length, sepal width, petal length, petal width). The dataset used for this project is sourced from Kaggle.


##  Dataset
- Dataset: [Iris Dataset on Kaggle](https://www.kaggle.com/datasets/saurabh00007/iriscsv)
- Columns:
  - Sepal Length
  - Sepal Width
  - Petal Length
  - Petal Width
  - Species (Target variable)


###  Install Dependencies
Before running the project, install the required dependencies by executing the following command:

```bash
pip install -r requirements.txt
```

Ensure your `requirements.txt` file contains:

```
flask
numpy
pandas
scikit-learn
joblib
```


### Run `main.py` (Preprocessing & Training)
Execute the following command to preprocess the dataset and train the models:

```bash
python main.py

```
This script will:
- Load and preprocess the Iris dataset.
- Train multiple machine learning models.
- Evaluate the trained models.


### Run `train_models.py` (Model Training & Saving)
Alternatively, you can directly train and save models using:

```bash
python train_models.py

```
This script will:
- Train SVM, Random Forest, Decision Tree, and KNN classifiers.
- Save the trained models in the `models/` directory.


### Start the Flask App
Once the models are trained and saved, launch the web application:

```bash
python app.py

```
Upon successful execution, the Flask app will start and display:


### Open the Web App in a Browser
Copy the given URL and open it in your browser:

🔗 [http://127.0.0.1:5000/]

This will load the web-based interface where you can enter the feature values and select a model to predict the flower species.

---

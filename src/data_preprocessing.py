import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(filepath):
    print("Loading dataset from:", filepath)
    
    # Load dataset
    df = pd.read_csv(filepath)
    
    # Drop unnecessary columns
    if 'Id' in df.columns:
        df.drop(columns=['Id'], inplace=True)
    
    # Check for missing values
    if df.isnull().sum().sum() == 0:
        print("No missing values found.")
    else:
        print("Warning: Missing values detected! Filling missing values...")
        df.fillna(df.mean(), inplace=True)

    # Split data into features and labels
    X = df.drop(columns=['Species'])
    y = df['Species']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Data successfully preprocessed!\n")
    
    return X_train, X_test, y_train, y_test  
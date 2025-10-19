import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os

def load_model(model_path='models/iris_model.pkl'):
    """Load trained model"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    with open(model_path, 'rb') as f:
        return pickle.load(f)

def load_data():
    """Load IRIS data from Week 3 feast setup"""
    data_path = 'data/iris.csv'
    df = pd.read_csv(data_path)
    return df

def evaluate_model():
    """Evaluate model performance"""
    # Load model and data
    model = load_model()
    df = load_data()
    
    # Prepare features and target
    feature_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    X = df[feature_cols]
    y = df['species']
    
    # Split data (same split as training)
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Make predictions
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    # Print results
    print("="*50)
    print("MODEL EVALUATION REPORT")
    print("="*50)
    print(f"\nAccuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))
    
    return accuracy, y_test, predictions

if __name__ == "__main__":
    evaluate_model()
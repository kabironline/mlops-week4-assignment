import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

def load_data():
    """Load IRIS data from Week 3"""
    data_path = 'data/iris.csv'
    df = pd.read_csv(data_path)
    return df

def train_model():
    """Train IRIS classification model"""
    # Load data
    df = load_data()
    
    # Prepare features and target
    feature_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    X = df[feature_cols]
    y = df['species']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Save model
    os.makedirs('models', exist_ok=True)
    with open('models/iris_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Model trained successfully!")
    print(f"Training accuracy: {model.score(X_train, y_train):.4f}")
    print(f"Test accuracy: {model.score(X_test, y_test):.4f}")
    
    return model

if __name__ == "__main__":
    train_model()
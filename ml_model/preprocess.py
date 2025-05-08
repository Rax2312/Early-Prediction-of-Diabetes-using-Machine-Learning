import pandas as pd
import numpy as np

def preprocess_input(input_data):
    """Replicates exact preprocessing from model training"""
    
    # Binary features mapping (1/2 encoding from your training)
    binary_map = {
        'Sex': {'Male': 1, 'Female': 2},
        'Smoker': {'No': 1, 'Yes': 2},
        'Diet Habit': {'Non-Vegetarian': 1, 'Vegetarian': 2},
        'Family History': {'No': 1, 'Yes': 2},
        'Acanthosis Nigricans': {'No': 1, 'Yes': 2}
    }
    
    processed = {}
    
    # Numeric features (already binned 1-4 during training)
    numeric_features = [
        'Age', 'BMI', 'BloodPressure', 'Glucose', 'Insulin',
        'Pulse Rate', 'Skin_Thickness', 'DiabetesPedigreeFunction',
        'RBC', 'Pregnancies', 'HbA1c'
    ]
    
    for feature in numeric_features:
        # Convert raw value to binned category (1-4)
        val = float(input_data[feature])
        if val <= 1.75:
            processed[feature] = 1
        elif val <= 2.5:
            processed[feature] = 2
        elif val <= 3.25:
            processed[feature] = 3
        else:
            processed[feature] = 4
    
    # Binary features
    for feature in binary_map:
        processed[feature] = binary_map[feature][input_data[feature]]
    
    # Maintain exact feature order expected by model
    feature_order = [
        'Age', 'BMI', 'BloodPressure', 'Glucose', 'Insulin',
        'Pulse Rate', 'Skin_Thickness', 'DiabetesPedigreeFunction',
        'RBC', 'Pregnancies', 'HbA1c', 'Sex', 'Smoker',
        'Diet Habit', 'Family History', 'Acanthosis Nigricans'
    ]
    
    return np.array([[processed[feature] for feature in feature_order]])
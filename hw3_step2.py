# hw3_step2_lr.py
# Logistic Regression implementation for mobile phone price prediction

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Function to build Logistic Regression model
def build_logistic_regression(labeled, challenge):
    """
    Process training and challenge data, selecting the best features
    
    Args:
        labeled: DataFrame containing labeled training data
        challenge: DataFrame containing challenge data for prediction
        
    Returns:
        labeled: DataFrame with selected features
        challenge: DataFrame with selected features
    """
    # Best features based on analysis
    features = ['ram', 'battery_power', 'px_height', 'px_width', 
                'mobile_wt', 'int_memory', 'sc_h', 'talk_time']
    
    # Select only the chosen features
    labeled_subset = labeled[features].copy()
    challenge_subset = challenge[features].copy()
    
    # Standardize features (important for logistic regression)
    scaler = StandardScaler()
    labeled_subset_scaled = scaler.fit_transform(labeled_subset)
    challenge_subset_scaled = scaler.transform(challenge_subset)
    
    # Convert back to DataFrame
    labeled_df = pd.DataFrame(labeled_subset_scaled, columns=features)
    challenge_df = pd.DataFrame(challenge_subset_scaled, columns=features)
    
    return labeled_df, challenge_df

# Function to train Logistic Regression model
def logistic_regression_train(df_x, df_y):
    """
    Train a Logistic Regression model on the given data
    
    Args:
        df_x: DataFrame containing training features
        df_y: Series containing training labels
        
    Returns:
        model: Dictionary containing the trained model parameters
    """
    # Convert to numpy arrays
    X = df_x.values
    y = df_y.values
    
    # Get unique classes
    classes = np.unique(y)
    num_classes = len(classes)
    num_features = X.shape[1]
    
    # Initialize weights and biases for each class (one-vs-rest approach)
    # Using a dictionary to store the parameters
    model = {
        'weights': np.zeros((num_classes, num_features)),
        'biases': np.zeros(num_classes),
        'classes': classes,
        'learning_rate': 0.01,
        'max_iterations': 1000
    }
    
    # Train the model using gradient descent
    for class_idx, class_label in enumerate(classes):
        # Create binary target: 1 for current class, 0 for others
        binary_y = np.where(y == class_label, 1, 0)
        
        # Initialize weights and bias for this class
        weights = np.zeros(num_features)
        bias = 0
        
        # Gradient descent
        for _ in range(model['max_iterations']):
            # Linear combination
            z = np.dot(X, weights) + bias
            
            # Sigmoid function
            y_pred = 1 / (1 + np.exp(-z))
            
            # Compute gradients
            dw = np.dot(X.T, (y_pred - binary_y)) / len(binary_y)
            db = np.sum(y_pred - binary_y) / len(binary_y)
            
            # Update parameters
            weights -= model['learning_rate'] * dw
            bias -= model['learning_rate'] * db
        
        # Store weights and bias for this class
        model['weights'][class_idx] = weights
        model['biases'][class_idx] = bias
    
    return model

# Function to predict using Logistic Regression model
def logistic_regression_predict(model, test):
    """
    Predict the price range for a single test sample using Logistic Regression
    
    Args:
        model: Dictionary containing the trained model parameters
        test: Series containing a single test sample
        
    Returns:
        The predicted price range (0, 1, 2, or 3)
    """
    # Convert test to numpy array
    X = test.values.reshape(1, -1)
    
    # Initialize probabilities array
    probs = np.zeros(len(model['classes']))
    
    # Calculate probability for each class
    for class_idx, _ in enumerate(model['classes']):
        # Get weights and bias for this class
        weights = model['weights'][class_idx]
        bias = model['biases'][class_idx]
        
        # Calculate linear combination
        z = np.dot(X, weights) + bias
        
        # Apply sigmoid function
        prob = 1 / (1 + np.exp(-z))
        
        # Store probability
        probs[class_idx] = prob
    
    # Return class with highest probability
    predicted_class_idx = np.argmax(probs)
    predicted_class = model['classes'][predicted_class_idx]
    
    return int(predicted_class)
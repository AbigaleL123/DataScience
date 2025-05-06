# hw3_step1_knn.py - Improved version

# 1. Import libraries
import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import time
import os

# Start timer
start_time = time.time()

# 2. Load the training dataset
print("Loading dataset...")
try:
    train_df = pd.read_csv('mobile_labeled.csv')
    print("Dataset successfully loaded!")
    print("Dataset shape:", train_df.shape)
    print("Columns in dataset:", train_df.columns.tolist())
    print("First 3 rows:")
    print(train_df.head(3))
except Exception as e:
    print(f"Error loading dataset: {e}")
    
    # Try alternative paths
    possible_paths = [
        './mobile_labeled.csv',
        '../mobile_labeled.csv',
        'data/mobile_labeled.csv'
    ]
    
    for path in possible_paths:
        try:
            print(f"Trying alternative path: {path}")
            train_df = pd.read_csv(path)
            print(f"Success! Dataset loaded from {path}")
            print("Columns in dataset:", train_df.columns.tolist())
            break
        except Exception as e:
            print(f"Failed with path {path}: {e}")
    
    # If we still don't have train_df defined, show available files and exit
    if 'train_df' not in locals():
        print("\nFailed to load the dataset. Let's check what files are available:")
        files = os.listdir('.')
        csv_files = [f for f in files if f.endswith('.csv')]
        print(f"Found {len(csv_files)} CSV files:")
        for f in csv_files:
            print(f"- {f}")
        print("\nExiting due to file loading error.")
        exit()

# 3. Separate features and target
# Check if 'price_range' exists in the dataset
if 'price_range' in train_df.columns:
    print("'price_range' column found!")
    X = train_df.drop('price_range', axis=1)
    y = train_df['price_range']
elif 'Price_Range' in train_df.columns:  # Try alternative capitalization
    print("Found 'Price_Range' instead of 'price_range', using this column")
    X = train_df.drop('Price_Range', axis=1)
    y = train_df['Price_Range']
elif 'price range' in train_df.columns:  # Try with space
    print("Found 'price range' instead of 'price_range', using this column")
    X = train_df.drop('price range', axis=1)
    y = train_df['price range']
else:
    # If we can't find the target column, let's check what the last column is
    # (often in datasets, the target is the last column)
    last_column = train_df.columns[-1]
    print(f"'price_range' not found! Assuming the last column '{last_column}' is the target")
    
    # Let's see the unique values of this column to verify it could be the target
    if last_column in train_df.columns:
        unique_values = train_df[last_column].unique()
        print(f"Unique values in '{last_column}': {unique_values}")
        
        # Check if values match expected range (0-3)
        if set(unique_values).issubset({0, 1, 2, 3}):
            print(f"The column '{last_column}' contains values 0-3, likely the target 'price_range'")
            X = train_df.drop(last_column, axis=1)
            y = train_df[last_column]
        else:
            print("ERROR: Could not identify the target column!")
            print("Please check the CSV file format and column names.")
            print("Exiting due to missing target column.")
            exit()

# 4. List of all features
all_features = list(X.columns)
print(f"Total features: {len(all_features)}")
print(f"Features: {all_features}")

# 5. Track the best performance
best_accuracy = 0
best_features = []
best_k = 0

# 6. Use cross-validation instead of a single train-test split
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 7. Create a tracking dictionary for visualization
feature_performance = {}

# 8. Try different subsets (prioritize key features)
# First, try individual features to identify potentially strong predictors
print("\nEvaluating individual features...")
individual_scores = {}

for feature in all_features:
    X_single = X[[feature]]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_single)
    
    # Try different K values for each feature
    feature_best_k = 0
    feature_best_score = 0
    
    for k in range(1, 21):
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X_scaled, y, cv=kf, scoring='accuracy')
        avg_score = np.mean(scores)
        
        if avg_score > feature_best_score:
            feature_best_score = avg_score
            feature_best_k = k
    
    individual_scores[feature] = feature_best_score
    print(f"  {feature}: {feature_best_score:.4f} (K={feature_best_k})")

# Sort features by their individual performance
sorted_features = sorted(individual_scores.items(), key=lambda x: x[1], reverse=True)
print("\nTop individual features:")
for feature, score in sorted_features[:5]:
    print(f"  {feature}: {score:.4f}")

# 9. Now try combinations, focusing on combinations of top performers
# This is more efficient than trying all possible combinations
top_features = [f[0] for f in sorted_features[:10]]  # Focus on top 10 features
print(f"\nTop features to combine: {top_features}")

total_combinations = 0
combinations_checked = 0

# Calculate total possible combinations
for length in range(2, 9):
    total_combinations += len(list(combinations(top_features, length)))

print(f"Evaluating {total_combinations} combinations of top features...")

for length in range(2, 9):  # subsets of size 2 to 8
    for feature_subset in combinations(top_features, length):
        combinations_checked += 1
        if combinations_checked % 50 == 0:
            print(f"  Progress: {combinations_checked}/{total_combinations} combinations checked...")
        
        # Select only the chosen features
        X_subset = X[list(feature_subset)]
        
        # Standardize the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_subset)
        
        # Try different K values
        for k in range(1, 21):  # Try K = 1 to 20
            knn = KNeighborsClassifier(n_neighbors=k)
            scores = cross_val_score(knn, X_scaled, y, cv=kf, scoring='accuracy')
            avg_score = np.mean(scores)
            
            # Save the best results
            if avg_score > best_accuracy:
                best_accuracy = avg_score
                best_features = feature_subset
                best_k = k
                
                # Store for visualization
                feature_key = ", ".join(feature_subset)
                if feature_key not in feature_performance:
                    feature_performance[feature_key] = []
                feature_performance[feature_key].append((k, avg_score))

# 10. Print the best features and best K
print("\n" + "="*50)
print("Best features:", list(best_features))
print("Best K:", best_k)
print("Best accuracy:", best_accuracy)

# Try to run the best model on the challenge set
print("\nRunning predictions on challenge set...")
try:
    challenge_df = pd.read_csv('mobile_challenge.csv')
    print("Challenge dataset loaded successfully!")
    print("Challenge dataset shape:", challenge_df.shape)
    print("Challenge dataset columns:", challenge_df.columns.tolist())
except Exception as e:
    print(f"Error loading challenge dataset: {e}")
    
    # Try alternative paths
    possible_paths = [
        './mobile_challenge.csv',
        '../mobile_challenge.csv',
        'data/mobile_challenge.csv'
    ]
    
    for path in possible_paths:
        try:
            print(f"Trying alternative path: {path}")
            challenge_df = pd.read_csv(path)
            print(f"Success! Challenge dataset loaded from {path}")
            break
        except Exception as e:
            print(f"Failed with path {path}: {e}")
    
    if 'challenge_df' not in locals():
        print("WARNING: Could not load challenge dataset. Skipping predictions.")

# Continue with predictions if we loaded the challenge dataset
if 'challenge_df' in locals():
    try:
        # Make sure all best features exist in the challenge dataset
        missing_features = [f for f in best_features if f not in challenge_df.columns]
        if missing_features:
            print(f"WARNING: The following features are missing in the challenge dataset: {missing_features}")
            # Use only available features
            available_features = [f for f in best_features if f in challenge_df.columns]
            print(f"Proceeding with available features: {available_features}")
            challenge_X = challenge_df[available_features]
            X_full_subset = X[available_features]
        else:
            challenge_X = challenge_df[list(best_features)]
            X_full_subset = X[list(best_features)]
        
        # Standardize using the same scaling as the training data
        scaler = StandardScaler()
        scaler.fit(X_full_subset)  # Fit on all training data
        challenge_X_scaled = scaler.transform(challenge_X)
        
        # Train the final model on all training data
        final_knn = KNeighborsClassifier(n_neighbors=best_k)
        final_knn.fit(scaler.transform(X_full_subset), y)
        
        # Make predictions
        challenge_predictions = final_knn.predict(challenge_X_scaled)
        
        # Export predictions to CSV
        pd.DataFrame({
            'id': challenge_df.index,
            'price_range': challenge_predictions
        }).to_csv('knn_predictions.csv', index=False)
        
        print(f"Predictions saved to knn_predictions.csv")
    except Exception as e:
        print(f"Error during prediction: {e}")
        print("Could not complete predictions on challenge dataset.")

print(f"Total runtime: {time.time() - start_time:.2f} seconds")

# 11. Visualization of feature selection (optional)
try:
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(individual_scores)), list(individual_scores.values()), tick_label=list(individual_scores.keys()))
    plt.xticks(rotation=90)
    plt.title('Feature Importance (Individual Performance)')
    plt.ylabel('Accuracy')
    plt.tight_layout()
    plt.savefig('knn_feature_importance.png')
    print("Feature importance visualization saved to 'knn_feature_importance.png'")
    plt.close()

    # Plot the performance of the best feature combination across different K values
    if feature_performance:
        best_feature_key = ", ".join(best_features)
        if best_feature_key in feature_performance:
            k_values, accuracies = zip(*feature_performance[best_feature_key])
            plt.figure(figsize=(10, 6))
            plt.plot(k_values, accuracies, 'o-')
            plt.title(f'Accuracy vs K for Best Feature Set: {best_feature_key}')
            plt.xlabel('K Value')
            plt.ylabel('Accuracy')
            plt.grid(True)
            plt.savefig('knn_k_optimization.png')
            print("K optimization visualization saved to 'knn_k_optimization.png'")
            plt.close()
except Exception as e:
    print(f"Visualization error: {e}")
    print("Skipping visualizations.")

print("Done! Analysis complete.")
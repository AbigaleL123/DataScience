

import pandas as pd
import numpy as np
from scipy.stats import norm

# Load the dataset
df = pd.read_csv('NEhousing.csv')

# --- Naive Bayes Train ---
def naive_bayes_train(df_x, df_y):
    model = {}
    df = df_x.copy()
    df['price'] = df_y

    discrete_features = ['bed', 'bath', 'city']
    continuous_features = ['acre_lot', 'house_size']

    total_samples = len(df)

    for price_tier, group in df.groupby('price'):
        model[price_tier] = {}

        model[price_tier]['tier'] = len(group) / total_samples

        for feature in discrete_features:
            group[feature] = group[feature].astype(str)
            model[price_tier][feature] = group[feature].value_counts(normalize=True)


        for feature in continuous_features:
            mean = group[feature].mean()
            std = group[feature].std()
            model[price_tier][feature] = (mean, std)

    return model

# --- Naive Bayes Predict ---
def naive_bayes_predict(model, test):
    probs = {}

    for tier in model:
        prob = model[tier]['tier']

        for feature in ['bed', 'bath', 'city']:
            val = str(test[feature])
            feature_probs = model[tier][feature]
            if val in feature_probs:
                prob *= feature_probs[val]
            else:
                min_prob = feature_probs.min() if not feature_probs.empty else 1e-6
                prob *= min_prob

        for feature in ['acre_lot', 'house_size']:
            mean, std = model[tier][feature]
            x = test[feature]
            if std == 0:
                pdf = 1.0 if x == mean else 1e-6
            else:
                pdf = norm.pdf(x, mean, std)
            prob *= round(pdf, 4)

        probs[tier] = prob

    best_tier = max(probs, key=probs.get)
    best_prob = round(probs[best_tier], 4)

    return best_tier, best_prob

# --- Get Prediction Accuracy ---
def get_prediction_accuracy(df_x, df_y, ts):
    df_combined = df_x.copy()
    df_combined['price'] = df_y

    df_shuffled = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)

    test_size = int(len(df_shuffled) * ts)
    test_data = df_shuffled.iloc[:test_size]
    train_data = df_shuffled.iloc[test_size:]

    x_train = train_data[['bed', 'bath', 'acre_lot', 'city', 'house_size']]
    y_train = train_data['price']
    x_test = test_data[['bed', 'bath', 'acre_lot', 'city', 'house_size']]
    y_test = test_data['price']

    model = naive_bayes_train(x_train, y_train)

    correct = 0
    for i in range(len(x_test)):
        test_row = x_test.iloc[i]
        actual = y_test.iloc[i]
        predicted, _ = naive_bayes_predict(model, test_row)
        if predicted == actual:
            correct += 1

    accuracy = correct / len(x_test)
    return round(accuracy, 4)

# Convert price into categorical tiers
def price_to_tier(price):
    if price < 300000:
        return 'tier1'
    elif price < 600000:
        return 'tier2'
    else:
        return 'tier3'

df['price'] = df['price'].apply(price_to_tier)


df_x = df[['bed', 'bath', 'acre_lot', 'city', 'house_size']]
df_y = df['price']

accu = get_prediction_accuracy(df_x, df_y, 0.3)
print("Accuracy:", accu)  









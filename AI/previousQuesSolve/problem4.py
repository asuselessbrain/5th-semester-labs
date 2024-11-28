import pandas as pd

# Sample weather data
data = {
    'Outlook': ['sunny', 'sunny', 'overcast', 'rainy', 'rainy', 'rainy', 'overcast', 'sunny', 'sunny', 'rainy', 
                'sunny', 'overcast', 'overcast', 'rainy'],
    'Temperature': ['hot', 'hot', 'hot', 'mild', 'cool', 'cool', 'cool', 'mild', 'cool', 'mild', 
                    'mild', 'mild', 'hot', 'mild'],
    'Humidity': ['high', 'high', 'high', 'high', 'normal', 'normal', 'normal', 'high', 'normal', 'normal',
                 'normal', 'high', 'normal', 'high'],
    'Windy': ['false', 'true', 'false', 'false', 'false', 'true', 'true', 'false', 'false', 'false',
              'true', 'true', 'false', 'true'],
    'Play': ['no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'no', 'yes', 'yes', 
             'yes', 'yes', 'yes', 'no']
}

# Convert data to DataFrame
df = pd.DataFrame(data)

# Laplace Smoothing function to calculate probabilities
def calculate_probabilities_with_smoothing(df, feature_name, feature_value, target_value):
    feature_data = df[df[feature_name] == feature_value]
    target_data = feature_data[feature_data['Play'] == target_value]
    # Apply Laplace smoothing by adding 1 to the numerator and the number of possible feature values to the denominator
    feature_count = len(feature_data)
    target_count = len(target_data)
    unique_feature_values = len(df[feature_name].unique())
    return (target_count + 1) / (feature_count + unique_feature_values)  # Smoothing

# Function to calculate overall probability of 'yes' or 'no'
def calculate_overall_probability(df, target_value):
    target_data = df[df['Play'] == target_value]
    return (len(target_data) + 1) / (len(df) + 2)  # Laplace smoothing for the target class

# Naïve Bayes Classifier with smoothing
def naive_bayes_classifier_with_smoothing(new_data):
    yes_probability = calculate_overall_probability(df, 'yes')
    no_probability = calculate_overall_probability(df, 'no')
    
    for feature, value in new_data.items():
        yes_probability *= calculate_probabilities_with_smoothing(df, feature, value, 'yes')
        no_probability *= calculate_probabilities_with_smoothing(df, feature, value, 'no')
    
    return 'yes' if yes_probability > no_probability else 'no'

# Input new data to classify
new_data = {
    'Outlook': input("Enter Outlook (sunny, overcast, rainy): ").strip().lower(),
    'Temperature': input("Enter Temperature (hot, mild, cool): ").strip().lower(),
    'Humidity': input("Enter Humidity (high, normal): ").strip().lower(),
    'Windy': input("Enter Windy (true, false): ").strip().lower()
}

# Classify the new data
result = naive_bayes_classifier_with_smoothing(new_data)
print(f"Prediction: {result}")

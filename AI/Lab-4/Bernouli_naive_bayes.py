from sklearn.datasets import load_iris, load_digits, load_wine
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score

def bernoulli_naive_bayes_classifier(X, y):
    # Convert features to binary using a threshold (e.g., presence or absence of a term)
    X_binary = (X > X.mean()).astype(int)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_binary, y, test_size=0.2, random_state=42)

    # Create a Bernoulli Naive Bayes classifier
    model = BernoulliNB()

    # Train the model on the training data
    model.fit(X_train, y_train)

    # Make predictions on the testing data
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

# Load different datasets
iris = load_iris()
digits = load_digits()
wine = load_wine()

# Perform Bernoulli Naive Bayes classification on different datasets
datasets = [(iris.data, iris.target), (digits.data, digits.target), (wine.data, wine.target)]
for X, y in datasets:
    accuracy = bernoulli_naive_bayes_classifier(X, y)
    print(f"Accuracy on dataset: {accuracy:.2f}")

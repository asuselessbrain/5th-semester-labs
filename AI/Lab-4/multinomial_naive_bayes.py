from sklearn.datasets import load_iris, load_digits, load_wine
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

def multinomial_naive_bayes_classifier(dataset):
    # Load the dataset
    if dataset == "iris":
        data = load_iris()
    elif dataset == "digits":
        data = load_digits()
    elif dataset == "wine":
        data = load_wine()
    else:
        raise ValueError("Dataset not supported.")

    X = data.data
    y = data.target

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a Multinomial Naive Bayes classifier
    model = MultinomialNB()

    # Train the model on the training data
    model.fit(X_train, y_train)

    # Make predictions on the testing data
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy on {dataset} dataset: {accuracy:.2f}")

# Perform Multinomial Naive Bayes classification on different datasets
datasets = ["iris", "digits", "wine"]
for dataset in datasets:
    multinomial_naive_bayes_classifier(dataset)

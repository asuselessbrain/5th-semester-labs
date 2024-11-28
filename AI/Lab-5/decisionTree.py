import numpy as np
import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(dataset_name):
    """Load the specified dataset."""
    if dataset_name == "iris":
        data = datasets.load_iris()
        X = data.data
        y = data.target
    elif dataset_name == "breast_cancer":
        data = datasets.load_breast_cancer()
        X = data.data
        y = data.target
    else:
        raise ValueError("Unsupported dataset name. Use 'iris' or 'breast_cancer'.")

    return X, y


def train_test_split(X, y, test_size=0.2, random_state=42):
    """Simple function to split data into train and test sets."""
    np.random.seed(random_state)
    indices = np.random.permutation(len(X))
    test_set_size = int(len(X) * test_size)
    test_indices = indices[:test_set_size]
    train_indices = indices[test_set_size:]
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


class NaiveBayesClassifier:
    def __init__(self):
        self.priors = {}
        self.means = {}
        self.variances = {}

    def fit(self, X, y):
        """Fit the Naive Bayes model by calculating priors, means, and variances."""
        self.classes = np.unique(y)
        for c in self.classes:
            X_c = X[y == c]
            self.priors[c] = len(X_c) / len(y)
            self.means[c] = X_c.mean(axis=0)
            self.variances[c] = X_c.var(axis=0)

    def gaussian_density(self, class_idx, x):
        """Calculate the Gaussian density function for a feature."""
        mean = self.means[class_idx]
        variance = self.variances[class_idx]
        numerator = np.exp(-((x - mean) ** 2) / (2 * variance))
        denominator = np.sqrt(2 * np.pi * variance)
        return numerator / denominator

    def predict(self, X):
        """Predict the class for each sample in X."""
        y_pred = [self._predict_single(x) for x in X]
        return np.array(y_pred)

    def _predict_single(self, x):
        """Predict the class for a single sample."""
        posteriors = []
        for c in self.classes:
            prior = np.log(self.priors[c])
            conditional = np.sum(np.log(self.gaussian_density(c, x)))
            posterior = prior + conditional
            posteriors.append(posterior)
        return self.classes[np.argmax(posteriors)]


def evaluate_model(y_test, y_pred):
    """Evaluate model performance."""
    accuracy = np.mean(y_test == y_pred)
    print(f"Accuracy: {accuracy:.2f}")

    # Confusion Matrix
    classes = np.unique(y_test)
    cm = np.zeros((len(classes), len(classes)), dtype=int)
    for true_label, pred_label in zip(y_test, y_pred):
        cm[true_label][pred_label] += 1

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()


# Load and split the data
X, y = load_data("iris")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train and evaluate the Naive Bayes model
nb_model = NaiveBayesClassifier()
nb_model.fit(X_train, y_train)
y_pred = nb_model.predict(X_test)
evaluate_model(y_test, y_pred)

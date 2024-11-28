import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import cdist

class RBFNN:
    def __init__(self, n_clusters, gamma=1.0):
        self.n_clusters = n_clusters
        self.gamma = gamma

    def fit(self, X, y):
        # Cluster centers using K-Means
        kmeans = KMeans(n_clusters=self.n_clusters)
        kmeans.fit(X)
        self.centers = kmeans.cluster_centers_

        # Calculate width parameter
        dists = cdist(X, self.centers)
        self.sigma = np.mean(np.min(dists, axis=1))

        # Compute activations for hidden layer
        self.phi = np.exp(-self.gamma * dists ** 2)

        # Solve linear system to find output weights
        self.weights = np.linalg.lstsq(self.phi, y)[0]

    def predict(self, X):
        dists = cdist(X, self.centers)
        phi = np.exp(-self.gamma * dists ** 2)
        return np.dot(phi, self.weights)

# Load the digits dataset
digits = load_digits()
X = digits.data
y = digits.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the RBFNN
rbfnn = RBFNN(n_clusters=50)  # Number of clusters is a hyperparameter to tune
rbfnn.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rbfnn.predict(X_test)

# Convert predictions to class labels
y_pred_labels = np.argmax(y_pred, axis=1)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred_labels)
print(f'Accuracy: {accuracy:.4f}')

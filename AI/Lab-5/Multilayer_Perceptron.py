import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class MLP:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Initialize weights and biases randomly
        self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size)
        self.bias_hidden = np.random.randn(1, self.hidden_size)
        self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size)
        self.bias_output = np.random.randn(1, self.output_size)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def softmax(self, x):
        exp_values = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_values / np.sum(exp_values, axis=1, keepdims=True)

    def forward(self, X):
        # Forward pass through the network
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = self.sigmoid(self.hidden_input)
        self.output = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.output_softmax = self.softmax(self.output)
        return self.output_softmax

    def backward(self, X, y, output, learning_rate):
        # Backpropagation
        self.output_error = y - output
        self.output_delta = self.output_error
        self.hidden_error = self.output_delta.dot(self.weights_hidden_output.T)
        self.hidden_delta = self.hidden_error * self.sigmoid_derivative(self.hidden_output)

        # Update weights and biases
        self.weights_hidden_output += learning_rate * self.hidden_output.T.dot(self.output_delta)
        self.bias_output += learning_rate * np.sum(self.output_delta, axis=0, keepdims=True)
        self.weights_input_hidden += learning_rate * X.T.dot(self.hidden_delta)
        self.bias_hidden += learning_rate * np.sum(self.hidden_delta, axis=0, keepdims=True)

    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            # Forward propagation
            output = self.forward(X)

            # Backward propagation and weight update
            self.backward(X, y, output, learning_rate)


# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Convert target variable to one-hot encoded vectors
num_classes = len(np.unique(y))
y_one_hot = np.zeros((len(y), num_classes))
y_one_hot[np.arange(len(y)), y] = 1

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.2, random_state=42)

# Define and train the Multilayer Perceptron
input_size = X_train.shape[1]
hidden_size = 10
output_size = num_classes
mlp = MLP(input_size, hidden_size, output_size)
mlp.train(X_train, y_train, epochs=1000, learning_rate=0.01)

# Test the trained model
output = mlp.forward(X_test)
predicted_class = np.argmax(output, axis=1)
accuracy = np.mean(predicted_class == y_test.argmax(axis=1))
print(f'Accuracy: {accuracy:.4f}')

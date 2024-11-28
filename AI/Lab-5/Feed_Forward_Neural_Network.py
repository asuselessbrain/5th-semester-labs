import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class FeedForwardNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Initialize weights and biases using Xavier initialization
        self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size) * np.sqrt(
            2 / (self.input_size + self.hidden_size))
        self.bias_hidden = np.zeros((1, self.hidden_size))
        self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size) * np.sqrt(
            2 / (self.hidden_size + self.output_size))
        self.bias_output = np.zeros((1, self.output_size))

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)

    def forward(self, X):
        # Forward pass through the network
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = self.relu(self.hidden_input)
        self.output = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        return self.output

    def backward(self, X, y, output, learning_rate):
        # Backpropagation
        self.output_error = y - output
        self.output_delta = self.output_error
        self.hidden_error = self.output_delta.dot(self.weights_hidden_output.T)
        self.hidden_delta = self.hidden_error * self.relu_derivative(self.hidden_output)

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

            # Calculate loss
            loss = np.mean(np.square(y - output))

            # Print loss every 100 epochs
            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Loss: {loss:.4f}')


# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target.reshape(-1, 1)

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Normalize target variable to range [-1, 1]
y = (y - np.min(y)) / (np.max(y) - np.min(y)) * 2 - 1

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define and train the Feed Forward Neural Network
ffnn = FeedForwardNN(input_size=X_train.shape[1], hidden_size=20, output_size=1)
ffnn.train(X_train, y_train, epochs=1000, learning_rate=0.01)

# Test the trained model
output = ffnn.forward(X_test)
print('Output after training:')
print(output)

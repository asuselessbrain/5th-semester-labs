import tensorflow as tf
from tensorflow.keras.layers import Input, Dense


class InputModule(tf.keras.Model):
    def __init__(self, input_shape):
        super(InputModule, self).__init__()
        self.input_layer = Input(shape=input_shape)

    def call(self, inputs):
        return self.input_layer(inputs)


class HiddenModule(tf.keras.Model):
    def __init__(self, units, activation='relu'):
        super(HiddenModule, self).__init__()
        self.hidden_layer = Dense(units, activation=activation)

    def call(self, inputs):
        return self.hidden_layer(inputs)


class OutputModule(tf.keras.Model):
    def __init__(self, num_classes):
        super(OutputModule, self).__init__()
        self.output_layer = Dense(num_classes, activation='softmax')

    def call(self, inputs):
        return self.output_layer(inputs)


class ModularNeuralNetwork(tf.keras.Model):
    def __init__(self, input_shape, hidden_units, num_classes):
        super(ModularNeuralNetwork, self).__init__()
        self.input_module = InputModule(input_shape)
        self.hidden_module = HiddenModule(hidden_units)
        self.output_module = OutputModule(num_classes)

    def call(self, inputs):
        x = self.input_module(inputs)
        x = self.hidden_module(x)
        return self.output_module(x)


# Example usage:
# Define input shape, hidden units, and number of classes
input_shape = (28, 28)
hidden_units = 128
num_classes = 10

# Create an instance of ModularNeuralNetwork
model = ModularNeuralNetwork(input_shape, hidden_units, num_classes)

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Load dataset (e.g., MNIST)
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocess the data
x_train, x_test = x_train / 255.0, x_test / 255.0

# Train the model
model.fit(x_train, y_train, epochs=5)

# Evaluate the model
model.evaluate(x_test, y_test)

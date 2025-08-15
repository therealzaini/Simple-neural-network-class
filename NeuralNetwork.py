import numpy as np
import json
from ActivationFunctions import ActivationFunctions

class NeuralNetwork:
    def __init__(self, structure: list[int],
                hidden_activation: str,
                output_activation: str,
                learning_rate: float = 0.1,
                epochs: int = 150) -> None:
        """The "structure" argument is a list where the i-th element represents the number of neurons of the i-th layer of the neural network,
        the 0-th being the input layer and the last being the output layer.
        
        Example: To make a neural network with 10 input neurons, 2 hidden layers each containing 16 neurons and 7 output neurons, we would pass 
        "[10, 16, 16, 7]" to the "structure" argument.
        
        The hidden layers and output layer can be passed to different activation functions."""
        self.structure = structure
        self.number_of_layers = len(structure)
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(structure[:-1], structure[1:])]
        self.biases = [np.random.randn(y, 1) for y in structure[1:]]
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.hidden_activation_name = hidden_activation
        self.output_activation_name = output_activation
        self.hidden_activation, self.hidden_activation_prime = ActivationFunctions.ACTIVATIONS[hidden_activation]
        self.output_activation, self.output_activation_prime = ActivationFunctions.ACTIVATIONS[output_activation]
    
    def forward_propagate(self, input_vector):
        """Feeds forward the input vector, rendered into an np.array of shape (k,1).
        Returns the output of the neural network and a tuple of lists of 
        the iterated versions of the input vector after and before being passed to the activation
        function."""
        activations = [input_vector]
        pre_activations = []
        current = input_vector
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = np.dot(w, current) + b
            pre_activations.append(z)
            if i < len(self.weights) - 1:
                current = self.hidden_activation(z)
            else:
                current = self.output_activation(z)
            activations.append(current)
            
        return activations[-1], (activations, pre_activations)
    
    def back_propagate(self, input_vector, expected_output):
        """Updates the weights and biases based on one input and its expected output."""
        _, (activations, zs) = self.forward_propagate(input_vector)
        delta_w = [np.zeros(w.shape) for w in self.weights]
        delta_b = [np.zeros(b.shape) for b in self.biases]
        error = (activations[-1] - expected_output) * self.output_activation_prime(zs[-1])
        delta_b[-1] = error
        delta_w[-1] = np.dot(error, activations[-2].T)
        for i in range(len(self.weights)-2, -1, -1):
            error = np.dot(self.weights[i+1].T, error) * self.hidden_activation_prime(zs[i])
            delta_b[i] = error
            delta_w[i] = np.dot(error, activations[i].T)
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * delta_w[i]
            self.biases[i] -= self.learning_rate * delta_b[i]
    
    def get_mse_loss(self, data):
        """Calculates the loss of the neural network using the mean squared error."""
        total_loss = 0
        for input_vector, expected_output in data:
            output, _ = self.forward_propagate(input_vector)
            total_loss += np.sum((output - expected_output)**2)
        return total_loss / len(data)

    def train(self, training_data, testing_data):
        """Trains the neural network based on two different categories of data : training and testing.
        Outputs the MSE loss of each."""
        train_losses = []
        test_losses = []
        for epoch in range(self.epochs):
            for input_vector, expected_output in training_data:
                self.back_propagate(input_vector, expected_output)
            train_loss = self.get_mse_loss(training_data)
            test_loss = self.get_mse_loss(testing_data)
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            print(f"Epoch {epoch+1}/{self.epochs} | Train loss: {train_loss:.8f} | Test loss: {test_loss:.8f}")
        return train_losses, test_losses
    
    def save(self, filename):
        """Save model to JSON file"""
        data = {
            'structure': self.structure,
            'hidden_activation': self.hidden_activation_name,
            'output_activation': self.output_activation_name,
            'learning_rate': self.learning_rate,
            'epochs': self.epochs,
            'weights': [w.tolist() for w in self.weights],
            'biases': [b.tolist() for b in self.biases]
        }
        with open(filename, 'w') as f:
            json.dump(data, f)

    @classmethod
    def load(filename):
        """Load model from JSON file"""
        with open(filename, 'r') as f:
            data = json.load(f)
        nn = NeuralNetwork(
            structure=data['structure'],
            hidden_activation=data['hidden_activation'],
            output_activation=data['output_activation'],
            learning_rate=data['learning_rate'],
            epochs=data['epochs']
        )
        nn.weights = [np.array(w) for w in data['weights']]
        nn.biases = [np.array(b) for b in data['biases']]
        return nn

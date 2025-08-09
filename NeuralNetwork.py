import numpy as np
import pickle

class NeuralNetwork:
    def __init__(self, structure: list[int],
                hidden_activation,
                hidden_activation_prime,
                output_activation,
                output_activation_prime,
                learning_rate: float = 0.1,
                epochs: int = 150) -> None:
        """The "structure" argument is a list where the i-th element represents the number of neurons of the i-th layer of the neural network,
        the 0-th being the input layer and the last being the output layer.
        
        Example: To make a neural network with 10 input neurons, 2 hidden layers each containing 16 neurons and 7 output neurons, we would pass 
        "[10, 16, 16, 7]" to the "structure" argument.
        
        The hidden layers and output layer can be passed to different activation functions."""
        self.number_of_layers = len(structure)
        self.weights = []
        for i, (x, y) in enumerate(zip(structure[:-1], structure[1:])):
            if i < len(structure) - 2:
                self.weights.append(np.random.randn(y, x) * np.sqrt(2.0 / x))
            else:
                self.weights.append(np.random.randn(y, x) * 0.1)
                
        self.biases = [np.random.randn(y, 1) for y in structure[1:]]
        self.hidden_activation = hidden_activation
        self.hidden_activation_prime = hidden_activation_prime
        self.output_activation = output_activation
        self.output_activation_prime = output_activation_prime
        self.learning_rate = learning_rate
        self.epochs = epochs
    
    def forward_propagate(self, input_vector):
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
        total_loss = 0
        for input_vector, expected_output in data:
            output, _ = self.forward_propagate(input_vector)
            total_loss += np.sum((output - expected_output)**2)
        return total_loss / len(data)

    def train(self, training_data, testing_data):
        train_losses = []
        test_losses = []
        for epoch in range(self.epochs):
            for input_vector, expected_output in training_data:
                self.back_propagate(input_vector, expected_output)
            train_loss = self.get_mse_loss(training_data)
            test_loss = self.get_mse_loss(testing_data)
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            print(f"Epoch {epoch+1}/{self.epochs} | Train loss: {train_loss:.6f} | Test loss: {test_loss:.6f}")
        return train_losses, test_losses
    
    def save(self, filename):
        """Save the model to a file using pickle"""
        with open(filename, 'wb') as f:
            pickle.dump({
                'weights': self.weights,
                'biases': self.biases,
                'structure': [self.weights[0].shape[1]] + [w.shape[0] for w in self.weights],
                'learning_rate': self.learning_rate,
                'epochs': self.epochs
            }, f)
    
    @classmethod
    def load(cls, filename, hidden_activation, hidden_activation_prime, 
             output_activation, output_activation_prime):
        """Load a model from a file"""
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        net = cls(
            data['structure'],
            hidden_activation,
            hidden_activation_prime,
            output_activation,
            output_activation_prime,
            data['learning_rate'],
            data['epochs']
        )
        net.weights = data['weights']
        net.biases = data['biases']
        return net

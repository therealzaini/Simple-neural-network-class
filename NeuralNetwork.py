import numpy as np

class NeuralNetwork:
    def __init__(self, structure, hidden_activation, hidden_activation_prime, 
                 output_activation, output_activation_prime, learning_rate, epochs):
        self.number_of_layers = len(structure)
        self.weights = []
        for i, (x, y) in enumerate(zip(structure[:-1], structure[1:])):
            if i < len(structure) - 2:  # Hidden layers: He initialization for ReLU
                self.weights.append(np.random.randn(y, x) * np.sqrt(2.0 / x))
            else:  # Output layer: standard initialization for linear
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
            if i < len(self.weights) - 1:  # Hidden layers
                current = self.hidden_activation(z)
            else:  # Output layer
                current = self.output_activation(z)
            activations.append(current)
            
        return activations[-1], (activations, pre_activations)
    
    def back_propagate(self, input_vector, expected_output):
        _, (activations, zs) = self.forward_propagate(input_vector)
        delta_w = [np.zeros(w.shape) for w in self.weights]
        delta_b = [np.zeros(b.shape) for b in self.biases]
        
        # Output layer error
        error = (activations[-1] - expected_output) * self.output_activation_prime(zs[-1])
        delta_b[-1] = error
        delta_w[-1] = np.dot(error, activations[-2].T)
        
        # Backpropagate through hidden layers
        for i in range(len(self.weights)-2, -1, -1):
            error = np.dot(self.weights[i+1].T, error) * self.hidden_activation_prime(zs[i])
            delta_b[i] = error
            delta_w[i] = np.dot(error, activations[i].T)
            
        # Update weights and biases
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
import numpy as np

class ActivationFunctions:

    @staticmethod
    def sigmoid(z):
        return 1/(1+np.exp(-z))
    
    @staticmethod
    def sigmoid_prime(z):
        s = ActivationFunctions.sigmoid(z)
        return s * (1 - s)
    
    @staticmethod
    def ReLU(z):
        return np.maximum(0, z)

    @staticmethod
    def ReLU_prime(z):
        return (z > 0).astype(float)
    
    @staticmethod
    def linear(z):
        return z

    @staticmethod
    def linear_prime(z):
        return np.ones_like(z)
    
    @staticmethod
    def tanh(z):
        return np.tanh(z)
    
    @staticmethod
    def tanh_prime(z):
        return 1 - np.tanh(z)**2
    
    ACTIVATIONS = {
        'sigmoid': (sigmoid, sigmoid_prime),
        'ReLU': (ReLU, ReLU_prime),
        'linear': (linear, linear_prime),
        'tanh': (tanh, tanh_prime)
    }

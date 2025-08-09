from NeuralNetwork import NeuralNetwork
import random 
import numpy as np
import matplotlib.pyplot as plt

def generate_data(num_samples=1000):
    patterns = ['linear', 'quadratic', 'geometric']
    data = []
    max_value = 10000.0
    
    for _ in range(num_samples):
        pattern = random.choice(patterns)
        
        if pattern == 'linear':
            start = random.randint(1, 50)
            step = random.randint(1, 10)
            sequence = [start + step*i for i in range(5)]
            next_term = start + step*5
            
        elif pattern == 'quadratic':
            start = random.randint(1, 20)
            sequence = [start + i**2 for i in range(5)]
            next_term = start + 5**2
            
        elif pattern == 'geometric':
            start = random.randint(1, 5)
            ratio = random.randint(2, 4)
            sequence = [start * ratio**i for i in range(5)]
            next_term = start * ratio**5
        
        normalized_seq = [x / max_value for x in sequence]
        normalized_next = next_term / max_value
        
        input_vector = np.array(normalized_seq).reshape(5, 1)
        expected_output = np.array([[normalized_next]])
        
        data.append((input_vector, expected_output))
    
    random.shuffle(data)
    split_idx = int(0.8 * len(data))
    return data[:split_idx], data[split_idx:]

def predict_next_term(model, sequence):
    max_value = 10000.0
    scaled_seq = np.array([x/max_value for x in sequence]).reshape(5, 1)
    output, _ = model.forward_propagate(scaled_seq)
    return output.item() * max_value  # Convert to Python float

# Real ReLU and its derivative
def ReLU(z):
    return np.maximum(0, z)

def ReLU_prime(z):
    return (z > 0).astype(float)

# Linear activation and its derivative
def linear(z):
    return z

def linear_prime(z):
    return 1

train, test = generate_data(20000)

# Use ReLU for hidden layers, linear for output
nn = NeuralNetwork(
    structure=[5, 64, 64, 32, 1],
    hidden_activation=ReLU,
    hidden_activation_prime=ReLU_prime,
    output_activation=linear,
    output_activation_prime=linear_prime,
    learning_rate=0.01,
    epochs=750
)

train_losses, test_losses = nn.train(train, test)

plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.show()

# Test prediction
test_sequence = [2, 4, 6, 8, 10]
prediction = predict_next_term(nn, test_sequence)
print(f"Sequence: {test_sequence} -> Prediction: {prediction:.2f} (Expected: 12)")
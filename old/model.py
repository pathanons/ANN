# model.py
import numpy as np
import pandas as pd
from typing import List, Tuple

class MultilayerPerceptron:
    def __init__(self, 
                 layer_sizes: List[int], 
                 learning_rate: float = 0.1,
                 momentum: float = 0.9,
                 random_state: int = None):
        if random_state is not None:
            np.random.seed(random_state)
            
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.momentum = momentum
        
        # Initialize weights and biases
        self.weights = []
        self.biases = []
        self.weight_updates = []
        self.bias_updates = []
        
        for i in range(len(layer_sizes)-1):
            weight = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2.0 / (layer_sizes[i] + layer_sizes[i+1]))
            bias = np.zeros((1, layer_sizes[i+1]))
            
            self.weights.append(weight)
            self.biases.append(bias)
            self.weight_updates.append(np.zeros_like(weight))
            self.bias_updates.append(np.zeros_like(bias))
    
    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def sigmoid_derivative(self, x: np.ndarray) -> np.ndarray:
        sig = self.sigmoid(x)
        return sig * (1 - sig)
    
    def forward(self, X: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        activations = [X]
        weighted_sums = []
        
        for i in range(len(self.weights)):
            weighted_sum = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            weighted_sums.append(weighted_sum)
            activations.append(self.sigmoid(weighted_sum))
            
        return activations, weighted_sums
    
    def backward(self, X: np.ndarray, y: np.ndarray, activations: List[np.ndarray], 
                weighted_sums: List[np.ndarray]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        m = X.shape[0]
        weight_gradients = []
        bias_gradients = []
        
        delta = (activations[-1] - y) * self.sigmoid_derivative(weighted_sums[-1])
        
        for i in range(len(self.weights)-1, -1, -1):
            weight_grad = np.dot(activations[i].T, delta) / m
            bias_grad = np.sum(delta, axis=0, keepdims=True) / m
            
            weight_gradients.insert(0, weight_grad)
            bias_gradients.insert(0, bias_grad)
            
            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * self.sigmoid_derivative(weighted_sums[i-1])
        
        return weight_gradients, bias_gradients
    
    def update_parameters(self, weight_gradients: List[np.ndarray], 
                         bias_gradients: List[np.ndarray]):
        for i in range(len(self.weights)):
            self.weight_updates[i] = (self.momentum * self.weight_updates[i] - 
                                    self.learning_rate * weight_gradients[i])
            self.weights[i] += self.weight_updates[i]
            
            self.bias_updates[i] = (self.momentum * self.bias_updates[i] - 
                                  self.learning_rate * bias_gradients[i])
            self.biases[i] += self.bias_updates[i]
    
    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 1000, 
            verbose: bool = False) -> List[float]:
        losses = []
        
        for epoch in range(epochs):
            activations, weighted_sums = self.forward(X)
            
            loss = np.mean(-y * np.log(activations[-1] + 1e-15) - 
                          (1-y) * np.log(1 - activations[-1] + 1e-15))
            losses.append(loss)
            
            weight_gradients, bias_gradients = self.backward(X, y, activations, weighted_sums)
            self.update_parameters(weight_gradients, bias_gradients)
            
            if verbose and (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")
        
        return losses
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        activations, _ = self.forward(X)
        predictions = activations[-1]
        return (predictions > 0.5).astype(int)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        predictions = self.predict(X)
        return np.mean(np.all(predictions == y, axis=1))

def load_iris_data(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    data = pd.read_csv(filename)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values - 1
    
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    
    y_encoded = np.zeros((len(y), 3))
    for i, label in enumerate(y):
        y_encoded[i, int(label)] = 1
    
    return X, y_encoded

def load_pattern_data(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    data = pd.read_csv(filename)
    X = data.iloc[:, :-2].values
    y = data.iloc[:, -2:].values
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    return X, y

def load_flood_data(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    data = pd.read_csv(filename)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    y = (y - y.mean(axis=0))/y.std(axis=0)
    
    return X, y
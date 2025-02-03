import numpy as np

class MLP:
    def __init__(self, layer_sizes, learning_rate=0.1, momentum=0.9):
        """
        Initialize Multi-Layer Perceptron with Generalized Delta Rule and Momentum
        
        Parameters:
        layer_sizes (list): List of integers representing the number of neurons in each layer
        learning_rate (float): Learning rate for weight updates
        momentum (float): Momentum coefficient (between 0 and 1)
        """
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.num_layers = len(layer_sizes)
        
        # Initialize weights and biases
        self.weights = []
        self.biases = []
        
        # Initialize previous weight and bias changes for momentum
        self.prev_weight_changes = []
        self.prev_bias_changes = []
        
        for i in range(len(layer_sizes)-1):
            # Initialize weights with small random values
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.1
            self.weights.append(w)
            
            # Initialize biases with zeros
            b = np.zeros((1, layer_sizes[i+1]))
            self.biases.append(b)
            
            # Initialize previous changes with zeros
            self.prev_weight_changes.append(np.zeros_like(w))
            self.prev_bias_changes.append(np.zeros_like(b))
    
    def sigmoid(self, x):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        """Derivative of sigmoid function for backpropagation"""
        return x * (1 - x)
    
    def forward_propagation(self, X):
        """
        Forward propagation through the network
        
        Parameters:
        X (numpy.ndarray): Input data
        
        Returns:
        list: List of activations for each layer
        """
        activations = [X]
        
        # Forward propagate through each layer
        for i in range(self.num_layers - 1):
            # Calculate net input: net = Σ(weight * input) + bias
            net = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            # Apply activation function
            activation = self.sigmoid(net)
            activations.append(activation)
            
        return activations
    
    def backward_propagation(self, X, y, activations):
        """
        Implement Generalized Delta Rule for backward propagation
        
        The Generalized Delta Rule with Momentum:
        1. For output layer: δₖ = (tₖ - yₖ)f'(netₖ)
        2. For hidden layers: δⱼ = f'(netⱼ)Σ(δₖwⱼₖ)
        3. Weight update: Δw(t) = -η * ∂E/∂w + α * Δw(t-1)
           where α is the momentum coefficient
        
        Parameters:
        X (numpy.ndarray): Input data
        y (numpy.ndarray): Target values
        activations (list): List of activations from forward propagation
        
        Returns:
        tuple: Lists of weight and bias gradients
        """
        m = X.shape[0]  # number of samples
        
        # Initialize lists to store gradients
        weight_gradients = []
        bias_gradients = []
        
        # Calculate error term (delta) for output layer
        # δₖ = (tₖ - yₖ)f'(netₖ)
        output_error = activations[-1] - y
        delta = output_error * self.sigmoid_derivative(activations[-1])
        
        # Backward propagate the error
        for i in range(self.num_layers - 2, -1, -1):
            # Calculate gradients for current layer
            # ∂E/∂w = δₖ * activation_input
            weight_grad = np.dot(activations[i].T, delta) / m
            # ∂E/∂b = δₖ
            bias_grad = np.sum(delta, axis=0, keepdims=True) / m
            
            # Store gradients
            weight_gradients.insert(0, weight_grad)
            bias_gradients.insert(0, bias_grad)
            
            # Calculate delta for hidden layer (if not at input layer)
            # δⱼ = f'(netⱼ)Σ(δₖwⱼₖ)
            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * self.sigmoid_derivative(activations[i])
        
        return weight_gradients, bias_gradients
    
    def update_parameters(self, weight_gradients, bias_gradients):
        """
        Update weights and biases using computed gradients and momentum
        
        For each weight and bias:
        Δw(t) = -η * ∂E/∂w + α * Δw(t-1)
        w(t+1) = w(t) + Δw(t)
        
        where:
        η is the learning rate
        α is the momentum coefficient
        """
        for i in range(len(self.weights)):
            # Calculate weight changes with momentum
            weight_change = -self.learning_rate * weight_gradients[i] + \
                          self.momentum * self.prev_weight_changes[i]
            
            # Calculate bias changes with momentum
            bias_change = -self.learning_rate * bias_gradients[i] + \
                        self.momentum * self.prev_bias_changes[i]
            
            # Update weights and biases
            self.weights[i] += weight_change
            self.biases[i] += bias_change
            
            # Store changes for next iteration
            self.prev_weight_changes[i] = weight_change
            self.prev_bias_changes[i] = bias_change
    
    def train(self, X, y, epochs=1000):
        """
        Train the neural network using Generalized Delta Rule with Momentum
        
        Parameters:
        X (numpy.ndarray): Input data
        y (numpy.ndarray): Target values
        epochs (int): Number of training iterations
        """
        for epoch in range(epochs):
            # Forward propagation
            activations = self.forward_propagation(X)
            
            # Backward propagation using Generalized Delta Rule
            weight_gradients, bias_gradients = self.backward_propagation(X, y, activations)
            
            # Update parameters with momentum
            self.update_parameters(weight_gradients, bias_gradients)
            
            # Print error every 100 epochs
            if epoch % 100 == 0:
                error = np.mean(np.square(activations[-1] - y))
                print(f"Epoch {epoch}, Error: {error:.4f}")
    
    def predict(self, X):
        """Make predictions for input X"""
        return self.forward_propagation(X)[-1]

# Test the implementation on XOR problem
if __name__ == "__main__":
    # XOR input and output
    X = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])
    
    y = np.array([[0],
                  [1],
                  [1],
                  [0]])
    
    # Create MLP with architecture: 2 (input) -> 4 (hidden) -> 1 (output)
    mlp = MLP([2, 4, 1], learning_rate=10., momentum=0.5)
    
    # Train the network
    mlp.train(X, y, epochs=2000)
    
    # Test the network
    predictions = mlp.predict(X)
    print("\nPredictions:")
    for x, pred, target in zip(X, predictions, y):
        print(f"Input: {x}, Predicted: {pred[0]:.4f}, Target: {target[0]}")
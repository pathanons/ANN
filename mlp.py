import numpy as np
import pandas as pd
from typing import List, Tuple, Optional

class Standardizer:
    """Simple standardization class using only NumPy"""
    def __init__(self):
        self.mean = None
        self.std = None
    
    def fit(self, X: np.ndarray):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0) + 1e-8  # Add small epsilon to avoid division by zero
        
    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.std
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)

def create_kfolds(n_samples: int, n_folds: int, shuffle: bool = True, random_seed: Optional[int] = None) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Create k-fold cross-validation splits using only NumPy"""
    if random_seed is not None:
        np.random.seed(random_seed)
    
    indices = np.arange(n_samples)
    if shuffle:
        np.random.shuffle(indices)
    
    fold_sizes = np.full(n_folds, n_samples // n_folds, dtype=int)
    fold_sizes[:n_samples % n_folds] += 1
    current = 0
    folds = []
    
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        val_indices = indices[start:stop]
        train_indices = np.concatenate([indices[:start], indices[stop:]])
        folds.append((train_indices, val_indices))
        current = stop
        
    return folds

def one_hot_encode(y: np.ndarray) -> np.ndarray:
    """Convert integer labels to one-hot encoded format using only NumPy"""
    n_classes = len(np.unique(y))
    n_samples = len(y)
    one_hot = np.zeros((n_samples, n_classes))
    one_hot[np.arange(n_samples), y] = 1
    return one_hot

class MLP:
    def __init__(
        self, 
        architecture: List[int],
        learning_rate: float = 0.1,
        momentum: float = 0.9,
        random_seed: Optional[int] = None
    ):
        """
        Initialize Multi-Layer Perceptron
        
        Args:
            architecture: List of integers representing number of neurons in each layer
            learning_rate: Learning rate for gradient descent
            momentum: Momentum coefficient for gradient descent
            random_seed: Random seed for reproducibility
        """
        if random_seed is not None:
            np.random.seed(random_seed)
            
        self.architecture = architecture
        self.learning_rate = learning_rate
        self.momentum = momentum
        
        # Initialize weights and biases
        self.weights = []
        self.biases = []
        self.velocity_w = []  # For momentum
        self.velocity_b = []
        
        for i in range(len(architecture)-1):
            # Xavier/Glorot initialization
            w = np.random.randn(architecture[i], architecture[i+1]) * np.sqrt(2.0/(architecture[i] + architecture[i+1]))
            b = np.zeros((1, architecture[i+1]))
            self.weights.append(w)
            self.biases.append(b)
            self.velocity_w.append(np.zeros_like(w))
            self.velocity_b.append(np.zeros_like(b))

    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-x))#-np.clip(x, -500, 500)))
    
    def sigmoid_derivative(self, x: np.ndarray) -> np.ndarray:
        """Derivative of sigmoid function"""
        sx = self.sigmoid(x)
        return sx * (1 - sx)

    def forward(self, X: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Forward pass through the network
        
        Args:
            X: Input data of shape (n_samples, n_features)
            
        Returns:
            Tuple of:
                activations: List of layer activations
                weighted_sums: List of weighted sums before activation
        """
        current_activation = X
        activations = [X]
        weighted_sums = []

        for w, b in zip(self.weights, self.biases):
            z = np.dot(current_activation, w) + b
            weighted_sums.append(z)
            current_activation = self.sigmoid(z)
            activations.append(current_activation)

        return activations, weighted_sums

    def backward(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        activations: List[np.ndarray], 
        weighted_sums: List[np.ndarray]
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Backward pass through the network
        
        Args:
            X: Input data
            y: Target values
            activations: List of activations from forward pass
            weighted_sums: List of weighted sums from forward pass
            
        Returns:
            Tuple of weight gradients and bias gradients
        """
        m = X.shape[0]
        delta = activations[-1] - y
        
        weight_gradients = []
        bias_gradients = []

        for i in range(len(self.weights) - 1, -1, -1):
            weight_gradients.insert(0, np.dot(activations[i].T, delta) / m)
            bias_gradients.insert(0, np.sum(delta, axis=0, keepdims=True) / m)
            
            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * self.sigmoid_derivative(weighted_sums[i-1])

        return weight_gradients, bias_gradients

    def update_parameters(
        self, 
        weight_gradients: List[np.ndarray], 
        bias_gradients: List[np.ndarray]
    ):
        """Update network parameters using gradient descent with momentum"""
        for i in range(len(self.weights)):
            self.velocity_w[i] = (self.momentum * self.velocity_w[i] - 
                                self.learning_rate * weight_gradients[i])
            self.velocity_b[i] = (self.momentum * self.velocity_b[i] - 
                                self.learning_rate * bias_gradients[i])
            
            self.weights[i] += self.velocity_w[i]
            self.biases[i] += self.velocity_b[i]

    def train(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        epochs: int = 1000, 
        verbose: bool = True
    ) -> List[float]:
        """
        Train the neural network using full-batch gradient descent
        
        Args:
            X: Training data
            y: Target values
            epochs: Number of training epochs
            verbose: Whether to print progress
            
        Returns:
            List of training losses
        """
        losses = []

        for epoch in range(epochs):
            # Forward pass
            activations, weighted_sums = self.forward(X)
            
            # Backward pass
            weight_gradients, bias_gradients = self.backward(
                X, y, activations, weighted_sums
            )
            
            # Update parameters
            self.update_parameters(weight_gradients, bias_gradients)

            # Calculate loss for monitoring
            activations, _ = self.forward(X)
            loss = np.mean(np.square(activations[-1] - y))
            losses.append(loss)

            if verbose and (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.6f}")

        return losses

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions for input data"""
        activations, _ = self.forward(X)
        return activations[-1]

def create_validation_report(
    train_scores: List[float],
    val_scores: List[float],
    architecture: List[int],
    learning_rate: float,
    momentum: float,
    epochs: int,
    batch_size: int,
    filename: str = "mlp_validation_report.txt"
) -> None:
    """
    Create a detailed validation report and save it to a file
    
    Args:
        train_scores: List of training scores for each fold
        val_scores: List of validation scores for each fold
        architecture: Network architecture
        learning_rate: Learning rate used
        momentum: Momentum coefficient used
        epochs: Number of training epochs
        batch_size: Size of mini-batches
        filename: Output filename for the report
    """
    with open(filename, 'w') as f:
        # Write header
        f.write("=== Multi-Layer Perceptron Validation Report ===\n\n")
        
        # Network configuration
        f.write("Network Configuration:\n")
        f.write(f"Architecture: {architecture}\n")
        f.write(f"Learning Rate: {learning_rate}\n")
        f.write(f"Momentum: {momentum}\n")
        f.write(f"Epochs: {epochs}\n")
        f.write(f"Batch Size: {batch_size}\n\n")
        
        # Detailed fold results
        f.write("Fold-wise Results:\n")
        f.write("-----------------\n")
        for fold, (train_score, val_score) in enumerate(zip(train_scores, val_scores), 1):
            f.write(f"Fold {fold}:\n")
            f.write(f"  Training MSE: {train_score:.6f}\n")
            f.write(f"  Validation MSE: {val_score:.6f}\n")
            f.write(f"  Validation Accuracy: {1 - val_score:.6f}\n\n")
        
        # Summary statistics
        f.write("Summary Statistics:\n")
        f.write("-----------------\n")
        f.write(f"Average Training MSE: {np.mean(train_scores):.6f} ± {np.std(train_scores):.6f}\n")
        f.write(f"Average Validation MSE: {np.mean(val_scores):.6f} ± {np.std(val_scores):.6f}\n")
        f.write(f"Average Validation Accuracy: {1 - np.mean(val_scores):.6f} ± {np.std(val_scores):.6f}\n")
        
        # Additional metrics
        f.write("\nAdditional Metrics:\n")
        f.write("-----------------\n")
        f.write(f"Best Validation MSE: {np.min(val_scores):.6f} (Fold {np.argmin(val_scores) + 1})\n")
        f.write(f"Worst Validation MSE: {np.max(val_scores):.6f} (Fold {np.argmax(val_scores) + 1})\n")
        f.write(f"Validation Score Range: {np.max(val_scores) - np.min(val_scores):.6f}\n")
        
        print(f"\nValidation report has been saved to {filename}")

def train_with_cross_validation(
    X: np.ndarray, 
    y: np.ndarray,
    architecture: List[int],
    n_folds: int = 10,
    learning_rate: float = 0.1,
    momentum: float = 0.9,
    epochs: int = 1000,
    batch_size: int = 32,
    random_seed: Optional[int] = None
) -> Tuple[List[float], List[float]]:
    """
    Train MLP with k-fold cross-validation
    
    Args:
        X: Input data
        y: Target values
        architecture: Network architecture
        n_folds: Number of folds for cross-validation
        learning_rate: Learning rate
        momentum: Momentum coefficient
        epochs: Number of training epochs
        batch_size: Size of mini-batches
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of training and validation scores for each fold
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Create k-fold splits
    folds = create_kfolds(len(X), n_folds, shuffle=True, random_seed=random_seed)
    train_scores = []
    val_scores = []
    
    standardizer = Standardizer()
    
    for fold, (train_idx, val_idx) in enumerate(folds, 1):
        print(f"\nFold {fold}/{n_folds}")
        
        # Split data
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Standardize features
        X_train = standardizer.fit_transform(X_train)
        X_val = standardizer.transform(X_val)
        
        # Initialize and train model
        model = MLP(
            architecture=architecture,
            learning_rate=learning_rate,
            momentum=momentum,
            random_seed=random_seed
        )
        
        losses = model.train(X_train, y_train, epochs=epochs, 
                           batch_size=batch_size, verbose=False)
        
        # Calculate scores
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        
        train_score = np.mean(np.square(train_pred - y_train))
        val_score = np.mean(np.square(val_pred - y_val))
        
        train_scores.append(train_score)
        val_scores.append(val_score)
        
        print(f"Training MSE: {train_score:.6f}")
        print(f"Validation MSE: {val_score:.6f}")
    
    print("\nCross-validation results:")
    print(f"Average training MSE: {np.mean(train_scores):.6f} ± {np.std(train_scores):.6f}")
    print(f"Average validation MSE: {np.mean(val_scores):.6f} ± {np.std(val_scores):.6f}")
    
    # Create validation report
    create_validation_report(
        train_scores=train_scores,
        val_scores=val_scores,
        architecture=architecture,
        learning_rate=learning_rate,
        momentum=momentum,
        epochs=epochs,
        batch_size=batch_size
    )
    
    return train_scores, val_scores

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
    y = (y - y.mean(axis=0)) / y.std(axis=0)
    
    return X, y.reshape(len(y),1)

# Example usage:
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    X = np.random.randn(150, 4)  # 150 samples, 4 features
    y_int = np.random.randint(0, 3, 150)  # 3 classes
    y = one_hot_encode(y_int)  # Convert to one-hot encoding
    
    # Define network architecture [input_size, hidden_layer_sizes..., output_size]
    architecture = [4, 4, 3]  # Example for Iris dataset
    
    # Train with cross-validation
    train_scores, val_scores = train_with_cross_validation(
        X=X,
        y=y,
        architecture=architecture,
        learning_rate=0.1,
        momentum=0.9,
        epochs=1000,
        batch_size=32,
        random_seed=42
    )
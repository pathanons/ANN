import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
from mlp import MLP, Standardizer, one_hot_encode,load_iris_data,load_pattern_data,load_flood_data

def run_experiment(
    X: np.ndarray,
    y: np.ndarray,
    architecture: List[int],
    param_values: List[float],
    fixed_momentum: float = None,
    fixed_lr: float = None,
    epochs: int = 1000,
    random_seed: int = 42
) -> Tuple[List[List[float]], List[float]]:
    """
    Run experiment with different learning rates or momentum values
    
    Args:
        X: Input data
        y: Target values
        architecture: Network architecture
        param_values: List of learning rate or momentum values to test
        fixed_momentum: Fixed momentum value when testing learning rates
        fixed_lr: Fixed learning rate value when testing momentum
        epochs: Number of training epochs
        batch_size: Size of mini-batches
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (list of loss histories, list of final validation accuracies)
    """
    # Standardize data
    standardizer = Standardizer()
    X_std = standardizer.fit_transform(X)
    
    # Split data into train and validation sets (80-20 split)
    np.random.seed(random_seed)
    indices = np.random.permutation(len(X))
    split = int(0.8 * len(X))
    train_idx, val_idx = indices[:split], indices[split:]
    
    X_train, X_val = X_std[train_idx], X_std[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    loss_histories = []
    val_accuracies = []
    
    for param in param_values:
        # Initialize model with current parameters
        if fixed_momentum is not None:
            # Testing different learning rates
            model = MLP(
                architecture=architecture,
                learning_rate=param,
                momentum=fixed_momentum,
                random_seed=random_seed
            )
        else:
            # Testing different momentum values
            model = MLP(
                architecture=architecture,
                learning_rate=fixed_lr,
                momentum=param,
                random_seed=random_seed
            )
        
        # Train model and collect loss history
        losses = model.train(X_train, y_train, epochs=epochs, verbose=False)
        loss_histories.append(losses)
        
        # Calculate validation accuracy
        val_pred = model.predict(X_val)
        val_accuracy = np.mean(np.argmax(val_pred, axis=1) == np.argmax(y_val, axis=1))
        val_accuracies.append(val_accuracy)
        
    return loss_histories, val_accuracies

def plot_results(
    param_values: List[float],
    loss_histories: List[List[float]],
    val_accuracies: List[float],
    param_name: str,
    fixed_param_value: float,
    save_prefix: str
):
    """
    Plot and save the experimental results
    
    Args:
        param_values: List of parameter values tested
        loss_histories: List of loss histories for each parameter value
        val_accuracies: List of validation accuracies for each parameter value
        param_name: Name of the parameter being tested ('Learning Rate' or 'Momentum')
        fixed_param_value: Value of the fixed parameter
        save_prefix: Prefix for saving plot files
    """
    # Plot loss curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    for i, (param, losses) in enumerate(zip(param_values, loss_histories)):
        plt.plot(losses, label=f'{param_name}={param:.4f}')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training Loss vs Epochs\n(Fixed {"Momentum" if param_name == "Learning Rate" else "Learning Rate"}={fixed_param_value:.4f})')
    plt.legend()
    plt.grid(True)
    
    # Plot validation accuracies
    plt.subplot(1, 2, 2)
    plt.plot(param_values, val_accuracies, 'bo-')
    plt.xlabel(param_name)
    plt.ylabel('Validation Accuracy')
    plt.title(f'Validation Accuracy vs {param_name}\n(Fixed {"Momentum" if param_name == "Learning Rate" else "Learning Rate"}={fixed_param_value:.4f})')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_{param_name.lower().replace(" ", "_")}.png')
    plt.close()
    
    # Save results to text file
    with open(f'{save_prefix}_{param_name.lower().replace(" ", "_")}_results.txt', 'w') as f:
        f.write(f'=== {param_name} Experiment Results ===\n')
        f.write(f'Fixed {"Momentum" if param_name == "Learning Rate" else "Learning Rate"}: {fixed_param_value:.4f}\n\n')
        
        f.write(f'{param_name} Values and Corresponding Validation Accuracies:\n')
        f.write('-' * 50 + '\n')
        for param, acc in zip(param_values, val_accuracies):
            f.write(f'{param_name}: {param:.4f}, Validation Accuracy: {acc:.4f}\n')
        
        f.write(f'\nBest {param_name}: {param_values[np.argmax(val_accuracies)]:.4f}')
        f.write(f'\nBest Validation Accuracy: {np.max(val_accuracies):.4f}')

if __name__ == "__main__":
    # config_name = 'cross'
    # config_name = 'ellipse'
    config_name = 'flood'
    IRIS_CONFIG = {
        'path':'./csv/iris.csv',
        'architecture' : [4, 8, 3],
        'lr_values' : [0.001, 0.01, 0.1, 0.5, 1.0],
        'fixed_momentum' : 0.0,
        'momentum_values' : [0.0, 0.5, 0.8, 0.9, 0.95, 0.99],
        'fixed_lr' : 0.01,
        'epoch':2000,
    }

    ELLIPSE_CONFIG = {
        'path':'./csv/ellipse.csv',
        'architecture' : [2, 8, 2],
        'lr_values' : [0.001, 0.01, 0.1, 0.5, 1.0],
        'fixed_momentum' : 0.5,
        'momentum_values' : [0.0, 0.5, 0.8, 0.9, 0.95, 0.99],
        'fixed_lr' : 1.0,
        'epoch':2000,
    }

    CROSS_CONFIG = {
        'path':'./csv/cross.csv',
        'architecture' : [2, 8, 2],
        'lr_values' : [0.001, 0.01, 0.1, 0.5, 1.0],
        'fixed_momentum' : 0.5,
        'momentum_values' : [0.0, 0.1, 0.3, 0.5, 0.8, 0.99],
        'fixed_lr' : 1.0,
        'epoch':2000,
    }

    FLOOD_CONFIG = {
        'path':'./csv/flood.csv',
        'architecture' : [8, 4, 2, 1],
        'lr_values' : [0.0001, 0.001, 0.01, 0.1],
        'fixed_momentum' : 0.1,
        'momentum_values' : [0.0, 0.5, 0.8, 0.9, 0.95, 0.99],
        'fixed_lr' : 0.001,
        'epoch':2000,
    }

    if config_name == 'iris':
        config = IRIS_CONFIG
        X,y = load_iris_data(config['path'])
    
    elif config_name == 'flood':
        config = FLOOD_CONFIG
        X,y = load_flood_data(config['path'])

    elif config_name == 'ellipse':
        config = ELLIPSE_CONFIG
        X,y = load_pattern_data(config['path'])

    else:
        config = CROSS_CONFIG
        X,y = load_pattern_data(config['path'])
    # Generate sample data (replace with your actual data)
    
    # Network architecture
    architecture = config['architecture']
    
    # Experiment 1: Different learning rates, fixed momentum
    lr_values = config['lr_values']
    fixed_momentum = config['fixed_momentum']
    momentum_values = config['momentum_values']
    fixed_lr = config['fixed_lr']
    
    loss_histories_lr, val_accuracies_lr = run_experiment(
        X=X,
        y=y,
        architecture=architecture,
        param_values=lr_values,
        fixed_momentum=fixed_momentum,
        epochs=config['epoch']
    )
    
    plot_results(
        param_values=lr_values,
        loss_histories=loss_histories_lr,
        val_accuracies=val_accuracies_lr,
        param_name='Learning Rate',
        fixed_param_value=fixed_momentum,
        save_prefix='experiment'
    )
    
    # Experiment 2: Different momentum values, fixed learning rate
    
    loss_histories_m, val_accuracies_m = run_experiment(
        X=X,
        y=y,
        architecture=architecture,
        param_values=momentum_values,
        fixed_lr=fixed_lr,
        epochs=config['epoch']
    )
    
    plot_results(
        param_values=momentum_values,
        loss_histories=loss_histories_m,
        val_accuracies=val_accuracies_m,
        param_name='Momentum',
        fixed_param_value=fixed_lr,
        save_prefix='experiment'
    )
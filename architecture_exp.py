import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
from mlp import MLP, Standardizer, one_hot_encode, create_kfolds,load_iris_data,load_pattern_data,load_flood_data
import argparse

def create_parser():
    # สร้าง parser object
    parser = argparse.ArgumentParser(
        description='โปรแกรมรับข้อความจาก command line'
    )
    
    # เพิ่ม argument สำหรับรับ string
    parser.add_argument(
        '-exp',
        type=str,
        required=True,
        help='ข้อความที่ต้องการรับจาก user'
    )
    
    return parser

def create_architectures_varying_nodes(
    input_size: int,
    output_size: int,
    hidden_nodes: List[int],
    n_layers: int = 2
) -> List[List[int]]:
    """Create list of architectures with varying number of nodes in hidden layers"""
    architectures = []
    for n in hidden_nodes:
        arch = [input_size] + [n] * n_layers + [output_size]
        architectures.append(arch)
    return architectures

def create_architectures_varying_layers(
    input_size: int,
    output_size: int,
    n_layers: List[int],
    nodes_per_layer: int = 4
) -> List[List[int]]:
    """Create list of architectures with varying number of layers"""
    architectures = []
    for n in n_layers:
        arch = [input_size] + [nodes_per_layer] * n + [output_size]
        architectures.append(arch)
    return architectures

def run_architecture_experiment(
    X: np.ndarray,
    y: np.ndarray,
    architectures: List[List[int]],
    n_folds: int = 5,
    learning_rate: float = 0.1,
    momentum: float = 0.9,
    epochs: int = 1000,
    random_seed: int = 42
) -> Tuple[List[List[float]], List[List[float]], List[float], List[float]]:
    """
    Run experiment with different network architectures using cross-validation
    
    Returns:
        Tuple of (loss histories, validation histories, mean validities, std validities)
    """
    np.random.seed(random_seed)
    
    # Standardize data
    standardizer = Standardizer()
    X_std = standardizer.fit_transform(X)
    
    all_loss_histories = []
    all_val_histories = []
    mean_validities = []
    std_validities = []
    
    for arch in architectures:
        # K-fold cross validation
        folds = create_kfolds(len(X), n_folds, shuffle=True, random_seed=random_seed)
        fold_losses = []
        fold_vals = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            X_train, X_val = X_std[train_idx], X_std[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            model = MLP(
                architecture=arch,
                learning_rate=learning_rate,
                momentum=momentum,
                random_seed=random_seed
            )
            
            # Train and collect history
            losses = []
            val_accuracies = []
            
            for epoch in range(epochs):
                # Train one epoch
                loss = model.train(X_train, y_train, epochs=1, verbose=False)[0]
                losses.append(loss)
                
                # Calculate validation accuracy
                val_pred = model.predict(X_val)
                val_acc = np.mean(np.argmax(val_pred, axis=1) == np.argmax(y_val, axis=1))
                val_accuracies.append(val_acc)
            
            fold_losses.append(losses)
            fold_vals.append(val_accuracies)
        
        # Average histories across folds
        mean_loss_history = np.mean(fold_losses, axis=0)
        mean_val_history = np.mean(fold_vals, axis=0)
        
        all_loss_histories.append(mean_loss_history)
        all_val_histories.append(mean_val_history)
        
        # Calculate final validation statistics
        final_vals = [vals[-1] for vals in fold_vals]
        mean_validities.append(np.mean(final_vals))
        std_validities.append(np.std(final_vals))
    
    return all_loss_histories, all_val_histories, mean_validities, std_validities

def plot_architecture_results(
    architectures: List[List[int]],
    loss_histories: List[List[float]],
    val_histories: List[List[float]],
    mean_validities: List[float],
    std_validities: List[float],
    experiment_type: str,
    save_prefix: str
):
    """Plot and save the experimental results"""
    plt.figure(figsize=(15, 5))
    
    # Plot loss curves
    plt.subplot(1, 3, 1)
    for i, (arch, losses) in enumerate(zip(architectures, loss_histories)):
        if experiment_type == 'nodes':
            label = f'{arch[1]} nodes/layer'
        else:
            label = f'{len(arch)-2} hidden layers'
        plt.plot(losses, label=label)
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss vs Epochs')
    plt.legend()
    plt.grid(True)
    
    # Plot validation accuracy curves
    plt.subplot(1, 3, 2)
    for i, (arch, vals) in enumerate(zip(architectures, val_histories)):
        if experiment_type == 'nodes':
            label = f'{arch[1]} nodes/layer'
        else:
            label = f'{len(arch)-2} hidden layers'
        plt.plot(vals, label=label)
    
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy')
    plt.title('Validation Accuracy vs Epochs')
    plt.legend()
    plt.grid(True)
    
    # Plot final validation accuracies
    plt.subplot(1, 3, 3)
    x = range(len(architectures))
    if experiment_type == 'nodes':
        plt.errorbar(x, mean_validities, yerr=std_validities, fmt='bo-')
        plt.xticks(x, [arch[1] for arch in architectures])
        plt.xlabel('Nodes per Hidden Layer')
    else:
        plt.errorbar(x, mean_validities, yerr=std_validities, fmt='bo-')
        plt.xticks(x, [len(arch)-2 for arch in architectures])
        plt.xlabel('Number of Hidden Layers')
    
    plt.ylabel('Final Validation Accuracy')
    plt.title('Final Validation Accuracy vs Architecture')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_{experiment_type}.png')
    plt.close()

def save_architecture_results(
    architectures: List[List[int]],
    mean_validities: List[float],
    std_validities: List[float],
    experiment_type: str,
    save_prefix: str,
    extra_params: Dict
):
    """Save detailed results to a text file"""
    with open(f'{save_prefix}_{experiment_type}_results.txt', 'w') as f:
        f.write(f'=== Network Architecture Experiment Results: {experiment_type.title()} ===\n\n')
        
        # Write experiment parameters
        f.write('Experiment Parameters:\n')
        f.write('-' * 30 + '\n')
        for param, value in extra_params.items():
            f.write(f'{param}: {value}\n')
        f.write('\n')
        
        # Write detailed results
        f.write('Detailed Results:\n')
        f.write('-' * 30 + '\n')
        for i, (arch, mean_val, std_val) in enumerate(zip(architectures, mean_validities, std_validities)):
            f.write(f'\nArchitecture {i+1}:\n')
            f.write(f'  Layer sizes: {arch}\n')
            if experiment_type == 'nodes':
                f.write(f'  Nodes per hidden layer: {arch[1]}\n')
            else:
                f.write(f'  Number of hidden layers: {len(arch)-2}\n')
            f.write(f'  Final validation accuracy: {mean_val:.4f} ± {std_val:.4f}\n')
        
        # Write summary
        f.write('\nSummary:\n')
        f.write('-' * 30 + '\n')
        best_idx = np.argmax(mean_validities)
        if experiment_type == 'nodes':
            f.write(f'Best number of nodes: {architectures[best_idx][1]}\n')
        else:
            f.write(f'Best number of layers: {len(architectures[best_idx])-2}\n')
        f.write(f'Best validation accuracy: {mean_validities[best_idx]:.4f} ± {std_validities[best_idx]:.4f}\n')


def main():
    # Generate sample data (replace with your actual data)
    parser = create_parser()
    args = parser.parse_args()
    config_name = args.exp
    print(config_name)

    np.random.seed(42)
    IRIS_CONFIG = {
        'path':'./csv/iris.csv',
        'lr':0.01,
        'momentum':0.0,
        'epoch':2000,
        'hidden_nodes':[4,8,16,32],
        'n_layers':[1,2,3,4,5]
    }

    ELLIPSE_CONFIG = {
        'path':'./csv/ellipse.csv',
        'lr':0.1,
        'momentum':0.9,
        'epoch':2000,
        'hidden_nodes':[4,8,16,32],
        'n_layers':[1,2,3,4,5]
    }

    CROSS_CONFIG = {
        'path':'./csv/cross.csv',
        'lr':1.0,
        'momentum':0.5,
        'epoch':2000,
        'hidden_nodes':[4,8,16,32],
        'n_layers':[1,2,3,4,5]
    }

    FLOOD_CONFIG = {
        'path':'./csv/flood.csv',
        'lr':0.01,
        'momentum':0.5,
        'epoch':200,
        'hidden_nodes':[2,4,8,16,32],
        'n_layers':[1,2,3,4]
    }

    XOR_CONFIG = {
        'lr':10.0,
        'momentum':0.5,
        'epoch':1000,
        'hidden_nodes':[2,4,8],
        'n_layers':[1,2,3,4]
    }

    EXP_CONFIG = {
        'range':[1,10],
        'step':0.05,
        'lr':0.01,
        'momentum':0.9,
        'epoch':200,
        'hidden_nodes':[2,3,4],
        'n_layers':[1,2,3]
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
    
    elif config_name == 'exp':
        config = EXP_CONFIG
        X = np.arange(config['range'][0],config['range'][1]+config['step'],config['step'])
        y = np.exp(-X)
        X = X.reshape(len(X),1)
        y = y.reshape(len(y),1)
    
    elif config_name == 'xor':
        config = XOR_CONFIG
        X = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])
    
        y = np.array([[0],
                    [1],
                    [1],
                    [0]])

    else:
        config = CROSS_CONFIG
        X,y = load_pattern_data(config['path'])
    
    # Common parameters
    input_size = X.shape[1]
    output_size = y.shape[1]
    learning_rate = config['lr']
    momentum = config['momentum']
    epochs = config['epoch']
    n_folds = 10
    
    # Experiment 1: Varying number of nodes
    hidden_nodes = config['hidden_nodes']
    node_architectures = create_architectures_varying_nodes(
        input_size=input_size,
        output_size=output_size,
        hidden_nodes=hidden_nodes
    )
    
    loss_histories, val_histories, mean_vals, std_vals = run_architecture_experiment(
        X=X,
        y=y,
        architectures=node_architectures,
        n_folds=n_folds,
        learning_rate=learning_rate,
        momentum=momentum,
        epochs=epochs,
    )
    
    plot_architecture_results(
        architectures=node_architectures,
        loss_histories=loss_histories,
        val_histories=val_histories,
        mean_validities=mean_vals,
        std_validities=std_vals,
        experiment_type='nodes',
        save_prefix='experiment'
    )
    
    save_architecture_results(
        architectures=node_architectures,
        mean_validities=mean_vals,
        std_validities=std_vals,
        experiment_type='nodes',
        save_prefix='experiment',
        extra_params={
            'Learning Rate': learning_rate,
            'Momentum': momentum,
            'Epochs': epochs,
            'Number of Folds': n_folds
        }
    )
    
    # Experiment 2: Varying number of layers
    n_layers = config['n_layers']
    layer_architectures = create_architectures_varying_layers(
        input_size=input_size,
        output_size=output_size,
        n_layers=n_layers
    )
    
    loss_histories, val_histories, mean_vals, std_vals = run_architecture_experiment(
        X=X,
        y=y,
        architectures=layer_architectures,
        n_folds=n_folds,
        learning_rate=learning_rate,
        momentum=momentum,
        epochs=epochs,
    )
    
    plot_architecture_results(
        architectures=layer_architectures,
        loss_histories=loss_histories,
        val_histories=val_histories,
        mean_validities=mean_vals,
        std_validities=std_vals,
        experiment_type='layers',
        save_prefix='experiment'
    )
    
    save_architecture_results(
        architectures=layer_architectures,
        mean_validities=mean_vals,
        std_validities=std_vals,
        experiment_type='layers',
        save_prefix='experiment',
        extra_params={
            'Learning Rate': learning_rate,
            'Momentum': momentum,
            'Epochs': epochs,
            'Number of Folds': n_folds
        }
    )

if __name__ == '__main__':
    # main('cross')
    # main('ellipse')
    main()
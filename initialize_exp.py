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

def run_seed_experiment(
    X: np.ndarray,
    y: np.ndarray,
    architecture: List[int],
    seeds: List[int],
    n_folds: int = 5,
    learning_rate: float = 0.1,
    momentum: float = 0.9,
    epochs: int = 1000
) -> Tuple[List[List[float]], List[List[float]], List[float], List[float]]:
    """
    Run experiment with different initialization seeds using cross-validation
    
    Returns:
        Tuple of (loss histories, validation histories, mean validities, std validities)
    """
    # Standardize data once for all experiments
    standardizer = Standardizer()
    X_std = standardizer.fit_transform(X)
    
    all_loss_histories = []
    all_val_histories = []
    mean_validities = []
    std_validities = []
    
    for seed in seeds:
        # Set seed for fold creation
        np.random.seed(seed)
        
        # K-fold cross validation
        folds = create_kfolds(len(X), n_folds, shuffle=True, random_seed=seed)
        fold_losses = []
        fold_vals = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            X_train, X_val = X_std[train_idx], X_std[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            model = MLP(
                architecture=architecture,
                learning_rate=learning_rate,
                momentum=momentum,
                random_seed=seed
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

def plot_seed_results(
    seeds: List[int],
    loss_histories: List[List[float]],
    val_histories: List[List[float]],
    mean_validities: List[float],
    std_validities: List[float],
    save_prefix: str
):
    """Plot and save the experimental results"""
    plt.figure(figsize=(15, 5))
    
    # Plot loss curves
    plt.subplot(1, 3, 1)
    for seed, losses in zip(seeds, loss_histories):
        plt.plot(losses, label=f'Seed {seed}')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss vs Epochs')
    plt.legend()
    plt.grid(True)
    
    # Plot validation accuracy curves
    plt.subplot(1, 3, 2)
    for seed, vals in zip(seeds, val_histories):
        plt.plot(vals, label=f'Seed {seed}')
    
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy')
    plt.title('Validation Accuracy vs Epochs')
    plt.legend()
    plt.grid(True)
    
    # Plot final validation accuracies with error bars
    plt.subplot(1, 3, 3)
    x = range(len(seeds))
    plt.errorbar(x, mean_validities, yerr=std_validities, fmt='bo-')
    plt.xticks(x, seeds, rotation=45)
    plt.xlabel('Initialization Seed')
    plt.ylabel('Final Validation Accuracy')
    plt.title('Final Validation Accuracy vs Seed')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_seeds.png')
    plt.close()

def analyze_seed_stability(
    mean_validities: List[float],
    std_validities: List[float]
) -> Dict:
    """Analyze the stability of results across different seeds"""
    overall_mean = np.mean(mean_validities)
    overall_std = np.std(mean_validities)
    
    best_idx = np.argmax(mean_validities)
    worst_idx = np.argmin(mean_validities)
    
    max_diff = np.max(mean_validities) - np.min(mean_validities)
    coefficient_of_variation = overall_std / overall_mean * 100
    
    return {
        'overall_mean': overall_mean,
        'overall_std': overall_std,
        'best_accuracy': mean_validities[best_idx],
        'best_accuracy_std': std_validities[best_idx],
        'worst_accuracy': mean_validities[worst_idx],
        'worst_accuracy_std': std_validities[worst_idx],
        'max_difference': max_diff,
        'coefficient_of_variation': coefficient_of_variation
    }

def save_seed_results(
    seeds: List[int],
    mean_validities: List[float],
    std_validities: List[float],
    stability_metrics: Dict,
    save_prefix: str,
    extra_params: Dict
):
    """Save detailed results to a text file"""
    with open(f'{save_prefix}_seed_results.txt', 'w') as f:
        f.write('=== Initialization Seed Experiment Results ===\n\n')
        
        # Write experiment parameters
        f.write('Experiment Parameters:\n')
        f.write('-' * 30 + '\n')
        for param, value in extra_params.items():
            f.write(f'{param}: {value}\n')
        f.write('\n')
        
        # Write detailed results for each seed
        f.write('Detailed Results by Seed:\n')
        f.write('-' * 30 + '\n')
        for seed, mean_val, std_val in zip(seeds, mean_validities, std_validities):
            f.write(f'\nSeed {seed}:\n')
            f.write(f'  Validation accuracy: {mean_val:.4f} ± {std_val:.4f}\n')
        
        # Write stability analysis
        f.write('\nStability Analysis:\n')
        f.write('-' * 30 + '\n')
        f.write(f'Overall mean accuracy: {stability_metrics["overall_mean"]:.4f}\n')
        f.write(f'Overall standard deviation: {stability_metrics["overall_std"]:.4f}\n')
        f.write(f'Best accuracy: {stability_metrics["best_accuracy"]:.4f} ± {stability_metrics["best_accuracy_std"]:.4f}\n')
        f.write(f'Worst accuracy: {stability_metrics["worst_accuracy"]:.4f} ± {stability_metrics["worst_accuracy_std"]:.4f}\n')
        f.write(f'Maximum accuracy difference: {stability_metrics["max_difference"]:.4f}\n')
        f.write(f'Coefficient of variation: {stability_metrics["coefficient_of_variation"]:.2f}%\n')
        
        # Write interpretation
        f.write('\nInterpretation:\n')
        f.write('-' * 30 + '\n')
        cv = stability_metrics["coefficient_of_variation"]
        if cv < 1:
            stability = "very stable"
        elif cv < 5:
            stability = "reasonably stable"
        elif cv < 10:
            stability = "moderately stable"
        else:
            stability = "relatively unstable"
            
        f.write(f'The model performance is {stability} across different initialization seeds.\n')
        f.write(f'The accuracy varies by up to {stability_metrics["max_difference"]*100:.2f}% points depending on the seed.\n')

if __name__ == "__main__":
    # config_name = 'cross'
    parser = create_parser()
    args = parser.parse_args()
    config_name = args.exp
    print(config_name)

    IRIS_CONFIG = {
        'path':'./csv/iris.csv',
        'architecture' : [4, 8, 3],
        'seeds' : [42, 123, 456, 789, 101112],  # Different initialization seeds to test
        'lr' : 0.01,
        'momentum' : 0.0,
        'epochs' : 1000,
        
    }

    ELLIPSE_CONFIG = {
        'path':'./csv/ellipse.csv',
        'architecture' : [2, 8, 8, 2],
        'seeds' : [42, 123, 456, 789, 101112],  # Different initialization seeds to test
        'lr' : 0.5,
        'momentum' : 0.3,
        'epochs' : 2000,
    }

    CROSS_CONFIG = {
        'path':'./csv/cross.csv',
        'architecture' : [2, 32, 2],
        'seeds' : [42, 123, 456, 789, 101112],  # Different initialization seeds to test
        'lr' : 1.0,
        'momentum' : 0.5,
        'epochs' : 1000,
    }

    FLOOD_CONFIG = {
        'path':'./csv/flood.csv',
        'architecture' : [8, 4, 2, 1],
        'seeds' : [42, 123, 456, 789, 101112],  # Different initialization seeds to test
        'lr' : 0.01,
        'momentum' : 0.5,
        'epochs' : 200,
    }

    XOR_CONFIG = {
        'lr':10.0,
        'momentum':0.5,
        'epochs':50,
        'architecture' : [2, 5, 1],
        'seeds' : [42, 123, 456, 789, 101112],  # Different initialization seeds to test
    }

    EXP_CONFIG = {
        'range':[1,10],
        'step':0.05,
        'lr':0.01,
        'momentum':0.9,
        'epochs':200,
        'architecture' : [1, 2, 1],
        'seeds' : [42, 123, 456, 789, 101112],  # Different initialization seeds to test
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
    # Generate sample data (replace with your actual data)
    # Experiment parameters
    architecture = config['architecture']  # Example for Iris dataset
    seeds = config['seeds'] # Different initialization seeds to test
    learning_rate = config['lr']
    momentum = config['momentum']
    epochs = config['epochs']
    n_folds = 10
    
    # Run experiment
    loss_histories, val_histories, mean_vals, std_vals = run_seed_experiment(
        X=X,
        y=y,
        architecture=architecture,
        seeds=seeds,
        n_folds=n_folds,
        learning_rate=learning_rate,
        momentum=momentum,
        epochs=epochs
    )
    
    # Plot results
    plot_seed_results(
        seeds=seeds,
        loss_histories=loss_histories,
        val_histories=val_histories,
        mean_validities=mean_vals,
        std_validities=std_vals,
        save_prefix='experiment'
    )
    
    # Analyze stability
    stability_metrics = analyze_seed_stability(mean_vals, std_vals)
    
    # Save results
    save_seed_results(
        seeds=seeds,
        mean_validities=mean_vals,
        std_validities=std_vals,
        stability_metrics=stability_metrics,
        save_prefix='experiment',
        extra_params={
            'Architecture': architecture,
            'Learning Rate': learning_rate,
            'Momentum': momentum,
            'Epochs': epochs,
            'Number of Folds': n_folds
        }
    )
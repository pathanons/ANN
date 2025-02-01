# experiments.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import time
from datetime import datetime
from model import MultilayerPerceptron
import os

DATASET_CONFIG = {
    'iris': {
        'max_epochs': 200,
        'node_architectures': [  # Test different nodes with 1 hidden layer
            [4, 4, 3],     # minimal
            [4, 8, 3],     # small
            [4, 16, 3],     # medium
        ],
        'layer_architectures': [  # Test different layers with moderate nodes
            [4, 8, 3],            # Shallow: 16+4+12+3 = 35
            [4, 3, 3, 3],         # Medium: 12+3+9+3+9+3 = 39
            [4, 3, 2, 2, 3],      # Deep: 12+3+6+2+4+2+6+3 = 38
        ],
        'architecture':[4,8,3],
        'lr':0.1,
        'momentum':0.0
    },
    'cross': {
        'max_epochs': 1000,
        'node_architectures': [  # Test different nodes with 1 hidden layer
            [2, 4, 2],    # small
            [2, 8, 2],    # large
            [2, 16, 2],   # very large
        ],
        'layer_architectures': [  # Test different layers with moderate nodes
            [2, 4, 2],            # Shallow: 22 parameters (8 + 4 + 8 + 2)
            [2, 4, 4, 2],         # Medium: 23 parameters (6 + 3 + 6 + 2 + 4 + 2)
            [2, 4, 4, 4, 2],      # Deep: 24 parameters (4 + 2 + 4 + 2 + 4 + 2 + 4 + 2)
        ],
        'architecture':[2,8,2],
        'lr':1.0,
        'momentum':0.3
    },
    'ellipse': {
        'max_epochs': 2000,
        'node_architectures': [  # Test different nodes with 1 hidden layer
            [2, 4, 2],    # small
            [2, 8, 2],    # large
            [2, 16, 2],   # very large
        ],
        'layer_architectures': [  # Test different layers with moderate nodes
            [2, 4, 2],            # Shallow: 22 parameters (8 + 4 + 8 + 2)
            [2, 4, 4, 2],         # Medium: 23 parameters (6 + 3 + 6 + 2 + 4 + 2)
            [2, 4, 4, 4, 2],      # Deep: 24 parameters (4 + 2 + 4 + 2 + 4 + 2 + 4 + 2)
        ],
        'architecture':[2,4,2],
        'lr':1.0,
        'momentum':0.3
    }
}

class ExperimentTracker:
    def __init__(self, model, X_train, y_train, X_test, y_test):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.train_losses = []
        self.test_losses = []
        self.train_accuracies = []
        self.test_accuracies = []
        self.convergence_epoch = None
        self.total_time = None
        
    def compute_loss(self, X, y):
        activations, _ = self.model.forward(X)
        return np.mean(-y * np.log(activations[-1] + 1e-15) - 
                      (1-y) * np.log(1 - activations[-1] + 1e-15))
    
    def track_epoch(self, epoch):
        train_loss = self.compute_loss(self.X_train, self.y_train)
        test_loss = self.compute_loss(self.X_test, self.y_test)
        
        train_acc = self.model.evaluate(self.X_train, self.y_train)
        test_acc = self.model.evaluate(self.X_test, self.y_test)
        
        self.train_losses.append(train_loss)
        self.test_losses.append(test_loss)
        self.train_accuracies.append(train_acc)
        self.test_accuracies.append(test_acc)
        
        # Check for convergence
        if len(self.train_losses) > 1:
            if abs(self.train_losses[-1] - self.train_losses[-2]) < 1e-6:
                if self.convergence_epoch is None:
                    self.convergence_epoch = epoch

def experiment_architectures(X, y, dataset_name: str):
    """Run experiments with different network architectures"""
    config = DATASET_CONFIG[dataset_name]
    max_epochs = config['max_epochs']
    
    # Split data
    indices = np.random.permutation(len(X))
    split = int(0.9 * len(X))
    train_idx, test_idx = indices[:split], indices[split:]
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # Results for nodes experiment
    node_results = []
    for arch in config['node_architectures']:
        start_time = time.time()
        model = MultilayerPerceptron(layer_sizes=arch)
        tracker = ExperimentTracker(model, X_train, y_train, X_test, y_test)
        
        for epoch in range(max_epochs):
            model.fit(X_train, y_train, epochs=1, verbose=False)
            tracker.track_epoch(epoch)
            
            # if tracker.convergence_epoch is not None:
            #     break
        
        end_time = time.time()
        tracker.total_time = end_time - start_time
        
        node_results.append({
            'Architecture': str(arch),
            'Hidden Nodes': arch[1],  # Number of nodes in hidden layer
            'Final Train Accuracy': tracker.train_accuracies[-1],
            'Final Test Accuracy': tracker.test_accuracies[-1],
            'Convergence Epoch': tracker.convergence_epoch or max_epochs,
            'Training Time': tracker.total_time,
            'Train Losses': tracker.train_losses,
            'Test Losses': tracker.test_losses,
            'Train Accuracies': tracker.train_accuracies,
            'Test Accuracies': tracker.test_accuracies
        })
    
    # Results for layers experiment
    layer_results = []
    for arch in config['layer_architectures']:
        start_time = time.time()
        model = MultilayerPerceptron(layer_sizes=arch)
        tracker = ExperimentTracker(model, X_train, y_train, X_test, y_test)
        
        for epoch in range(max_epochs):
            model.fit(X_train, y_train, epochs=1, verbose=False)
            tracker.track_epoch(epoch)

        
        end_time = time.time()
        tracker.total_time = end_time - start_time
        
        layer_results.append({
            'Architecture': str(arch),
            'Hidden Layers': len(arch) - 2,  # Number of hidden layers
            'Final Train Accuracy': tracker.train_accuracies[-1],
            'Final Test Accuracy': tracker.test_accuracies[-1],
            'Convergence Epoch': tracker.convergence_epoch or max_epochs,
            'Training Time': tracker.total_time,
            'Train Losses': tracker.train_losses,
            'Test Losses': tracker.test_losses,
            'Train Accuracies': tracker.train_accuracies,
            'Test Accuracies': tracker.test_accuracies
        })
    
    # Plot results
    plot_architecture_results(dataset_name, node_results, layer_results)
    
    # Save results to CSV
    pd.DataFrame(node_results).to_csv(f'results/{dataset_name}/node_architecture_results.csv', index=False)
    pd.DataFrame(layer_results).to_csv(f'results/{dataset_name}/layer_architecture_results.csv', index=False)
    
    return {'nodes': node_results, 'layers': layer_results}
def plot_architecture_results(dataset_name: str, node_results: List[Dict], layer_results: List[Dict]):
    """Generate plots comparing different architecture configurations"""
    # Create figure for node comparison
    plt.figure(figsize=(15, 10))
    
    # Plot node results
    plt.subplot(2, 2, 1)
    nodes = [res['Hidden Nodes'] for res in node_results]
    train_acc = [res['Final Train Accuracy'] for res in node_results]
    test_acc = [res['Final Test Accuracy'] for res in node_results]
    plt.plot(nodes, train_acc, 'o-', label='Train Accuracy')
    plt.plot(nodes, test_acc, 'o-', label='Test Accuracy')
    plt.xlabel('Number of Nodes in Hidden Layer')
    plt.ylabel('Accuracy')
    plt.title('Effect of Hidden Layer Size')
    plt.legend()
    plt.grid(True)
    
    # Plot convergence epochs for nodes
    plt.subplot(2, 2, 2)
    conv_epochs = [res['Convergence Epoch'] for res in node_results]
    plt.plot(nodes, conv_epochs, 'o-')
    plt.xlabel('Number of Nodes in Hidden Layer')
    plt.ylabel('Convergence Epoch')
    plt.title('Convergence Speed vs Hidden Layer Size')
    plt.grid(True)
    
    # Plot layer results
    plt.subplot(2, 2, 3)
    layers = [res['Hidden Layers'] for res in layer_results]
    train_acc = [res['Final Train Accuracy'] for res in layer_results]
    test_acc = [res['Final Test Accuracy'] for res in layer_results]
    plt.plot(layers, train_acc, 'o-', label='Train Accuracy')
    plt.plot(layers, test_acc, 'o-', label='Test Accuracy')
    plt.xlabel('Number of Hidden Layers')
    plt.ylabel('Accuracy')
    plt.title('Effect of Network Depth')
    plt.legend()
    plt.grid(True)
    
    # Plot convergence epochs for layers
    plt.subplot(2, 2, 4)
    conv_epochs = [res['Convergence Epoch'] for res in layer_results]
    plt.plot(layers, conv_epochs, 'o-')
    plt.xlabel('Number of Hidden Layers')
    plt.ylabel('Convergence Epoch')
    plt.title('Convergence Speed vs Network Depth')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'results/{dataset_name}/architecture_analysis.png')
    plt.close()
    
    # Plot learning curves for each configuration
    # Nodes
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    for res in node_results:
        plt.plot(res['Train Losses'], label=f'{res["Hidden Nodes"]} nodes')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss vs Nodes')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    for res in node_results:
        plt.plot(res['Test Accuracies'], label=f'{res["Hidden Nodes"]} nodes')
    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy')
    plt.title('Test Accuracy vs Nodes')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'results/{dataset_name}/node_learning_curves.png')
    plt.close()
    
    # Layers
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    for res in layer_results:
        plt.plot(res['Train Losses'], label=f'{res["Hidden Layers"]} layers')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss vs Layers')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    for res in layer_results:
        plt.plot(res['Test Accuracies'], label=f'{res["Hidden Layers"]} layers')
    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy')
    plt.title('Test Accuracy vs Layers')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'results/{dataset_name}/layer_learning_curves.png')
    plt.close()

def experiment_hyperparameters(X, y, dataset_name: str):
    """Run experiments with different learning rates and momentum values"""
    config = DATASET_CONFIG[dataset_name]
    max_epochs = config['max_epochs']
    architecture = config['architecture']
    
    learning_rates = [0.01, 0.1, 0.5, 1.0, 10.0]
    momentums = [0.0, 0.3, 0.5, 0.9]
    
    # For fixing one parameter and varying the other
    default_lr = config['lr']    # middle value
    default_momentum = config['momentum']  # middle value
    
    results = []
    lr_results = []  # For learning rate comparison
    momentum_results = []  # For momentum comparison
    
    indices = np.random.permutation(len(X))
    split = int(0.9 * len(X))
    train_idx, test_idx = indices[:split], indices[split:]
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # Test all combinations for complete results
    for lr in learning_rates:
        for mom in momentums:
            model = MultilayerPerceptron(
                layer_sizes=architecture,
                learning_rate=lr,
                momentum=mom
            )
            
            tracker = ExperimentTracker(model, X_train, y_train, X_test, y_test)
            
            start_time = time.time()
            for epoch in range(max_epochs):
                model.fit(X_train, y_train, epochs=1, verbose=False)
                tracker.track_epoch(epoch)
                    
            end_time = time.time()
            tracker.total_time = end_time - start_time
            
            results.append({
                'Learning Rate': lr,
                'Momentum': mom,
                'Final Train Accuracy': tracker.train_accuracies[-1],
                'Final Test Accuracy': tracker.test_accuracies[-1],
                'Convergence Epoch': tracker.convergence_epoch or max_epochs,
                'Training Time': tracker.total_time,
                'Train Losses': tracker.train_losses,
                'Test Losses': tracker.test_losses
            })
            
            # Store results for fixed parameter comparisons
            if mom == default_momentum:
                lr_results.append({
                    'Learning Rate': lr,
                    'Train Losses': tracker.train_losses,
                    'Test Losses': tracker.test_losses,
                    'Convergence Epoch': tracker.convergence_epoch or max_epochs
                })
            
            if lr == default_lr:
                momentum_results.append({
                    'Momentum': mom,
                    'Train Losses': tracker.train_losses,
                    'Test Losses': tracker.test_losses,
                    'Convergence Epoch': tracker.convergence_epoch or max_epochs
                })
    
    # Plot learning curves for different learning rates (fixed momentum)
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    for res in lr_results:
        plt.plot(res['Train Losses'], 
                label=f'lr={res["Learning Rate"]} (mom={default_momentum})')
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title(f'Learning Rate Effect (Momentum={default_momentum})')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    for res in lr_results:
        plt.plot(res['Test Losses'], 
                label=f'lr={res["Learning Rate"]} (mom={default_momentum})')
    plt.xlabel('Epoch')
    plt.ylabel('Test Loss')
    plt.title(f'Learning Rate Effect on Test Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'results/{dataset_name}/learning_rate_effects.png')
    plt.close()
    
    # Plot learning curves for different momentum values (fixed learning rate)
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    for res in momentum_results:
        plt.plot(res['Train Losses'], 
                label=f'mom={res["Momentum"]} (lr={default_lr})')
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title(f'Momentum Effect (Learning Rate={default_lr})')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    for res in momentum_results:
        plt.plot(res['Test Losses'], 
                label=f'mom={res["Momentum"]} (lr={default_lr})')
    plt.xlabel('Epoch')
    plt.ylabel('Test Loss')
    plt.title(f'Momentum Effect on Test Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'results/{dataset_name}/momentum_effects.png')
    plt.close()
    
    # Add convergence comparison plot
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    lr_epochs = [res['Convergence Epoch'] for res in lr_results]
    plt.plot(learning_rates, lr_epochs, 'o-')
    plt.xlabel('Learning Rate')
    plt.ylabel('Convergence Epoch')
    plt.title(f'Convergence Speed vs Learning Rate (Momentum={default_momentum})')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    mom_epochs = [res['Convergence Epoch'] for res in momentum_results]
    plt.plot(momentums, mom_epochs, 'o-')
    plt.xlabel('Momentum')
    plt.ylabel('Convergence Epoch')
    plt.title(f'Convergence Speed vs Momentum (Learning Rate={default_lr})')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'results/{dataset_name}/hyperparameter_convergence.png')
    plt.close()
    
    # Save results to CSV
    pd.DataFrame(results).to_csv(f'results/{dataset_name}/hyperparameter_results.csv', index=False)
    
    return results

def experiment_weight_initialization(X, y, dataset_name: str):
    """Run experiments with different weight initializations"""
    config = DATASET_CONFIG[dataset_name]
    max_epochs = config['max_epochs']
    architecture = config['architecture']  # Use same architecture as hyperparameter experiment
    seeds = [42, 123, 456, 789, 101112]
    
    results = []
    
    indices = np.random.permutation(len(X))
    split = int(0.9 * len(X))
    train_idx, test_idx = indices[:split], indices[split:]
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    for seed in seeds:
        model = MultilayerPerceptron(
            layer_sizes=architecture,
            random_state=seed
        )
        
        tracker = ExperimentTracker(model, X_train, y_train, X_test, y_test)
        
        start_time = time.time()
        for epoch in range(max_epochs):
            model.fit(X_train, y_train, epochs=1, verbose=False)
            tracker.track_epoch(epoch)
            
                
        end_time = time.time()
        tracker.total_time = end_time - start_time
        
        results.append({
            'Random Seed': seed,
            'Final Train Accuracy': tracker.train_accuracies[-1],
            'Final Test Accuracy': tracker.test_accuracies[-1],
            'Convergence Epoch': tracker.convergence_epoch or max_epochs,
            'Training Time': tracker.total_time,
            'Train Losses': tracker.train_losses,
            'Test Losses': tracker.test_losses,
            'Train Accuracies': tracker.train_accuracies,
            'Test Accuracies': tracker.test_accuracies
        })
    
    # Plot learning curves comparing different initializations
    plt.figure(figsize=(15, 10))
    
    # Training Loss
    plt.subplot(2, 2, 1)
    for res in results:
        plt.plot(res['Train Losses'], label=f'Seed {res["Random Seed"]}')
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Training Loss for Different Initializations')
    plt.legend()
    plt.grid(True)
    
    # Test Loss
    plt.subplot(2, 2, 2)
    for res in results:
        plt.plot(res['Test Losses'], label=f'Seed {res["Random Seed"]}')
    plt.xlabel('Epoch')
    plt.ylabel('Test Loss')
    plt.title('Test Loss for Different Initializations')
    plt.legend()
    plt.grid(True)
    
    # Training Accuracy
    plt.subplot(2, 2, 3)
    for res in results:
        plt.plot(res['Train Accuracies'], label=f'Seed {res["Random Seed"]}')
    plt.xlabel('Epoch')
    plt.ylabel('Training Accuracy')
    plt.title('Training Accuracy for Different Initializations')
    plt.legend()
    plt.grid(True)
    
    # Test Accuracy
    plt.subplot(2, 2, 4)
    for res in results:
        plt.plot(res['Test Accuracies'], label=f'Seed {res["Random Seed"]}')
    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy')
    plt.title('Test Accuracy for Different Initializations')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'results/{dataset_name}/initialization_learning_curves.png')
    plt.close()
    
    # Create summary statistics plot
    plt.figure(figsize=(15, 5))
    
    # Final accuracy comparison
    plt.subplot(1, 3, 1)
    seeds = [res['Random Seed'] for res in results]
    train_acc = [res['Final Train Accuracy'] for res in results]
    test_acc = [res['Final Test Accuracy'] for res in results]
    width = 0.35
    x = np.arange(len(seeds))
    plt.bar(x - width/2, train_acc, width, label='Train Accuracy')
    plt.bar(x + width/2, test_acc, width, label='Test Accuracy')
    plt.xlabel('Random Seed')
    plt.ylabel('Final Accuracy')
    plt.title('Final Accuracy Comparison')
    plt.xticks(x, seeds, rotation=45)
    plt.legend()
    plt.grid(True)
    
    # Convergence speed comparison
    plt.subplot(1, 3, 2)
    conv_epochs = [res['Convergence Epoch'] for res in results]
    plt.bar(seeds, conv_epochs)
    plt.xlabel('Random Seed')
    plt.ylabel('Convergence Epoch')
    plt.title('Convergence Speed Comparison')
    plt.xticks(rotation=45)
    plt.grid(True)
    
    # Box plot of final accuracies
    plt.subplot(1, 3, 3)
    plt.boxplot([train_acc, test_acc], labels=['Train', 'Test'])
    plt.ylabel('Accuracy')
    plt.title('Accuracy Distribution')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'results/{dataset_name}/initialization_summary.png')
    plt.close()
    
    # Calculate and display statistical measures
    stats_df = pd.DataFrame({
        'Train Accuracy': train_acc,
        'Test Accuracy': test_acc,
        'Convergence Epoch': conv_epochs
    })
    
    stats_summary = stats_df.agg(['mean', 'std', 'min', 'max'])
    stats_summary.to_csv(f'results/{dataset_name}/initialization_stats.csv')
    
    # Save detailed results
    pd.DataFrame(results).to_csv(f'results/{dataset_name}/initialization_results.csv', index=False)
    
    return results

def setup_experiment_directories(dataset_names):
    """Create necessary directories for storing results"""
    if not os.path.exists('results'):
        os.makedirs('results')
    
    for dataset in dataset_names:
        dataset_dir = f'results/{dataset}'
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)

def generate_summary_report(dataset_name: str, results: Dict):
    """Generate comprehensive summary report of all experiments"""
    report = []
    
    # 1. Architecture Experiments Summary
    report.append("=" * 80)
    report.append("ARCHITECTURE EXPERIMENTS SUMMARY")
    report.append("=" * 80)
    
    # Node experiment results
    node_df = pd.DataFrame(results['architecture']['nodes'])
    best_node = node_df.loc[node_df['Final Test Accuracy'].idxmax()]
    
    report.append("\nA. Hidden Node Analysis (Single Hidden Layer)")
    report.append("-" * 40)
    report.append(f"• Best number of nodes: {best_node['Hidden Nodes']}")
    report.append(f"• Best test accuracy: {best_node['Final Test Accuracy']:.4f}")
    report.append(f"• Training time: {best_node['Training Time']:.2f} seconds")
    report.append(f"• Convergence epoch: {best_node['Convergence Epoch']}")
    report.append("\nEffect of increasing nodes:")
    report.append(f"• Accuracy range: {node_df['Final Test Accuracy'].min():.4f} - {node_df['Final Test Accuracy'].max():.4f}")
    report.append(f"• Average convergence epoch: {node_df['Convergence Epoch'].mean():.1f}")
    
    # Layer experiment results
    layer_df = pd.DataFrame(results['architecture']['layers'])
    best_layer = layer_df.loc[layer_df['Final Test Accuracy'].idxmax()]
    
    report.append("\nB. Network Depth Analysis")
    report.append("-" * 40)
    report.append(f"• Optimal number of hidden layers: {best_layer['Hidden Layers']}")
    report.append(f"• Best test accuracy: {best_layer['Final Test Accuracy']:.4f}")
    report.append(f"• Training time: {best_layer['Training Time']:.2f} seconds")
    report.append(f"• Convergence epoch: {best_layer['Convergence Epoch']}")
    report.append("\nEffect of increasing depth:")
    report.append(f"• Accuracy range: {layer_df['Final Test Accuracy'].min():.4f} - {layer_df['Final Test Accuracy'].max():.4f}")
    report.append(f"• Average convergence epoch: {layer_df['Convergence Epoch'].mean():.1f}")
    
    # 2. Hyperparameter Experiments Summary
    report.append("\n" + "=" * 80)
    report.append("HYPERPARAMETER EXPERIMENTS SUMMARY")
    report.append("=" * 80)
    
    hyper_df = pd.DataFrame(results['hyperparameter'])
    best_hyper = hyper_df.loc[hyper_df['Final Test Accuracy'].idxmax()]
    
    report.append("\nA. Learning Rate Analysis")
    report.append("-" * 40)
    # Group by learning rate with fixed momentum
    default_momentum = 0.5
    lr_analysis = hyper_df[hyper_df['Momentum'] == default_momentum]
    report.append(f"Analysis with fixed momentum = {default_momentum}:")
    for lr in lr_analysis['Learning Rate'].unique():
        lr_data = lr_analysis[lr_analysis['Learning Rate'] == lr]
        report.append(f"• Learning Rate {lr}:")
        report.append(f"  - Test Accuracy: {lr_data['Final Test Accuracy'].values[0]:.4f}")
        report.append(f"  - Convergence Epoch: {lr_data['Convergence Epoch'].values[0]}")
    
    report.append("\nB. Momentum Analysis")
    report.append("-" * 40)
    # Group by momentum with fixed learning rate
    default_lr = 0.1
    mom_analysis = hyper_df[hyper_df['Learning Rate'] == default_lr]
    report.append(f"Analysis with fixed learning rate = {default_lr}:")
    for mom in mom_analysis['Momentum'].unique():
        mom_data = mom_analysis[mom_analysis['Momentum'] == mom]
        report.append(f"• Momentum {mom}:")
        report.append(f"  - Test Accuracy: {mom_data['Final Test Accuracy'].values[0]:.4f}")
        report.append(f"  - Convergence Epoch: {mom_data['Convergence Epoch'].values[0]}")
    
    report.append("\nC. Best Overall Configuration")
    report.append("-" * 40)
    report.append(f"• Learning Rate: {best_hyper['Learning Rate']}")
    report.append(f"• Momentum: {best_hyper['Momentum']}")
    report.append(f"• Test Accuracy: {best_hyper['Final Test Accuracy']:.4f}")
    report.append(f"• Convergence Epoch: {best_hyper['Convergence Epoch']}")
    report.append(f"• Training Time: {best_hyper['Training Time']:.2f} seconds")
    
    # 3. Weight Initialization Experiments Summary
    report.append("\n" + "=" * 80)
    report.append("WEIGHT INITIALIZATION EXPERIMENTS SUMMARY")
    report.append("=" * 80)
    
    init_df = pd.DataFrame(results['initialization'])
    best_init = init_df.loc[init_df['Final Test Accuracy'].idxmax()]
    
    report.append("\nA. Statistical Analysis")
    report.append("-" * 40)
    report.append("Test Accuracy Statistics:")
    report.append(f"• Mean: {init_df['Final Test Accuracy'].mean():.4f}")
    report.append(f"• Standard Deviation: {init_df['Final Test Accuracy'].std():.4f}")
    report.append(f"• Range: {init_df['Final Test Accuracy'].min():.4f} - {init_df['Final Test Accuracy'].max():.4f}")
    
    report.append("\nB. Convergence Analysis")
    report.append("-" * 40)
    report.append("Convergence Speed Statistics:")
    report.append(f"• Mean Epoch: {init_df['Convergence Epoch'].mean():.1f}")
    report.append(f"• Standard Deviation: {init_df['Convergence Epoch'].std():.1f}")
    report.append(f"• Range: {init_df['Convergence Epoch'].min():.1f} - {init_df['Convergence Epoch'].max():.1f}")
    
    report.append("\nC. Best Initialization")
    report.append("-" * 40)
    report.append(f"• Random Seed: {best_init['Random Seed']}")
    report.append(f"• Test Accuracy: {best_init['Final Test Accuracy']:.4f}")
    report.append(f"• Convergence Epoch: {best_init['Convergence Epoch']}")
    report.append(f"• Training Time: {best_init['Training Time']:.2f} seconds")
    
    # Overall Summary
    report.append("\n" + "=" * 80)
    report.append("OVERALL BEST CONFIGURATION")
    report.append("=" * 80)
    report.append(f"• Architecture: {best_layer['Architecture']}")
    report.append(f"• Learning Rate: {best_hyper['Learning Rate']}")
    report.append(f"• Momentum: {best_hyper['Momentum']}")
    report.append(f"• Random Seed: {best_init['Random Seed']}")
    report.append(f"• Best Test Accuracy: {max(best_layer['Final Test Accuracy'], best_hyper['Final Test Accuracy'], best_init['Final Test Accuracy']):.4f}")
    
    # Save report
    with open(f'results/{dataset_name}/summary_report.txt', 'w') as f:
        f.write('\n'.join(report))
    
    return '\n'.join(report)

def run_experiments(datasets: Dict[str, Tuple[str, callable]]):
    """
    Run all experiments on specified datasets
    
    Args:
        datasets: Dictionary mapping dataset names to (filename, loader_function) tuples
    
    Returns:
        Dict containing results for all experiments
    """
    # Create directories for results
    setup_experiment_directories(datasets.keys())
    
    # Store all results
    all_results = {}
    
    for name, (filename, loader_func) in datasets.items():
        print(f"\n{'='*50}")
        print(f"Running experiments on {name.upper()} dataset")
        print(f"{'='*50}")
        
        try:
            # Load and preprocess data
            print(f"\nLoading {name} dataset from {filename}...")
            X, y = loader_func(filename)
            print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
            
            # 1. Architecture Experiments
            print("\n1. Running Architecture Experiments...")
            print("   - Testing different numbers of nodes in single hidden layer")
            print("   - Testing different numbers of hidden layers")
            arch_results = experiment_architectures(X, y, name)
            print("✓ Architecture experiments completed")
            
            # 2. Hyperparameter Experiments
            print("\n2. Running Hyperparameter Experiments...")
            print("   - Testing different learning rates")
            print("   - Testing different momentum values")
            hyper_results = experiment_hyperparameters(X, y, name)
            print("✓ Hyperparameter experiments completed")
            
            # 3. Weight Initialization Experiments
            print("\n3. Running Weight Initialization Experiments...")
            print("   - Testing different random seeds")
            init_results = experiment_weight_initialization(X, y, name)
            print("✓ Weight initialization experiments completed")
            
            # Store results
            all_results[name] = {
                'architecture': {
                    'nodes': arch_results['nodes'],
                    'layers': arch_results['layers']
                },
                'hyperparameter': hyper_results,
                'initialization': init_results
            }
            
            # Generate summary report
            print("\nGenerating summary report...")
            report = generate_summary_report(name, all_results[name])
            print("✓ Summary report generated")
            
            # Print best configurations
            print(f"\nBest Configurations for {name.upper()}:")
            print("-" * 40)
            
            # Get best architecture
            best_node = max(arch_results['nodes'], 
                          key=lambda x: x['Final Test Accuracy'])
            best_layer = max(arch_results['layers'], 
                           key=lambda x: x['Final Test Accuracy'])
            best_arch = max([best_node, best_layer], 
                          key=lambda x: x['Final Test Accuracy'])
            
            # Get best hyperparameters
            best_hyper = max(hyper_results, 
                           key=lambda x: x['Final Test Accuracy'])
            
            # Get best initialization
            best_init = max(init_results, 
                          key=lambda x: x['Final Test Accuracy'])
            
            print(f"• Best Architecture: {best_arch['Architecture']}")
            print(f"  - Test Accuracy: {best_arch['Final Test Accuracy']:.4f}")
            print(f"  - Convergence Epoch: {best_arch['Convergence Epoch']}")
            
            print(f"\n• Best Hyperparameters:")
            print(f"  - Learning Rate: {best_hyper['Learning Rate']}")
            print(f"  - Momentum: {best_hyper['Momentum']}")
            print(f"  - Test Accuracy: {best_hyper['Final Test Accuracy']:.4f}")
            
            print(f"\n• Best Initialization:")
            print(f"  - Random Seed: {best_init['Random Seed']}")
            print(f"  - Test Accuracy: {best_init['Final Test Accuracy']:.4f}")
            
            print(f"\nAll results saved in: results/{name}/")
            
        except Exception as e:
            print(f"\nError processing {name} dataset:")
            print(f"Error message: {str(e)}")
            continue
    
    return all_results
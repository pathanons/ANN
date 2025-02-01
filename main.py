# main.py
from model import MultilayerPerceptron, load_iris_data, load_pattern_data
from experiments import run_experiments

def main():
    # Define datasets and their loaders
    datasets = {
        'iris': ('iris.csv', load_iris_data),
        'cross': ('cross.csv', load_pattern_data),
        'ellipse': ('ellipse.csv', load_pattern_data)
    }
    
    # Run all experiments
    results = run_experiments(datasets)
    print("\nAll experiments completed successfully!")

if __name__ == "__main__":
    main()
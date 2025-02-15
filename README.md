# CoEvolution-GA

## Project Overview

This project explores the co-evolution of neural networks using a genetic algorithm. The key objective is to train neural networks to classify data effectively while evolving their structure, activation functions, and optimization parameters over generations.

## Objectives

- **Neural Network Evolution**: Implement a genetic algorithm to evolve different neural network architectures and hyperparameters.
- **Dataset Classification**: Use the `glass.data` dataset to test the effectiveness of evolved neural networks in classification tasks.
- **Optimization through Evolution**: Improve the network’s learning ability by evolving activation functions, optimizers, and regularization parameters.
- **Performance Evaluation**: Assess network performance using a **confusion matrix** and **F1-score** to measure classification accuracy.
- **Mutation & Crossover in Neural Networks**: Apply genetic operators (mutation, crossover) to evolve weights and hyperparameters efficiently.
- **Scalability & Efficiency**: Explore methods like parallelization and weight caching to optimize training time.

## Repository Structure

```
CoEvolution-GA/
│-- main.py                 # Main script coordinating the evolutionary process
│-- Genetic.py              # Genetic algorithm implementation
│-- Selection_methods.py    # Various selection strategies (e.g., tournament, roulette)
│-- algorithms.py           # Contains different neural network training methods
│-- create_problem_sets.py  # Loads and processes dataset (glass.data)
│-- fitness_functions.py    # Evaluates networks using confusion matrix & F1-score
│-- function_selection.py   # Manages function selection during evolution
│-- mutations.py            # Implements mutation strategies for evolving networks
│-- neural_network.py       # Defines and trains neural networks (MLPClassifier)
│-- settings.py             # Configuration parameters
│-- setup.py                # Environment setup script
│-- inputs/                 # Contains dataset files
│-- outputs/                # Stores trained network results
│   ├── distribution/       # Stores weight distributions & analysis
```

## How to Run the Project

### Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/AbednAboH/CoEvolution-GA.git
   cd CoEvolution-GA
   ```
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Evolutionary Process

Execute the script:
```bash
python main.py
```

## Example Outputs

```
Final Best Neural Network:
- Hidden Layers: (64, 32)
- Activation Function: ReLU
- Optimizer: Adam
- F1 Score: 0.89
```

---


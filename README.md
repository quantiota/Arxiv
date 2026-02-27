# Structured Knowledge Accumulation (SKA) – ArXiv Repository

## Repository Overview
This repository contains the code and preprint article for the Structured Knowledge Accumulation (SKA) framework, a novel forward-only learning paradigm that eliminates backpropagation and optimizes knowledge accumulation through entropy minimization.

## Getting Started

### Prerequisites
The code requires the following dependencies:
- Python 3.9 or higher
- PyTorch
- NumPy
- Pandas
- Matplotlib

You can install the required packages using:
```bash
pip install torch numpy pandas matplotlib
```

### Directory Structure
```
/project/Arxiv/
├── ska/
│   ├── __init__.py
│   ├── entropy.py
│   ├── model.py
│   ├── utils.py
│   └── visualization.py
│   └── wrappers.py
├── data/
│   └── mnist_subset_100_per_class.pt
├── scripts/
│   └── train.py
├── saved/          # For saving metrics and results
└── figures/        # For saving visualizations
    ├── cosine/
    ├── entropy/
    ├── knowledge/
    ├── lagrangian/
    ├── output/
    └── tensor_net/
```

### Dataset
The repository uses a preprocessed MNIST subset with 100 samples per class. The data file should be placed in the `data` directory as:
```
/project/Arxiv/data/mnist_subset_100_per_class.pt
```



## Running the Code

1. Make sure all directories are set up correctly:
```bash
mkdir -p data figures/entropy figures/cosine figures/knowledge figures/lagrangian figures/weight figures/tensor_net saved
```

2. Place the MNIST data file in the `data` directory.

3. Run the training script:
```bash
python scripts/train.py
```

The script will:
- Load the MNIST subset data
- Initialize the SKA model
- Run forward passes through the network
- Update weights using the SKA algorithm without backpropagation
- Generate visualizations of entropy, alignment, and other metrics
- Save results in the `saved` directory
- Create visualization plots in the `figures` directory

## Understanding the Results

After running the script, you'll find:

1. **CSV files** in the `saved` directory:
   - entropy_history.csv
   - cosine_history.csv
   - frobenius_history.csv
   - weight_frobenius_history.csv
   - tensor_net_history.csv
   - output_neuron_activation.csv

2. **Visualization plots** in the `figures` directory, organized by metric type.

## Customizing the Model

You can modify the SKA model parameters in `ska/model.py`:
- Layer sizes
- Learning rate
- Number of forward steps (K)
- Activation functions

## Issues & Discussions
- Issues: If you encounter problems, report them in the Issues tab.
- Discussions: For theoretical insights, implementation questions, or general feedback, join the Discussions tab.


### Important Note

The layered SKA Neural Network presented here is a **discrete approximation** (a "shadow") of the underlying continuous [Riemannian Neural Field (RNF)](https://github.com/quantiota/Riemannian-Neural-Fields).  

It is provided **for educational purposes only** to illustrate the core mechanism of local entropy reduction through decision shifts $\Delta_D$.

The true SKA dynamics and all its deeper properties live in the continuous RNF.  
The layered discretization is useful for teaching and rapid experimentation, but it is not the complete theory.

## Contact
For inquiries, reach out at info@quantiota.org.
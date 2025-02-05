# Layer-wise dynamic event-triggered neural network control

This repository contains the code to reproduce the experimental results in ["Layer-Wise Dynamic Event-Triggered Neural Network Control For Discrete-Time Nonlinear Systems"](https://ut3-toulouseinp.hal.science/hal-04870932/document).

## How to Run the Code

### Prerequisites

Before running the code, ensure you have the following packages installed:

- Python 3.12.4
- NumPy
- Torch
- Cvxpy
- SciPy
- Matplotlib

You can install the required packages using pip:

```bash
pip install numpy scipy matplotlib torch cvxpy
```

### Solver

This code uses the MOSEK solver for optimization. You need to install MOSEK and obtain a license to run it.

You can install MOSEK using pip:

```bash
pip install mosek
```

A MOSEK license is required, visit the [MOSEK website](https://www.mosek.com/products/academic-licenses/)


## Contents

### Folders

- **weights**: Contains the weights and biases obtained from the training process.
- **results**: Includes the results of the LMI computations.
- **auxiliary_code**: Provides useful code for plotting and other auxiliary functions.

### Scripts

- `LMI.py`: Executes the LMI problem.
- `config.py`: Contains parameters to decide which LMI has to be solved.
- `system.py`: Contains the system definition.
- `test_results.py`: Plots the results shown in the paper.

# Layer-wise dynamic event-triggered neural network control

This repository contains the code to reproduce the experimental results in ["Layer-Wise Dynamic Event-Triggered Neural Network Control For Discrete-Time Nonlinear Systems"](https://ut3-toulouseinp.hal.science/hal-04870932/document).

## How to Run the Code


This project uses [`uv`](https://github.com/astral-sh/uv) for dependency management. Follow the steps below to set up the environment.

## Setup Instructions

### 1. Install `uv`
If you don’t have `uv` installed, install it with:
```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```
### 2. Install dependencies
Run:
```sh
uv sync
```
### 3. Activate the Virtual Environment
If you need to manually activate it:
```sh
source .venv/bin/activate
```

## Mosek License Requirement
This project requires a valid Mosek license.  

- If you already have a license, ensure it is accessible via the `MOSEKLM_LICENSE_FILE` environment variable or placed in the default Mosek license path.  
    - If you need a license, you can obtain a free academic license from [Mosek’s website](https://www.mosek.com/products/academic-licenses/).


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

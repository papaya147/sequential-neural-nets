
# Sequential Neural Networks

A lightweight implementation of sequential neural networks from scratch in Python. This project is designed for educational purposes, allowing users to understand the inner workings of neural networks for regression and classification tasks.

## Overview

This repository provides a simple, modular implementation of sequential dense neural networks built from the ground up. It supports common machine learning tasks such as regression and classification, with example use cases in the `notebooks/` directory.

## Features

- **Layers**: Dense (fully connected) layers.
- **Optimizers**: Stochastic Gradient Descent (SGD).
- **Activation Functions**:
  - Linear
  - ReLU
  - Sigmoid
  - Softmax
- **Loss Functions**:
  - Mean Absolute Error (MAE)
  - Mean Squared Error (MSE)
  - Root Mean Squared Error (RMSE)
  - Categorical Crossentropy

## Example Notebooks

Explore practical applications of the library with the following Jupyter notebooks:
- [California Housing Regression](https://github.com/papaya147/sequential-neural-nets/blob/main/notebooks/california-housing.ipynb)
- [MNIST Digit Classification](https://github.com/papaya147/sequential-neural-nets/blob/main/notebooks/mnist-digits-flat.ipynb)
- [MNIST Fashion Classification](https://github.com/papaya147/sequential-neural-nets/blob/main/notebooks/mnist-fashion-flat.ipynb)

## Installation

To get started, clone the repository and set up the environment:

```bash
git clone https://github.com/papaya147/sequential-neural-nets.git
cd sequential-neural-nets
conda create --prefix ./env python=3.11
conda activate ./env
pip install -r requirements.txt
```
Set the Python environment in the notebooks to run them now!

### Prerequisites
- Conda (recommended for environment management)
- Dependencies in `requirements.txt`

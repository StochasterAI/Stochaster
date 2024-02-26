# Stochaster: A Neural Network Framework

## Introduction

**Stochaster** is a versatile and user-friendly artificial neural network framework. Designed with clarity, conciseness, and readability in mind, it aims to simplify the process of creating and understanding neural networks.

## Features

- **Autograd Engine**: Implements reverse-mode autodifferentiation for efficient backpropagation.
  - Customizable tensors via the `Tensor` class.
  - An autodiff wrapper for custom functions through the `Function` class.

- **Neural Network Library**: A brief yet powerful collection of tools for building neural networks.
  - Supports common layer types and optimizers.
  - Currently provides MNIST-level capabilities.

## Paradigms

- **Code Readability**: The engine's source code prioritizes the `Tensor` and `Function` abstractions to ensure that anyone familiar with the mathematics of backpropagation can easily understand and work with the framework.
- **API Design**: The API for both the engine and neural network library is designed to be like PyTorch.


Stochaster Org. also provides secondary-level educational Jupyter Notebooks on ANNs in [Notebooks](https://github.com/StochasterAI/Notebooks) covering from top-level APIs (e.g. PyTorch) to linear algebra/calculus present in basic ANNs.

## Installation/Build

The only option for using Stochaster is building from source for now.
PyPi package (via pip) will be available after succesful tests

## Example Usage

- **Engine Example**: _Section to be completed._
- **Neural Network Examples**: Check out the `examples` folder for practical implementations of Stochaster's NN and Engine libraries. Currently, it features an MNIST classifier implemented using cross-entropy loss and SGD for optimization.

## Tracing and Visualization

- _Upcoming Features_: Tracing for backpropagation and its visualization to enhance understanding and debugging capabilities.

## Running Tests

To run the unit tests you will have to install [PyTorch](https://pytorch.org/). Then simply:

```bash
python -m pytest
```

## License

Stochaster is made available under the MIT License, promoting open and permissive software use and redistribution.

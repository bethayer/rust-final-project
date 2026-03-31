# Rust Final Project

Members: Benjamin Thayer (bthayer3), Scott Phan (sphan7), Andrew Cheng (ascheng2)

## Introduction

For this project, we aim to implement a Feedforward Neural Network (FNN) / Multilayer Perceptron model in Rust. Our goal is to train this network on images of handwritten digits, in order for it to be able to recognize input images of numbers with high accuracy. We chose this project because it provides a relatively straightforward introduction to most machine learning technologies and how they work at the lowest level.

## Technical Overview

Our project will implement:
- Data input and parameter initialization
- Backward propagation
- Forward propagation
- Activation and softmax functions
- Training loop w/ these processes
- Prediction/inference
- Analysis of trained model accuracy

**Checkpoint 1 (4/13 - 4/17):** Implement the low-level classes and corresponding methods for network structures and training processes

**Checkpoint 2 (4/27 - 5/1):** Test, debug and train network and confirm ability to read handwritten digits

## Possible Challenges

- Transferring our calculus and linear algebra knowledge into Rust
- Integrating all the components of our project and debugging issues that might occur within the neural network

## References

- [Building neural network to read handwritten letters in c++](https://medium.com/@thakeenathees/neural-network-from-scratch-c-e2dc8977646b)
- [3blue1brown video on neural network](https://www.youtube.com/watch?v=aircAruvnKk&t=148s)
- [Andrej Karpathy (OpenAI, Tesla) intro to neural networks and back propagation](https://www.youtube.com/watch?v=VMj-3S1tku0)
  - [micrograd (library from video)](https://github.com/karpathy/micrograd)
- [Neural Network from scratch in python no pytorch or tensor flow - Samson Zhang](https://www.youtube.com/watch?v=w8yWXqWQYmU)
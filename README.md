# C-NN-Framework
![C++](https://img.shields.io/badge/C++-20-%2300599C?logo=cplusplus)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Finished-success)
![Accuracy](https://img.shields.io/badge/MNIST%20Accuracy-97.97%25-brightgreen)

# Motivation
My main goal for this project was to obtain a deeper understanding of how neural networks operate under the hood, focusing on the mathematics behind Deep Learning and low-level memory optimization.

# Results (MNIST)
By using the framework, I was able to achieve **97.97% accuracy** on the MNIST dataset (10,000 images), which borders the theoretical limit of architectures without convolutional layers.

| Metrics | Value                  |
| :--- |:-----------------------|
| **Architecture** | 784 -> 128 -> 32 -> 10 |
| **Epochs** | 10                     |
| **Final cost (CE)** | ~0.006                 |
| **Test accuracy** | **~98.00%**            |

### Cost over epochs:
![Cost decrease](https://github.com/piotrowski-j46/C-NN-Framework/blob/main/Assets/cost_cropped.gif?raw=true)

### Predictions:
![Predictions](https://github.com/piotrowski-j46/C-NN-Framework/blob/main/Assets/pred_cropped.gif?raw=true)

# How it works?
This framework was written using only the **standard C++ library** (STL). It's based on a **custom-built Matrix Engine**, optimized for cache efficiency, which provides all the essential features required for neural network training. The framework works by providing mathematical tools, classes describing standard neural network layers, and a native IDX reader that allows testing the network with the MNIST dataset right away.

The training loop can be split into three essential parts:
1.  **Forward propagation** - The model saves the current input data in the specific layer's cache, calculates the outcome of the linear function on the input, and passes it forward.
2.  **Backward propagation** - The model calculates gradients of weights and biases of each layer (besides the final one) and passes them backwards to previous layers.
3.  **Weight update** - By using the gradient, layers can compute how "wrong" they are and use the result of this computation to adjust their weights and biases.

# Features
- **He-Initialization** - Implemented to prevent the "Dying ReLU" problem and ensure proper gradient flow in deep networks.
- **Stable Softmax** - Optimised implementation with numerical stability fixes (preventing NaN/Infinity issues).
- **Mini-Batch Gradient Descent** - Balances computation speed and convergence stability.
- **Cache-Friendly Matrix Engine** - Custom linear algebra library designed with row-major order for CPU cache optimization.
- **Native IDX Parser** - Reads MNIST database files directly without external dependencies.
- **Model Serialization** - Save and load trained weights/biases to reuse the model without retraining.
- **Math Tools** - Includes Z-Score Normalization and One-Hot Encoding for data preprocessing.

# Build 
### Prerequisites
* **C++ Compiler** supporting C++20 (GCC, Clang, or MSVC)
* **CMake** (version 3.10+)

### Compilation
The project uses CMake for cross-platform compatibility.

1. Create a build directory:
   ```bash
   mkdir build
   cd build
   
2. Generate build files:
   ```
   cmake ..

3. Compile the project (Release mode for maximum performance):
   ```
   cmake --build . --config Release

# Usage
Run the executable from the terminal. The framework supports two modes: train and test.

1. Train Model
   Trains the network on the MNIST dataset and saves the model to mnist_model.

   ```
   # Linux / macOS
   ./NeuralNet train

   # Windows
   .\NeuralNet.exe train
2. Test Accuracy
   Loads the saved model and evaluates performance on the 10k test images.

   ```
   # Linux / macOS
   ./NeuralNet test

   # Windows
   .\NeuralNet.exe test
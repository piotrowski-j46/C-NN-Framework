# C-NN-Framework
![C++](https://img.shields.io/badge/C++-20-%2300599C?logo=cplusplus)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Finished-success)
![Accuracy](https://img.shields.io/badge/MNIST%20Accuracy-97.97%25-brightgreen)

# Motivation
My main goal for this project was to obtain a deeper understanding of how neural networks operate under the hood, focusing on the mathematics behind Deep Learning and low-level memory optimization.

# Results (MNIST)
The framework achieves **97.97% accuracy** on the MNIST test dataset (10,000 images). This result approaches the theoretical limit for Multilayer Perceptron (MLP) architectures without convolutional layers.

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
   ```
   
2. Generate build files:
   ```
   cmake ..
   ```

3. Compile the project (Release mode for maximum performance):
   ```
   cmake --build . --config Release
   ```
# Usage
Run the executable from the terminal. The framework supports two modes: train and test.

1. Train Model
   Trains the network on the MNIST dataset and saves the model to mnist_model.

   ```
   # Linux / macOS
   ./NeuralNet train

   # Windows
   .\NeuralNet.exe train
   ```
2. Test Accuracy
   Loads the saved model and evaluates performance on the 10k test images.

   ```
   # Linux / macOS
   ./NeuralNet test

   # Windows
   .\NeuralNet.exe test
   ```
# Architecture
## Matrix engine
The core of the framework is a custom-built mathematical engine optimized for 
linear algebra operations essential to machine learning. 
The Matrix class encapsulates data storage, dimensionality, and transposition states. 
By maintaining internal awareness of matrix dimensions, the engine enforces strict 
shape validation, ensuring that all arithmetic operations are mathematically legal and 
preventing runtime dimension mismatch errors.
## Layers
To ensure architectural consistency and code reusability, the framework utilizes a 
polymorphic design based on an abstract Layer base class. This abstraction defines the interface for 
all specialized layers, mandating the implementation of the two critical phases of neural network 
operation:
- Forward Propagation: The process where the layer accepts input data, performs a specific 
transformation (linear or non-linear), and passes the result to the subsequent layer.
- Backward Propagation: The mechanism for training the network. It involves calculating the
gradients of the loss function with respect to the layer's inputs and parameters,
allowing the network to minimize error through optimization algorithms.
### Dense Layer
The DenseLayer (fully connected layer) is the fundamental building block 
where every input neuron is connected to every output neuron. Each Dense Layer manages two sets of
learnable parameters:
- Weights ($W$)
- Biases ($B$) <br>
The layer implements propagation logic as follows:
#### Forward Propagation
The forward pass computes the linear transformation of the input. For an input matrix $X$, the output $Y$ is calculated using the formula:$$Y = X \cdot W + B$$
Where:
- $X$ is the input vector/matrix.
- $W$ is the weights matrix.
- $B$ is the bias vector.
#### Backward Propagation
During the backward pass, the layer computes gradients to update its parameters and passes the error signal to previous layers. 
Given the output gradient ($\frac{\partial L}{\partial Y}$, denoted as $dY$) and learning rate $\alpha$:
1. Gradient of Weights ($dW$):Calculated by multiplying the transposed input by the output gradient:
$$dW = X^T \cdot dY$$
2. Gradient of Biases ($dB$): Derived by performing the column-wise sum on the output gradient:
$$dB = \sum dY$$
3. Input Gradient ($dX$): The error signal to be propagated to the previous layer:
$$dX = dY \cdot W^T$$
4. Parameter Update (Gradient Descent): Finally, the parameters are updated to minimize the loss:
$$W \leftarrow W - \alpha \cdot dW$$ $$B \leftarrow B - \alpha \cdot dB$$

### Activation Layer
Activation layers are critical components that introduce non-linearity into 
the network, enabling it to learn complex patterns and solve non-trivial 
problems that a simple linear combination could not. 
This framework implements two key activation functions:
#### ReLu
- Formula: $f(x) = \max(0, x)$
- Usage: The primary activation function used in hidden layers. 
It is computationally efficient and helps mitigate the vanishing gradient problem, 
allowing for faster convergence during training compared to traditional 
sigmoid/tanh functions.
- To maximize performance and prevent the "dying ReLU" problem (where neurons permanently stop learning), 
the framework employs He Initialization.While standard implementations often assume a Normal Distribution with
variance $\sigma^2 = \frac{2}{n}$, this framework samples weights from a Uniform Distribution.
To maintain the same variance required for proper signal propagation, the scaling boundaries must be adjusted. 
  - Normal Distribution 
    - Requires $\sigma = \sqrt{\frac{2}{n_{in}}}$. 
  - Uniform Distribution: 
    - To achieve the same variance of 
        $\frac{2}{n_{in}}$, the weights are sampled from the range $[-\sqrt{\frac{6}{n_{in}}}, \sqrt{\frac{6}{n_{in}}}]$.The 
        factor of 6 (instead of 2) in the numerator compensates for the inherent properties of the Uniform distribution 
        (where variance is $\frac{(range)^2}{12}$), ensuring consistent gradient flow throughout deep networks.
#### Sigmoid
- Formula: $\sigma(x) = \frac{1}{1 + e^{-x}}$
- Usage: Maps values to a range between $(0, 1)$.
While less common in deep hidden layers today, 
it is implemented for historical compatibility and specific binary 
classification tasks.
### Softmax Layer
The Softmax layer is typically used as the output layer for multi-class classification problems. 
It transforms raw output scores (logits) into a probability distribution, where all values sum to 1.0.
-Numerical Stability: The implementation utilizes the Log-Sum-Exp trick (subtracting the maximum value from input vector) to
prevent floating-point overflow/underflow and NaN propagation, ensuring stability even with large input values.
## Neural Network
The NeuralNetwork class serves as the high-level container and orchestrator of the model.

- **Layer Management**: It utilizes RAII (Resource Acquisition Is Initialization) principles via std::unique_ptr for automatic and safe memory management. Layers are stored polymorphically, allowing for flexible architecture construction.
- **Training Loop**: Encapsulates the forward and backward propagation cycles, managing data flow and parameter updates.

## IDXReader
A specialized utility class designed to parse the IDX file format (commonly used for the MNIST dataset).

- **Binary Parsing**: Efficiently reads binary data streams, automatically handling file headers and magic numbers.
- **Format Detection**: It distinguishes between image files (idx3-ubyte) and label files (idx1-ubyte), 
parsing metadata (dimensions, count) to load data correctly into the matrix engine.

## Loss
The framework employs a polymorphic design (Strategy Pattern) for loss calculation, decoupling the optimization goal 
from the network architecture.
- **Mean Squared Error (MSE)**: Standard loss for regression tasks.
- **Cross-Entropy Loss**: optimized for classification tasks. 
It includes epsilon clipping to prevent $\log(0)$ errors, ensuring numerical stability during training.

# Optimization
High performance is achieved through low-level memory and algorithmic optimizations:

- **Cache Locality**: The custom Matrix engine uses flattened 1D vectors instead of nested arrays, ensuring contiguous memory allocation. 
This drastically reduces CPU cache misses.
- **Transposed Multiplication**: Matrix multiplication logic detects transposition requirements and iterates over memory in 
row-major order, preventing costly memory jumps.
- **Parallelization**: Critical mathematical operations are parallelized using OpenMP, fully utilizing multi-core 
CPU architectures to accelerate training and inference.

# Performance
Benchmarks performed on the MNIST dataset (60,000 training images) using a standard desktop CPU with OpenMP enabled.

| Operation | Metric | Value |
| :--- | :--- | :--- |
| **Training Speed** | Time per Epoch | **~7.9 s** |
| **Training Throughput** | Images per Second | **~7,600 img/s** |
| **Inference Speed** | Full Test Set (10k imgs) | **< 20 ms** |
| **Inference Latency** | Single Image | **< 0.1 ms** (Real-time capable) |
| **Accuracy** | After 10 Epochs | **~98%** |

*Tested on Intel Core i7-3770k using OpenMP backend.*

# Interface
The framework provides a streamlined Command Line Interface (CLI) for easy interaction:

- **train mode**: Automatically loads MNIST data, preprocesses it (Z-Score normalization), builds the architecture, 
and runs the training loop with live progress tracking.
- **test mode**: Loads the pre-trained model weights and evaluates performance on the unseen test dataset (10,000 images).
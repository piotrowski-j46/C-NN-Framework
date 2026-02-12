//
// Created by piotr on 23.12.2025.
//

#include "ActivationLayer.h"


ActivationLayer::ActivationLayer(const std::function<double(double)>& activation_func,const std::function<double(double)>& activation_func_derivative) : input_cache(Matrix(1,1)),
    activation_func(activation_func), activation_func_derivative(activation_func_derivative) {
}



Matrix ActivationLayer::forward(const Matrix &input) {
    this->input_cache = input;
    return input.apply(activation_func);
}

Matrix ActivationLayer::backward(const Matrix &output_gradient, double learning_rate) {
    Matrix derivative = input_cache.apply(activation_func_derivative);
    return output_gradient.hadamard_prod(derivative);
}


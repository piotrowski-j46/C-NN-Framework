//
// Created by piotr on 09.02.2026.
//

#include "SoftMaxLayer.h"


SoftMaxLayer::SoftMaxLayer(const std::function<Matrix(const Matrix&)> &activation_func,
    const std::function<double(double, double)> &activation_func_derivative): input_cache(Matrix(1,1)),
        activation_func(activation_func), activation_func_derivative(activation_func_derivative) {
}


Matrix SoftMaxLayer::forward(const Matrix &input) {
    this->input_cache = input;
    return input.apply(activation_func);
}

Matrix SoftMaxLayer::backward(const Matrix &output_gradient, double learning_rate) {
    return output_gradient;
}
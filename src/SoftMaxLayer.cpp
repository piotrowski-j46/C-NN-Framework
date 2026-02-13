//
// Created by piotr on 09.02.2026.
//

#include "SoftMaxLayer.h"


SoftMaxLayer::SoftMaxLayer(const std::function<Matrix(const Matrix&)> &activation_func): input_cache(1,1),
activation_func(activation_func){
}


Matrix SoftMaxLayer::forward(const Matrix &input) {
    this->input_cache = input;
    return input.apply(activation_func);
}

Matrix SoftMaxLayer::backward(const Matrix &output_gradient, float learning_rate) {
    return output_gradient;
}
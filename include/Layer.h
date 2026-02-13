//
// Created by piotr on 23.12.2025.
//

#ifndef MICRO_NN_FRAMEWORK_LAYER_H
#define MICRO_NN_FRAMEWORK_LAYER_H
#include "Matrix.h"


class Layer {
public:
    virtual ~Layer() = default;

    virtual Matrix forward(const Matrix& input) = 0;
    virtual Matrix backward(const Matrix& output_gradient, float learning_rate) = 0;
};


#endif //MICRO_NN_FRAMEWORK_LAYER_H
//
// Created by piotr on 23.12.2025.
//

#ifndef MICRO_NN_FRAMEWORK_NEURALNETWORK_H
#define MICRO_NN_FRAMEWORK_NEURALNETWORK_H
#include <vector>

#include "Layer.h"
#include "MSE.h"


class NeuralNetwork {
public:

    void add_layer(Layer* layer);
    Matrix predict(Matrix input);
    void train(Matrix& input, const Matrix& target_values, const double learning_rate);
private:
    MSE mse;
    std::pmr::vector<std::unique_ptr<Layer>> layers;
};


#endif //MICRO_NN_FRAMEWORK_NEURALNETWORK_H
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

    void add_layer(std::unique_ptr<Layer> &layer);
    void set_loss(std::unique_ptr<Loss> &loss);
    [[nodiscard]] Matrix predict(Matrix input) const;
    void save(const std::string& directory) const;
    void load(const std::string& directory) const;
    void train(const Matrix& input, const Matrix& target_values, float learning_rate);
private:
    std::unique_ptr<Loss> loss_function;
    std::vector<std::unique_ptr<Layer>> layers;
};


#endif //MICRO_NN_FRAMEWORK_NEURALNETWORK_H
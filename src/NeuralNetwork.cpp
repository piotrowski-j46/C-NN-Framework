//
// Created by piotr on 23.12.2025.
//

#include "NeuralNetwork.h"

#include <utility>

#include "Timer.h"

void NeuralNetwork::add_layer(std::unique_ptr<Layer> &layer) {
    layers.push_back(std::move(layer));
}

void NeuralNetwork::set_loss(std::unique_ptr<Loss> &loss) {
    loss_function = std::move(loss);
}


Matrix NeuralNetwork::predict(Matrix input) const {
    Matrix result = std::move(input);
    for (auto& layer : layers) {
        result = layer->forward(result);
    }
    return result;
}

void NeuralNetwork::train(const Matrix& input, const Matrix& target_values,const float learning_rate) {
    Matrix activation = input;

    for (const auto& layer : layers) {
        activation = layer->forward(activation);
    }

    Matrix grad = loss_function->compute_gradient(activation, target_values);
    for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
        grad = (*it)->backward(grad, learning_rate);
    }
}

void NeuralNetwork::save(const std::string& directory) const {
    for (const auto& layer : layers) {
        layer->save_weights(directory,"layer");
    }
}

void NeuralNetwork::load(const std::string &directory) const {
    int counter = 1;
    for (const auto& layer : layers) {
        if (!layer->has_weights()) continue;
        layer->load_weights(directory, "layer_" + std::to_string(counter));
        ++counter;
    }
}




//
// Created by piotr on 23.12.2025.
//

#include "DenseLayer.h"

#include <filesystem>
#include <format>
#include <fstream>
#include <regex>

#include "MSE.h"
#include "Utils.h"

// DenseLayer::DenseLayer(const int input_size, const int output_size) : input_cache(Matrix(1,1)),
//                                                                       weights(Matrix::random(input_size, output_size)),
//                                                                       bias(Matrix(1, output_size)) {
// }

// output determines amount of neurons
//bias init according to andriej karpathy x:y pos/neg ratio
DenseLayer::DenseLayer(const int input_size, const int output_size) : input_cache(Matrix(1, 1)),
                                                                      bias(Matrix(1, output_size, 0.1)),
                                                                      weights(Matrix(1, 1)), velocity_weights(Matrix(1,1)),
                                                                      velocity_bias(1,1) {
    const float limit = std::sqrt(6.0 / input_size);
    weights = Matrix::random(input_size, output_size, -1 * limit, limit);
    velocity_weights = Matrix{weights.get_rows(), weights.get_columns()};
    velocity_bias = Matrix(bias.get_rows(), bias.get_columns());
}

// Calculates f(x) = wx + b (single neuron output)
Matrix DenseLayer::forward(const Matrix &input) {
    this->input_cache = input;
    return input * weights + bias;
}

Matrix DenseLayer::backward(const Matrix &output_gradient, double learning_rate) {
    // double momentum = 0.9; // magic number, add as paramater later
    Matrix input_t = input_cache.transpose_matrix();
    Matrix weights_t = weights.transpose_matrix();

    Matrix w_grad = input_t * output_gradient;
    Matrix bias_grad = output_gradient.sum_columns();
    Matrix back_grad = output_gradient * weights_t;

    // velocity_weights = (velocity_weights * momentum) - (w_grad*learning_rate);
    // velocity_bias = (velocity_bias * momentum) - (bias_grad*learning_rate);

    weights = weights - learning_rate * w_grad;
    bias = bias - learning_rate * bias_grad;
    //
    // weights = weights + velocity_weights;
    // bias = bias + velocity_bias;
    // std::cout << "BIAS UPDATE: " << std::endl;
    // output_gradient.sum_columns().print_matrix();
    return back_grad;
}

void DenseLayer::save_weights(const std::string& filename) {
    //check if directory exists
    //contains an info about amount of records
    const  std::filesystem::path path = Utils::get_path("weights");
    if (!std::filesystem::exists(path)) {
        std::cerr << "Weights directory not found, creating directory" << std::endl;
        std::filesystem::create_directory("weights");
    }
    std::ofstream save(path/filename);
    save << "WEIGHTS\n";
    save << weights.get_matrix().size() << "\n";
    for (const auto& entry : weights.get_matrix()) {
        save << entry << "\n";
    }
    save << "BIAS\n";
    save << bias.get_matrix().size() << "\n";
    for (const auto& entry : bias.get_matrix()) {
        save << entry << "\n";
    }

}

void DenseLayer::load_weights(std::string filename) {
    //if bias and weights aren't present throw exception: unkown fileformat
    //checks info for amount of records

    std::vector<float>* loc = nullptr;
    int weights_size = weights.get_matrix().size(), bias_size = bias.get_matrix().size();
    const std::filesystem::path path = Utils::get_path("weights", filename);
    const std::regex bias_reg("BIAS"), weights_reg("WEIGHTS");
    std::ifstream load(path);
    std::string buffer;

    if (!load.is_open()) {
        std::cerr << "File can't be opened!" << std::endl;
        return;
    }

    while (std::getline(load, buffer)) {
        if (std::regex_search(buffer, weights_reg)) {
            loc = &weights.get_matrix();
            loc->clear();
            std::getline(load,buffer);
            if (std::stoi(buffer) != weights_size) {
                std::cerr << "Dimension mismatch! Expected dimension: " << weights_size
                << " Provided dimension: " << buffer << std::endl;
                throw std::invalid_argument("Dimension mismatch");
            }
            continue;
        }
        if (std::regex_search(buffer, bias_reg)){
            loc = &bias.get_matrix();
            loc->clear();
            std::getline(load,buffer);
            if (std::stoi(buffer) != bias_size) {
                std::cerr << "Dimension mismatch! Expected dimension: " << bias_size
                << " Provided dimension: " << buffer << std::endl;
                throw std::invalid_argument("Dimension mismatch");
            }
            continue;
        }
        if (loc) loc->push_back(std::stof(buffer));
    }

    weights = Matrix(weights.get_rows(), weights.get_columns(), weights.get_matrix());
    bias = Matrix(bias.get_rows(), bias.get_columns(), bias.get_matrix());
}





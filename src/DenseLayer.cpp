//
// Created by piotr on 23.12.2025.
//

#include "DenseLayer.h"

#include <filesystem>
#include <format>
#include <fstream>
#include <regex>

#include "Utils.h"




// Bias initialization according to Andrej Karpathy x:y pos/neg ratio
DenseLayer::DenseLayer(const int input_size, const int output_size) : input_cache(1, 1),
                                                                      bias(1, output_size, 0.1),
                                                                      weights(Matrix::random(input_size, output_size,
                                                                          -std::sqrt(6.0f / input_size),
                                                                          std::sqrt(6.0f / input_size))) {
    /*
     * 6.0 is used in numerator as get_random() method uses uniform distribution instead of normal distribution.
     * The variance for uniform distribution is 2/3n while for normal distribution it's 2/n.
     * By using 6.0 in numerator we get the following formula: 6/3n = 2/n, which is perfectly aligned with normal distribution
     * and allows for perfect compatibility with he initialization required for ReLu.
     */
}

Matrix DenseLayer::forward(const Matrix &input) {
    this->input_cache = input;
    return input * weights + bias;
}

Matrix DenseLayer::backward(const Matrix &output_gradient, const float learning_rate) {
    const Matrix input_t = input_cache.transpose();
    const Matrix weights_t = weights.transpose();

    const Matrix w_grad = input_t * output_gradient;
    const Matrix bias_grad = output_gradient.sum_columns();
    Matrix back_grad = output_gradient * weights_t;

    weights = weights - learning_rate * w_grad;
    bias = bias - learning_rate * bias_grad;

    return back_grad;
}

void DenseLayer::save_weights(const std::string& directory, const std::string& filename) {
    std::filesystem::path path = Utils::get_path("weights");
    if (!std::filesystem::exists(path)) {
        std::cerr << "weights directory not found, creating directory" << std::endl;
        std::filesystem::create_directory(path);
    }

    if (!std::filesystem::exists(path/directory)) {
        std::cerr << directory << " directory not found, creating directory" << std::endl;
        std::filesystem::create_directory(path/directory);
    }
    std::string temp_filename = filename + "_1";
    int counter = 1;
    while (std::filesystem::exists(path/directory/temp_filename)) {
        temp_filename = filename + "_" + std::to_string(counter);
        ++counter;
    }
    std::ofstream save(path/directory/temp_filename);
    save << "WEIGHTS\n";
    save << weights.get_raw_data().size() << "\n";
    for (const auto& entry : weights.get_raw_data()) {
        save << entry << "\n";
    }
    save << "BIAS\n";
    save << bias.get_raw_data().size() << "\n";
    for (const auto& entry : bias.get_raw_data()) {
        save << entry << "\n";
    }

}

void DenseLayer::load_weights(const std::string& directory, const std::string &filename) {

    std::vector<float>* loc = nullptr;
    const int weights_size = weights.get_raw_data().size();
    const int bias_size = bias.get_raw_data().size();
    const std::filesystem::path path = Utils::get_path("weights/" + directory, filename);
    const std::regex bias_reg("BIAS"), weights_reg("WEIGHTS");
    std::ifstream load(path);
    std::string buffer;

    if (!load.is_open()) {
        std::cerr << "File can't be opened!" << std::endl;
        return;
    }

    while (std::getline(load, buffer)) {
        if (std::regex_search(buffer, weights_reg)) {
            loc = &weights.get_raw_data();
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
            loc = &bias.get_raw_data();
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

    weights = Matrix(weights.get_rows(), weights.get_columns(), weights.get_raw_data());
    bias = Matrix(bias.get_rows(), bias.get_columns(), bias.get_raw_data());
}





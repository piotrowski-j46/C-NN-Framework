//
// Created by piotr on 23.12.2025.
//

#ifndef MICRO_NN_FRAMEWORK_DENSELAYER_H
#define MICRO_NN_FRAMEWORK_DENSELAYER_H
#include "Layer.h"


class DenseLayer : public Layer{
public:
    DenseLayer(int input_size, int output_size);

    Matrix forward(const Matrix &input) override;
    Matrix backward(const Matrix &output_gradient, double learning_rate) override;
    void save_weights(const std::string& filename);
    void load_weights(const std::string filename);

private:
    Matrix input_cache;
    Matrix bias;
    Matrix weights;
    Matrix velocity_weights;
    Matrix velocity_bias;
};


#endif //MICRO_NN_FRAMEWORK_DENSELAYER_H
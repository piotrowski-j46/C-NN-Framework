//
// Created by piotr on 23.12.2025.
//

#ifndef MICRO_NN_FRAMEWORK_DENSELAYER_H
#define MICRO_NN_FRAMEWORK_DENSELAYER_H
#include "Layer.h"


class DenseLayer final : public Layer{
public:
    DenseLayer(int input_size, int output_size);

    Matrix forward(const Matrix &input) override;
    Matrix backward(const Matrix &output_gradient, float learning_rate) override;
    void save_weights(const std::string& directory, const std::string& filename) override;
    void load_weights(const std::string& directory, const std::string &filename) override;
    [[nodiscard]] bool has_weights() const override{return true;};

private:
    Matrix input_cache;
    Matrix bias;
    Matrix weights;
};


#endif //MICRO_NN_FRAMEWORK_DENSELAYER_H
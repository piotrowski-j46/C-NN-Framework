//
// Created by piotr on 09.02.2026.
//

#ifndef MICRO_NN_FRAMEWORK_SOFTMAXLAYER_H
#define MICRO_NN_FRAMEWORK_SOFTMAXLAYER_H
#include "Layer.h"


class SoftMaxLayer final : public Layer{
public:
    SoftMaxLayer(const std::function<Matrix(const Matrix&)>& activation_func);

    Matrix forward(const Matrix &input) override;
    Matrix backward(const Matrix &output_gradient, float learning_rate) override;
    void save_weights(const std::string& directory, const std::string& filename) override{};
    void load_weights(const std::string& directory, const std::string &filename) override{};
    [[nodiscard]] bool has_weights() const override{return false;};
private:
    Matrix input_cache;
    std::function<Matrix(const Matrix &)> activation_func;
};


#endif //MICRO_NN_FRAMEWORK_SOFTMAXLAYER_H
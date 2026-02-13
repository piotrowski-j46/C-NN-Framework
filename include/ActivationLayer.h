//
// Created by piotr on 23.12.2025.
//

#ifndef MICRO_NN_FRAMEWORK_ACTIVATIONLAYER_H
#define MICRO_NN_FRAMEWORK_ACTIVATIONLAYER_H
#include "Layer.h"


class ActivationLayer final : public Layer{
public:
    ActivationLayer(const std::function<float(float)>& activation_func,
                    const std::function<float(float)>& activation_func_derivative);


    Matrix forward(const Matrix &input) override;
    Matrix backward(const Matrix &output_gradient, float learning_rate) override;
    void save_weights(const std::string& directory, const std::string& filename) override{};
    void load_weights(const std::string& directory, const std::string &filename) override{};
    [[nodiscard]] bool has_weights() const override{return false;};
private:
    Matrix input_cache;
    std::function<float(float)> activation_func;
    std::function<float(float)> activation_func_derivative;
};


#endif //MICRO_NN_FRAMEWORK_ACTIVATIONLAYER_H
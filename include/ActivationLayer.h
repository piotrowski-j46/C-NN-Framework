//
// Created by piotr on 23.12.2025.
//

#ifndef MICRO_NN_FRAMEWORK_ACTIVATIONLAYER_H
#define MICRO_NN_FRAMEWORK_ACTIVATIONLAYER_H
#include "Layer.h"


class ActivationLayer : public Layer{
public:
    ActivationLayer(const std::function<double(double)>& activation_func,
                    const std::function<double(double)>& activation_func_derivative);


    Matrix forward(const Matrix &input) override;
    Matrix backward(const Matrix &output_gradient, double learning_rate) override;
private:
    Matrix input_cache;
    std::function<double(double)> activation_func;
    std::function<double(double)> activation_func_derivative;
};


#endif //MICRO_NN_FRAMEWORK_ACTIVATIONLAYER_H
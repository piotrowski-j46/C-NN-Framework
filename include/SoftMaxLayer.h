//
// Created by piotr on 09.02.2026.
//

#ifndef MICRO_NN_FRAMEWORK_SOFTMAXLAYER_H
#define MICRO_NN_FRAMEWORK_SOFTMAXLAYER_H
#include "Layer.h"


class SoftMaxLayer final : public Layer{
public:
    SoftMaxLayer(const std::function<Matrix(const Matrix&)>& activation_func,
                    const std::function<double(double,double)>& activation_func_derivative);

    Matrix forward(const Matrix &input) override;
    Matrix backward(const Matrix &output_gradient, double learning_rate) override;
private:
    Matrix input_cache;
    std::function<Matrix(Matrix &)> activation_func;
    std::function<double(double, double)> activation_func_derivative;
};


#endif //MICRO_NN_FRAMEWORK_SOFTMAXLAYER_H
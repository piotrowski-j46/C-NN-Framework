//
// Created by piotr on 24.12.2025.
//

#ifndef MICRO_NN_FRAMEWORK_MSE_H
#define MICRO_NN_FRAMEWORK_MSE_H
#include "Loss.h"


class MSE final : public Loss{
public:
    float compute_cost(const Matrix &predicted, const Matrix &target) override;
    Matrix compute_gradient(const Matrix &predicted, const Matrix &target) override;
};


#endif //MICRO_NN_FRAMEWORK_MSE_H
//
// Created by piotr on 24.12.2025.
//

#ifndef MICRO_NN_FRAMEWORK_LOSS_H
#define MICRO_NN_FRAMEWORK_LOSS_H
#include "Matrix.h"

#endif //MICRO_NN_FRAMEWORK_LOSS_H

class Loss {
public:
    virtual ~Loss() = default;
    virtual double compute_cost(const Matrix& predicted, const Matrix& target) = 0;
    virtual double cross_entropy(const Matrix &predicted, const Matrix& target) = 0;
    virtual Matrix compute_gradient(const Matrix& predicted, const Matrix& target) = 0;
};
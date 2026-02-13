//
// Created by piotr on 24.12.2025.
//

#ifndef MICRO_NN_FRAMEWORK_LOSS_H
#define MICRO_NN_FRAMEWORK_LOSS_H
#include "Matrix.h"


class Loss {
public:
    virtual ~Loss() = default;
    virtual float compute_cost(const Matrix& predicted, const Matrix& target) = 0;
    virtual Matrix compute_gradient(const Matrix& predicted, const Matrix& target) = 0;
};

#endif //MICRO_NN_FRAMEWORK_LOSS_H

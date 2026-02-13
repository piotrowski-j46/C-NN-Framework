//
// Created by piotr on 13.02.2026.
//

#ifndef C__NN_FRAMEWORK_CROSSENTROPY_H
#define C__NN_FRAMEWORK_CROSSENTROPY_H
#include "Loss.h"


class CrossEntropy final: public Loss{
public:
    float compute_cost(const Matrix &predicted, const Matrix& target) override;
    Matrix compute_gradient(const Matrix &predicted, const Matrix& target) override;
};


#endif //C__NN_FRAMEWORK_CROSSENTROPY_H
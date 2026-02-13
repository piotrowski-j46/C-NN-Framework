//
// Created by piotr on 13.02.2026.
//

#include "CrossEntropy.h"

#include "Utils.h"

float CrossEntropy::compute_cost(const Matrix &predicted, const Matrix &target) {
    const int n = predicted.get_rows();
    const Matrix log_mul = predicted.apply(Utils::log).hadamard_prod(target);
    const float cost = -1*log_mul.sum();
    return cost/n; // Cost divided by batch size to show error per prediction
}

Matrix CrossEntropy::compute_gradient(const Matrix &predicted, const Matrix &target) {
    const int n = predicted.get_rows();
    return (predicted-target)/(1.0*n);
}


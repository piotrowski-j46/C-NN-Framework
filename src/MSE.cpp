//
// Created by piotr on 24.12.2025.
//

#include "MSE.h"

#include "Utils.h"

double MSE::compute_cost(const Matrix &predicted, const Matrix &target) {
    const int n = predicted.get_rows();
    const double total_cost = Matrix::pow(predicted - target).sum()/(2*n);
    return total_cost;
}


Matrix MSE::compute_gradient(const Matrix &predicted, const Matrix &target) {
    const int n = predicted.get_rows();
    return (predicted-target)/(1.0*n);
}

double MSE::cross_entropy(const Matrix &predicted, const Matrix &target) {
    const Matrix log_mul = predicted.apply(Utils::log).hadamard_prod(target);
    const double cost = -1*log_mul.sum();
    return cost;
}





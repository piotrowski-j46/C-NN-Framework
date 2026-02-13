//
// Created by piotr on 24.12.2025.
//

#include "MSE.h"

#include "Utils.h"

float MSE::compute_cost(const Matrix &predicted, const Matrix &target) {
    const int n = predicted.get_rows();
    const float total_cost = Matrix::pow(predicted - target).sum()/(2.0f*n);
    return total_cost;
}


Matrix MSE::compute_gradient(const Matrix &predicted, const Matrix &target) {
    const int n = predicted.get_rows();
    return (predicted-target)/(1.0*n);
}





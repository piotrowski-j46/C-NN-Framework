//
// Created by piotr on 16.12.2025.
//

#include "Matrix.h"
#include <format>
#include <iostream>
#include <random>
#include "Timer.h"


// Constructors

Matrix::Matrix(const int row, const int column, const std::vector<float> &values) {
    if (values.size() != column * row) {
        std::cerr << "Matrix dimensions don't match! " << row << ", " << column <<
                " Change number of columns/row or number of values." << std::endl;
        throw std::invalid_argument("Wrong dimensions!");
    }
    dimension_ = Dimension{row, column};
    data.reserve(row * column);
    for (size_t i = 0; i < row; i++) {
        for (size_t j = 0; j < column; j++) {
            data.push_back(values[j + (i * column)]);
        }
    }
    T.reserve(row * column);
    T = transpose_raw();
}

Matrix::Matrix(const int rows, const int cols) {
    dimension_ = {rows, cols};
    data.reserve(rows * cols);
    data.resize(rows * cols, 0);
    T.reserve(rows * cols);
    T = transpose_raw();
}

Matrix::Matrix(const int rows, const int cols, const float value) {
    dimension_ = {rows, cols};
    data.reserve(rows * cols);
    data.resize(rows * cols, value);
    T.reserve(rows * cols);
    T = transpose_raw();
}

// Accessors

std::vector<float> &Matrix::get_raw_data() {
    return data;
}

Matrix::Dimension Matrix::get_dimension() const {
    return dimension_;
}

int Matrix::get_columns() const {
    return dimension_.column;
}

int Matrix::get_rows() const {
    return dimension_.row;
}

float Matrix::operator()(const int r, const int c) const {
    if (r >= dimension_.row || c >= dimension_.column) throw std::out_of_range("Index out of bounds");
    return data[r * dimension_.column + c];
}

float &Matrix::operator()(const int r, const int c) {
    if (r >= dimension_.row || c >= dimension_.column) throw std::out_of_range("Index out of bounds");
    return data[r * dimension_.column + c];
}

// Element access

float Matrix::operator[](const int n) const {
    return data[n];
}

// Factories

Matrix Matrix::random(const int rows, const int cols, const float lower_bound, const float upper_bound) {
    std::vector<float> rand_values;
    rand_values.reserve(cols * rows);

    for (size_t i = 0; i < rows * cols; i++) {
        rand_values.push_back(get_random_value(lower_bound, upper_bound));
    }

    Matrix random_matrix = Matrix(rows, cols, rand_values);
    return random_matrix;
}

float Matrix::get_random_value(const float lower_bound, const float upper_bound) {
    std::uniform_real_distribution<float> unif(lower_bound, upper_bound);
    std::random_device rd;
    return unif(rd);
}

// Arithmetic operators

Matrix Matrix::operator+(const Matrix &m) const {
    if (dimension_ != m.get_dimension() &&
        !(dimension_.column == m.get_columns() && m.get_rows() == 1)) {
        std::cerr << "Dimensions: " << dimension_ << ", " << m.get_dimension() << " don't match!" << std::endl;
        throw std::invalid_argument("Dimension mismatch!");
    }

    Matrix result(dimension_.row, dimension_.column);

    if (dimension_ == m.get_dimension()) {
        for (size_t i = 0; i < data.size(); i++) {
            result.data[i] = data[i] + m.data[i];
        }
    } else {
        for (size_t i = 0; i < data.size(); i++) {
            result.data[i] = data[i] + m.data[i % dimension_.column];
        }
    }
    result.T = result.transpose_raw();
    return result;
}

Matrix Matrix::operator-(const Matrix &m) const {
    if (dimension_ != m.get_dimension() &&
        !(dimension_.column == m.get_columns() && m.get_rows() == 1)) {
        std::cerr << "Dimensions: " << dimension_ << ", " << m.get_dimension() << " don't match!" << std::endl;
        throw std::invalid_argument("Dimension mismatch!");
    }

    Matrix result(dimension_.row, dimension_.column);

    if (dimension_ == m.get_dimension()) {
        for (size_t i = 0; i < data.size(); i++) {
            result.data[i] = data[i] - m.data[i];
        }
    } else {
        for (size_t i = 0; i < data.size(); i++) {
            result.data[i] = data[i] - m.data[i % dimension_.column];
        }
    }
    result.T = result.transpose_raw();
    return result;
}

Matrix Matrix::operator*(const Matrix &m) const {
    if (dimension_.column != m.dimension_.row) {
        std::cerr << "Dimensions: " << dimension_ << ", " << m.get_dimension() << " don't match!" << std::endl;
        throw std::invalid_argument("Dimension mismatch!");
    }

    Matrix result(dimension_.row, m.dimension_.column);
    // OpenMP parallelization for performance
#pragma omp parallel for
    for (int i = 0; i < get_rows(); ++i) {
        for (int j = 0; j < m.get_columns(); ++j) {
            float val = 0;
            const int row_offset = (i * m.get_columns()) + j;
            // Optimization: Accessing 'm' via transposed cache for linear memory access (cache locality)
            for (int k = 0; k < get_columns(); ++k) {
                // Direct vector access avoids function call overhead and bounds checking in tight loop
                val += data[k + (i * get_columns())] * m.T[k + (j * m.get_rows())];
            }
            result.data[row_offset] = val;
        }
    }

    result.T = result.transpose_raw(); // Transpose data refresh after getting the result
    return result;
}

// Scalar operators

Matrix Matrix::operator*(const float d) const {
    Matrix result(dimension_.row, dimension_.column);
    for (size_t i = 0; i < result.data.size(); i++) {
        result.data[i] = data[i] * d;
    }
    result.T = result.transpose_raw();
    return result;
}

Matrix Matrix::operator/(const float d) const {
    Matrix result(dimension_.row, dimension_.column);
    const float inv_d = 1 / d;
    for (size_t i = 0; i < result.data.size(); i++) {
        result.data[i] = data[i] * inv_d;
    }
    result.T = result.transpose_raw();
    return result;
}

Matrix operator*(const float d, const Matrix &m) {
    return m * d;
}

Matrix operator+(const Matrix &m, const float d) {
    return m + Matrix(1, m.get_columns(), d);
}

Matrix operator-(const Matrix &m, const float d) {
    return m - Matrix(1, m.get_columns(), d);
}

// Mathematical operations

Matrix Matrix::softmax(const Matrix &m) {
    Matrix distribution{m.get_rows(), m.get_columns()};

    for (int i = 0; i < m.get_rows(); ++i) {
        const float *row_start = &m.data[i * m.get_columns()],
                *row_end = row_start + m.get_columns();
        float sum = 0;
        const float max_val = *std::max_element(row_start, row_end);

        for (int j = 0; j < m.get_columns(); ++j) {
            float val = row_start[j];
            sum += std::exp(val - max_val);
        }
        float inv_sum = 1.0f / sum;
        for (int j = 0; j < m.get_columns(); ++j) {
            const float val = row_start[j];
            distribution.data[j + (i * m.get_columns())] = std::exp(val - max_val) * inv_sum;
        }
    }
    distribution.T = distribution.transpose_raw();
    return distribution;
}

Matrix Matrix::hadamard_prod(const Matrix &m) const {
    if (dimension_ != m.dimension_) {
        std::cerr << "Dimensions: " << dimension_ << ", " << m.get_dimension() << " don't match!" << std::endl;
        throw std::invalid_argument("Dimension mismatch!");
    }

    Matrix result = Matrix(get_rows(), get_columns());

    for (size_t i = 0; i < get_rows() * get_columns(); i++) {
        result.data[i] = data[i] * m.data[i];
    }
    result.T = result.transpose_raw();
    return result;
}


Matrix Matrix::pow(const Matrix &m) {
    return m.hadamard_prod(m);
}

std::vector<float> Matrix::transpose_raw() const{
    std::vector<float> transposed = data;
    for (size_t i = 0, ac = 0; i < dimension_.column; ++i) {
        for (size_t j = 0; j < dimension_.row; ++j, ++ac) {
            transposed[ac] = data[j*dimension_.column + i];
            // transposed[ac] = (*this)(j,i);
        }
    }
    return transposed;
}

Matrix Matrix::transpose() const{
    Matrix transposed{dimension_.column, dimension_.row, transpose_raw()};
    return transposed;
}

float Matrix::sum() const {
    float sum = 0;
    for (const float val: data) {
        sum += val;
    }
    return sum;
}

Matrix Matrix::sum_columns() const {
    Matrix sum{1, get_columns()};
    for (int i = 0; i < get_columns(); ++i) {
        float col_sum = 0;
        for (int j = 0; j < get_rows(); ++j) {
            col_sum += data[i + j * get_columns()];
        }
        sum.data[i] = col_sum;
    }
    sum.T = sum.transpose_raw();
    return sum;
}


// Activation/functional

Matrix Matrix::apply(const std::function<float(float)> &func) const {
    Matrix result = Matrix(get_rows(), get_columns());
    for (size_t i = 0; i < get_rows() * get_columns(); i++) {
        result.data[i] = func(data[i]);
    }
    result.T = result.transpose_raw();
    return result;
}

Matrix Matrix::apply(const std::function<float(float, float)> &func) const {
    Matrix result = Matrix(get_rows(), get_columns());
    for (size_t i = 0; i < get_rows() * get_columns(); i++) {
        result.data[i] = func(data[i], data[i]);
    }
    result.T = result.transpose_raw();
    return result;
}

Matrix Matrix::apply(const std::function<Matrix(Matrix &)> &func) const {
    Matrix copy = *this;
    Matrix result = func(copy);
    result.T = result.transpose_raw();
    return result;
}

// Debugging

void Matrix::print_matrix() const {
    for (size_t i = 0; i < dimension_.row; i++) {
        for (size_t j = 0; j < dimension_.column; j++) {
            std::cout << std::format("{:.6f}", (data[j + (i * dimension_.column)])) << " ";
        }
        std::cout << std::endl;
    }
}

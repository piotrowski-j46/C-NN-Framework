//
// Created by piotr on 16.12.2025.
//

#include "Matrix.h"


#include <format>
#include <iostream>
#include <random>

#include "Timer.h"

Matrix::Matrix(const int row, const int column, const std::vector<float>& values) {
    if (values.size() != column*row) {
        std::cerr << "Matrix dimensions don't match! " << row <<  ", "<< column <<" Change number of columns/row or number of values." << std::endl;
        throw std::invalid_argument("Wrong dimensions!");
    }
    dimension_ = Dimension{row, column};
    matrix.reserve(row*column);
    for (size_t i = 0; i < row; i++) {
        for (size_t j = 0; j < column; j++) {
            matrix.push_back(values[j + (i*column)]);
        }
    }
    // std::cout <<" TEN KONSTRUKTOR WYPIERDALA PROGRAM!!! " << std::endl;
    T.reserve(row*column);
    T = transpose();
}

Matrix::Matrix(const int rows, const int cols){
    dimension_ = {rows, cols};
    matrix.reserve(rows * cols);
    matrix.resize(rows*cols, 0);
    T.reserve(rows*cols);
    // std::cout <<" TEN KONSTRUKTOR WYPIERDALA PROGRAM!!! " << std::endl;
    T = transpose();
    // for (const auto& entry : T) {
    //     std::cout << entry << std::endl;
    // }
}

Matrix::Matrix(const int rows, const int cols, double value){
    // std::cout <<" TEN KONSTRUKTOR WYPIERDALA PROGRAM!!! " << std::endl;
    //no  T method, needs to be implemnted
    dimension_ = {rows, cols};
    matrix.reserve(rows * cols);
    matrix.resize(rows*cols, value);
    T.reserve(rows*cols);
    T = transpose();
}



float Matrix::get_random_value(const float lower_bound, const float upper_bound) {
    std::uniform_real_distribution<float> unif(lower_bound,upper_bound);
    std::random_device rd;
    return unif(rd);
}

Matrix Matrix::softmax(const Matrix &m) {
    //add - max val
    Matrix distribution{m.get_rows(), m.get_columns()};
    for (int i = 0; i < m.get_rows(); ++i) {
        const float* row_start = &m.matrix[i*m.get_columns()],
        * row_end = row_start + m.get_columns();
        float sum = 0;
        const float max_val = *std::max_element(row_start, row_end);
        for (int j = 0; j < m.get_columns(); ++j) {
            float val = row_start[j];
            sum += std::exp(val - max_val);
        }
        for (int j = 0; j < m.get_columns(); ++j) {
            float val = row_start[j];
            distribution.matrix[j + (i*m.get_columns())] = std::exp(val - max_val)/sum;
        }
    }
    return distribution;
}


Matrix Matrix::random(const int rows,const int cols, const float lower_bound, const float upper_bound) {
    std::vector <float> rand_values;
    rand_values.reserve(cols*rows);
    for (size_t i = 0; i < rows*cols; i++) {
        rand_values.push_back(get_random_value(lower_bound, upper_bound));
    }
    Matrix random_matrix = Matrix (rows,cols, rand_values);
    return random_matrix;
}


Matrix::Dimension Matrix::get_dimension() const{
    return dimension_;
}

int Matrix::get_columns() const {
    return dimension_.column;
}

int Matrix::get_rows() const {
    return dimension_.row;
}

Matrix Matrix::hadamard_prod(const Matrix &m) const{
    if (dimension_ != m.dimension_) {
        std::cerr << "Dimensions: "  << dimension_ << ", " << m.get_dimension() << " don't match!" << std::endl;
        throw std::invalid_argument("Dimension mismatch!");
    }

    Matrix result = Matrix(get_rows(), get_columns());

    for (size_t i = 0; i < get_rows()*get_columns(); i ++) {
        result.matrix[i] = matrix[i] * m.matrix[i];
    }
    result.T = result.transpose();
    return result;
}

Matrix Matrix::apply(const std::function<double(double)>& func) const {
    Matrix result = Matrix(get_rows(), get_columns());
    for (size_t i = 0; i < get_rows()*get_columns(); i++) {
        result.matrix[i] = func(matrix[i]);
    }
    result.T = result.transpose();
    return result;
}

Matrix Matrix::apply(const std::function<double(double, double)>& func) const {
    Matrix result = Matrix(get_rows(), get_columns());
    for (size_t i = 0; i < get_rows()*get_columns(); i++) {
        result.matrix[i] = func(matrix[i], matrix[i]);
    }
    result.T = result.transpose();
    return result;
}

Matrix Matrix::apply(const std::function<Matrix(Matrix&)>& func) const {
    Matrix copy = *this;
    Matrix result = func(copy);
    result.T = result.transpose();
    return result;
}

Matrix Matrix::pow(const Matrix &m) {
    return m.hadamard_prod(m);
}

float Matrix::operator[](int n) const {
    return matrix[n];
}

std::vector<float>& Matrix::get_matrix() {
    return matrix;
}

Matrix Matrix::operator+(const Matrix &m) const {
    if (dimension_ != m.get_dimension() &&
        !(dimension_.column == m.get_columns() && m.get_rows() == 1)) {
        std::cerr << "Dimensions: "  << dimension_ << ", " << m.get_dimension() << " don't match!" << std::endl;
        throw std::invalid_argument("Dimension mismatch!");
    }

    Matrix result(dimension_.row, dimension_.column);

    if (dimension_ == m.get_dimension()) {
        for (size_t i = 0; i < matrix.size(); i++) {
            result.matrix[i] = matrix[i] + m.matrix[i];
        }
    }else {
        for (size_t i = 0; i < matrix.size(); i++) {
            result.matrix[i] = matrix[i]+m.matrix[i % dimension_.column];
        }
    }
    result.T = result.transpose();
    return result;
}

Matrix Matrix::operator-(const Matrix &m) const {
    if (dimension_ != m.get_dimension() &&
        !(dimension_.column == m.get_columns() && m.get_rows() == 1)) {
        std::cerr << "Dimensions: "  << dimension_ << ", " << m.get_dimension() << " don't match!" << std::endl;
        throw std::invalid_argument("Dimension mismatch!");
    }

    Matrix result(dimension_.row, dimension_.column);

    if (dimension_ == m.get_dimension()) {
        for (size_t i = 0; i < matrix.size(); i++) {
            result.matrix[i] = matrix[i] - m.matrix[i];
        }
    }else {
        for (size_t i = 0; i < matrix.size(); i++) {
            result.matrix[i] = matrix[i] - m.matrix[i % dimension_.column];
        }
    }
    result.T = result.transpose();
    return result;
}

Matrix Matrix::operator*(const Matrix &m) const {
    if (dimension_.column != m.dimension_.row) {
        std::cout << "TUTAJ WYPIERDALA! " << std::endl;
        std::cerr << "Dimensions: "  << dimension_ << ", " << m.get_dimension() << " don't match!" << std::endl;
        throw std::invalid_argument("Dimension mismatch!");
    }

    Matrix result(dimension_.row,m.dimension_.column);

    #pragma omp parallel for
    for (int i =0; i < get_rows(); ++i) {
        for (int j = 0; j < m.get_columns(); ++j) {
            float val = 0;
            int row_offset = (i*m.get_columns()) + j;
            for (int k = 0; k < get_columns(); ++k) {
                // std::cout << matrix[k+(i*get_columns())] << "*"  << m.T[k+(j*m.get_rows())] <<  " ";
                val += matrix[k+(i*get_columns())] * m.T[k+(j*m.get_rows())];
            }
            // std::cout << "row_offset " << row_offset << std::endl;
            result.matrix[row_offset] = val;
        }
    }
    result.T = result.transpose();
    return result;
}

Matrix Matrix::operator*(const double d) const {
    Matrix result(dimension_.row, dimension_.column);
    for (size_t i = 0; i < result.matrix.size(); i++) {
        result.matrix[i] = matrix[i]*d;
    }
    result.T = result.transpose();
    return result;
}

Matrix Matrix::operator/(const double d) const {
    Matrix result(dimension_.row, dimension_.column);
    const double inv_d = 1/d;
    for (size_t i = 0; i < result.matrix.size(); i++) {
        result.matrix[i] = matrix[i] * inv_d;
    }
    result.T = result.transpose();
    return result;
}

std::vector<float> Matrix::transpose(){
    // maybe swap can be implemented????
    std::vector<float> transposed = matrix;
    for (size_t i = 0, ac = 0; i < dimension_.column; ++i) {
        for (size_t j = 0; j < dimension_.row; ++j, ++ac) {
            transposed[ac] = (*this)(j, i);
        }
    }
    return transposed;
}

Matrix Matrix::transpose_matrix(){
    Matrix transposed{dimension_.column, dimension_.row, transpose()};
    return transposed;
}

double Matrix::sum() const {
    double sum = 0;
    for (const double val : matrix) {
        sum += val;
    }
    return sum;
}

Matrix Matrix::sum_columns() const {
    Matrix sum{1, get_columns()};
    for (int i = 0; i < get_columns(); ++i) {
        float col_sum = 0;
        for (int j = 0; j < get_rows(); ++j) {
            col_sum += (*this)(j,i);
        }
        sum.matrix[i] = col_sum;
    }
    sum.T = sum.transpose();
    return sum;
}


Matrix operator*(const double d, const Matrix &m){
    return m*d;
}

Matrix operator+(const Matrix &m, const double d) {
    return m + Matrix(1, m.get_columns(), d);
}

Matrix operator-(const Matrix &m, const double d) {
    return m - Matrix(1, m.get_columns(), d);
}


float& Matrix::operator()(const int r, const int c) {
    if (r >= dimension_.row || c >= dimension_.column) throw std::out_of_range("Index out of bounds");
    // std::cout << "INDEX: " << r * dimension_.column + c << std::endl;
    return matrix[r * dimension_.column + c];
}

float Matrix::operator()(const int r, const int c) const {
    // std::cout << "INDEX: " << r * dimension_.column + c << std::endl;
    if (r >= dimension_.row || c >= dimension_.column) throw std::out_of_range("Index out of bounds");
    return matrix[r * dimension_.column + c];
}



void Matrix::print_matrix() const {
    for (size_t i = 0; i < dimension_.row; i++) {
        for (size_t j = 0; j < dimension_.column; j++) {
            std::cout << std::format("{:.6f}",(matrix[j + (i*dimension_.column)] ))<< " ";
        }
        std::cout << std::endl;
    }
}

//
// Created by piotr on 16.12.2025.
//

#ifndef MICRO_NN_FRAMEWORK_MATRIX_H
#define MICRO_NN_FRAMEWORK_MATRIX_H
#include <functional>
#include <iostream>
#include <memory>
#include <random>
#include <vector>


class Matrix {
public:
    struct Dimension {
        int row;
        int column;

        friend bool operator ==(const Dimension& lhs, const Dimension& rhs) {
            return lhs.column == rhs.column && lhs.row == rhs.row;
        }
        friend bool operator !=(const Dimension& lhs, const Dimension& rhs) {
            return !(lhs == rhs);
        }

        friend std::ostream& operator<<(std::ostream& os, const Dimension& d) {
            os << "[" << d.row << ", " << d.column << "]";
            return os;
        }
    };

    // Constructors
    Matrix(int row, int column, const std::vector<float>& values);
    explicit Matrix(int rows, int cols); // Zero initialized
    Matrix(int rows, int cols, float value); // Scalar fill

    // Accessors
    std::vector<float>& get_raw_data();
    [[nodiscard]] Dimension get_dimension() const;
    [[nodiscard]] int get_rows() const;
    [[nodiscard]] int get_columns() const;
    float operator()(int r, int c) const;
    float& operator()(int r, int c);

    // Element access
    float operator [](int n) const;

    // Factories
    Matrix static random(int rows, int cols, float lower_bound, float upper_bound);
    [[nodiscard]] static float get_random_value(float lower_bound, float upper_bound);

    // Arithmetic operators
    Matrix operator +(const Matrix &m) const;
    Matrix operator -(const Matrix &m) const;
    Matrix operator *(const Matrix &m) const;

    // Scalar operators
    friend Matrix operator *(float d, const Matrix &m);
    friend Matrix operator +(const Matrix &m, float d);
    friend Matrix operator -(const Matrix &m, float d);
    Matrix operator *(float d) const;
    Matrix operator /(float d) const;


    // Mathematical operations
    [[nodiscard]] Matrix hadamard_prod(const Matrix& m) const;
    static Matrix softmax(const Matrix& m);
    static Matrix pow(const Matrix &m);
    [[nodiscard]] std::vector<float> transpose_raw() const;
    [[nodiscard]] Matrix transpose() const;
    [[nodiscard]] float sum() const;
    [[nodiscard]] Matrix sum_columns() const;

    // Activation/functional
    Matrix apply(const std::function<float(float)>& func) const;
    Matrix apply(const std::function<float(float, float)> &func) const;
    Matrix apply(const std::function<Matrix(Matrix&)> &func) const;

    // Debugging
    void print_matrix() const;

private:
    Dimension dimension_;
    std::vector<float> data;
    std::vector<float> T;
};


#endif //MICRO_NN_FRAMEWORK_MATRIX_H
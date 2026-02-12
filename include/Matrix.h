//
// Created by piotr on 16.12.2025.
// Class that represents matrix and allows for matrix operations such as transposing and matrices arithmetics.
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

    Matrix(int row, int column, const std::vector<float>& values);
    explicit Matrix(int rows, int cols);
    Matrix(int rows, int cols, double value);

    std::vector<float>& get_matrix(); //helper method for matrix access
    Matrix static random(int rows, int cols, float lower_bound, float upper_bound);
    Matrix operator +(const Matrix &m) const;
    Matrix operator -(const Matrix &m) const;
    Matrix operator *(const Matrix &m) const;
    friend Matrix operator *(double d, const Matrix &m);
    friend Matrix operator +(const Matrix &m, double d);
    friend Matrix operator -(const Matrix &m, double d);
    float operator [](int n) const;
    Matrix operator *(double d) const;
    Matrix operator /(double d) const;
    float& operator()(int r, int c);
    float operator()(int r, int c) const;
    [[nodiscard]] Matrix hadamard_prod(const Matrix& m) const;

    static Matrix softmax(const Matrix& m);
    static Matrix pow(const Matrix &m);
    Matrix apply(const std::function<double(double)>& func) const;

    Matrix apply(const std::function<double(double, double)> &func) const;

    Matrix apply(const std::function<Matrix(Matrix&)> &func) const;

    [[nodiscard]] static float get_random_value(float lower_bound, float upper_bound);
    [[nodiscard]] std::vector<float> transpose();
    [[nodiscard]] Matrix transpose_matrix();
    [[nodiscard]] double sum() const;
    [[nodiscard]] Matrix sum_columns() const;
    [[nodiscard]] Dimension get_dimension() const;
    [[nodiscard]] int get_rows() const;
    [[nodiscard]] int get_columns() const;
    void print_matrix() const;

private:
    Dimension dimension_;
    std::vector<float> matrix;
    std::vector<float> T;
};


#endif //MICRO_NN_FRAMEWORK_MATRIX_H
//
// Created by piotr on 23.12.2025.
//

#ifndef MICRO_NN_FRAMEWORK_UTILS_H
#define MICRO_NN_FRAMEWORK_UTILS_H
#include <cmath>
#include <filesystem>
#include "Matrix.h"
//softmax needs to be added (output activation function)
namespace Utils {

    inline float mean = 0;
    inline float std = 0;
    inline double sigmoid(const double x) {
        return 1/(1+std::exp(-1 * x));
    }

    inline double relu(const double x) {
        return x > 0 ? x : 0;
    }

    inline double sigmoid_derivative(const double x) {
        double z = sigmoid(x);
        return z * (1-z);
    }

    inline double relu_derivative(const double x) {
        return x > 0 ? 1 : 0;
    }

    inline Matrix one_hot_encode(const std::vector<float>& labels, int num_classes) {
        int n = labels.size();
        Matrix target{n, num_classes, 0.0};

        for (int i = 0; i < n; ++i) {
            if (labels[i] < num_classes) {
                target(i,labels[i]) = 1.0;
            }
        }
        return target;
    }

    inline double log(const double x) {
        // 1e-9 in case of zero
        return std::log(x + 1e-9);
    }

    inline double softmax_derivative(const double x, const double y) {
        return x - y;
    }

    inline Matrix get_batch(Matrix& samples, int start, int batch_size) {
        int cols = samples.get_columns();
        int rows_to_take = std::min(batch_size, samples.get_rows() - start);

        int start_index = start * cols;
        int end_index = start_index + (rows_to_take*cols);
        std::vector<float> values{samples.get_matrix().begin() + start_index,
                                    samples.get_matrix().begin() + end_index};
        return Matrix{rows_to_take, samples.get_columns(), values};
    }



    inline Matrix z_score_normalization(const Matrix& m) {
        if (mean == 0 && std == 0) {
            mean = m.sum()/(m.get_rows()*m.get_columns());
            std = sqrtf(Matrix::pow(m-mean).sum()/(m.get_rows()*m.get_columns()));
        }

        return (m-mean)/std;
    }

    inline std::filesystem::path get_path(const std::string& directory, const std::string& filename = "") {
        std::filesystem::path current_dir = std::filesystem::current_path();
        for (int i = 0; i < 3; ++i) {
            std::filesystem::path potential_path = current_dir / directory / filename;

            if (std::filesystem::exists(potential_path)) {
                return potential_path;
            }

            if (current_dir.has_parent_path()) {
                current_dir = current_dir.parent_path();
            }else {
                break;
            }
        }
        throw std::runtime_error("CRITICAL ERROR: Could not find: '"  + directory  + "\\" + filename +
                             "' in " + std::filesystem::current_path().string() + " or its parents.");
    }

}
#endif //MICRO_NN_FRAMEWORK_UTILS_H
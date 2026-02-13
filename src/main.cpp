#include <filesystem>
#include <iostream>

#include "ActivationLayer.h"
#include "CrossEntropy.h"
#include "DenseLayer.h"
#include "IDXReader.h"
#include "Matrix.h"
#include "NeuralNetwork.h"
#include "SoftMaxLayer.h"
#include "Timer.h"
#include "Utils.h"

constexpr int INPUT_SIZE = 784;
constexpr int HIDDEN_1 = 128;
constexpr int HIDDEN_2 = 32;
constexpr int OUTPUT_SIZE = 10;
float LEARNING_RATE = 0.08;
constexpr int EPOCHS = 10;
constexpr int BATCH_SIZE = 32;
constexpr int TOTAL_SAMPLES = 60000;

std::unique_ptr<NeuralNetwork> build_mnist_network() {
    auto net = std::make_unique<NeuralNetwork>();
    std::unique_ptr<Loss> CE = std::make_unique<CrossEntropy>();

    std::unique_ptr<Layer> dense_l1 = std::make_unique<DenseLayer>(INPUT_SIZE,HIDDEN_1);
    std::unique_ptr<Layer> dense_l2 = std::make_unique<DenseLayer>(HIDDEN_1,HIDDEN_2);
    std::unique_ptr<Layer> dense_l3= std::make_unique<DenseLayer>(HIDDEN_2,OUTPUT_SIZE);
    std::unique_ptr<Layer> activation_l1 = std::make_unique<ActivationLayer>(Utils::relu, Utils::relu_derivative);
    std::unique_ptr<Layer> activation_l2 = std::make_unique<ActivationLayer>(Utils::relu, Utils::relu_derivative);
    std::unique_ptr<Layer> activation_l3 = std::make_unique<SoftMaxLayer>(Matrix::softmax);

    net->set_loss(CE);

    net->add_layer(dense_l1);
    net->add_layer(activation_l1);
    net->add_layer(dense_l2);
    net->add_layer(activation_l2);
    net->add_layer(dense_l3);
    net->add_layer(activation_l3);

    return net;
}

float calculate_accuracy(const Matrix& predictions, const std::vector<float>& targets) {
    int correct = 0;
    const int total = predictions.get_rows();

    for (int i = 0; i < targets.size(); i++) {
        float best_digit = 0;
        double highest_prob = -1.0;
        for (int j = 0; j < 10; j++) {
            if (const double val = predictions(i, j); val > highest_prob) {
                highest_prob = val;
                best_digit = j;
            }
        }
        if (static_cast<int>(best_digit) == static_cast<int>(targets[i])) ++correct;
    }
    return static_cast<float>(correct) / total;
}

void print_header() {
    std::cout << "============================================" << std::endl;
    std::cout << "   C++ NEURAL NETWORK FRAMEWORK (MNIST)     " << std::endl;
    std::cout << "   High-Performance | Cache-Optimized       " << std::endl;
    std::cout << "============================================" << std::endl;
}

void train_mode() {
    CrossEntropy ce;
    std::cout << "[INFO] Loading MNIST Training Data..." << std::endl;

    auto raw_images = IDXReader::load_mnist("train-images.idx3-ubyte");
    auto raw_labels = IDXReader::load_mnist("train-labels.idx1-ubyte");

    if(raw_images.empty() || raw_labels.empty()) {
        std::cerr << "[ERROR] Failed to load dataset." << std::endl;
        return;
    }

    std::cout << "[INFO] Preprocessing data (Normalization & One-Hot)..." << std::endl;
    Matrix X(60000, 784, raw_images);
    Utils::reset_normalization();
    X = Utils::z_score_normalization(X);

    Matrix Y = Utils::one_hot_encode(raw_labels, 10);

    auto net = build_mnist_network();
    std::cout << "[INFO] Network Architecture Built: " << INPUT_SIZE << "->" << HIDDEN_1 << "->" << HIDDEN_2 << "->" << OUTPUT_SIZE << std::endl;

    std::cout << "[INFO] Starting Training (" << EPOCHS << " epochs, LR: " << LEARNING_RATE << ")..." << std::endl;


    for (int epoch = 1; epoch <= EPOCHS; ++epoch) {

        std::cout << "Epoch " << epoch << "/" << EPOCHS << " [";

        for(int b=0; b < TOTAL_SAMPLES; b+=BATCH_SIZE) {
            Matrix x_batch = Utils::get_batch(X, b, BATCH_SIZE);
            Matrix y_batch = Utils::get_batch(Y, b, BATCH_SIZE);

            net->train(x_batch, y_batch, LEARNING_RATE);

            if (b % (TOTAL_SAMPLES / 10) == 0) std::cout << "=" << std::flush;
        }
        std::cout << "] Done." << std::endl;

        Matrix pred = net->predict(X);
        /*
         * Multiple trial and error runs made me believe that reducing LR by 90% on 7th epoch
         * makes training more effective
         */
        if (epoch == 7) LEARNING_RATE *= 0.1;
        std::cout << "EPOCH: "  << epoch << " COST: " << ce.compute_cost(pred, Y) << std::endl;
    }

    std::cout << "[INFO] Saving model to 'mnist_model'..." << std::endl;
    net->save("mnist_model");
    std::cout << "[SUCCESS] Training complete." << std::endl;
}


void test_mode() {
    std::cout << "[INFO] Loading MNIST Test Data..." << std::endl;
    auto raw_images = IDXReader::load_mnist("t10k-images.idx3-ubyte");
    auto raw_labels = IDXReader::load_mnist("t10k-labels.idx1-ubyte");

    if(raw_images.empty() || raw_labels.empty()) return;

    Matrix X(10000, 784, raw_images);
    X = Utils::z_score_normalization(X);

    auto net = build_mnist_network();
    std::cout << "[INFO] Loading pretrained weights..." << std::endl;
    try {
        net->load("mnist_model");
    } catch (...) {
        std::cerr << "[ERROR] Could not load weights! Train the model first." << std::endl;
        return;
    }

    std::cout << "[INFO] Running inference on 10,000 images..." << std::endl;
    Matrix predictions = net->predict(X);

    float accuracy = calculate_accuracy(predictions, raw_labels);

    std::cout << "\n================ RESULT ================" << std::endl;
    std::cout << " Final Accuracy: " << std::fixed << std::setprecision(2) << accuracy * 100.0f << "%" << std::endl;
    std::cout << "========================================" << std::endl;
}

int main(const int argc, char* argv[]) {
    print_header();

    if (argc < 2) {
        std::cout << "Usage:\n";
        std::cout << "  ./NeuralNet train   - Train model from scratch\n";
        std::cout << "  ./NeuralNet test    - Test saved model accuracy\n";
        return 0;
    }

    if (const std::string mode = argv[1]; mode == "train") {
        train_mode();
    } else if (mode == "test") {
        test_mode();
    } else {
        std::cerr << "Unknown command: " << mode << std::endl;
    }

    return 0;
}

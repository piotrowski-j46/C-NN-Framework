#include <filesystem>
#include <iostream>

#include "ActivationLayer.h"
#include "DenseLayer.h"
#include "IDXReader.h"
#include "Matrix.h"
#include "NeuralNetwork.h"
#include "SoftMaxLayer.h"
#include "Timer.h"
#include "Utils.h"

// framework works for XOR problem, estimating with high certainty (>98%)

int main() {
    MSE mse;
    NeuralNetwork nn;
    IDXReader train;
    IDXReader test;

    std::vector<float> raw_labels = test.load_mnist("train-labels.idx1-ubyte");
    Matrix X_train {60000,784, train.load_mnist("train-images.idx3-ubyte")};
    Matrix Y_train = Utils::one_hot_encode(raw_labels,10);
    X_train = Utils::z_score_normalization(X_train);
    DenseLayer *l1 = new DenseLayer(784,128);
    DenseLayer *l2 = new DenseLayer(128,32);
    DenseLayer *l3 = new DenseLayer(32,10);
    nn.add_layer(l1);
    nn.add_layer(new ActivationLayer(Utils::relu, Utils::relu_derivative));
    nn.add_layer(l2);
    nn.add_layer(new ActivationLayer(Utils::relu, Utils::relu_derivative));
    nn.add_layer(l3);
    nn.add_layer(new SoftMaxLayer(Matrix::softmax, Utils::softmax_derivative));

    // // //BATCH START
    double learning_rate = 0.08;
    int batch_size = 32;
    int total_samples = 60000;

    // //BATCH END
    // for (int epoch = 0; epoch < 30; ++epoch) {
    //
    //     for (int i = 0; i < total_samples; i += batch_size) {
    //         Matrix batch_input = Utils::get_batch(X_train, i, batch_size);
    //         Matrix batch_target = Utils::get_batch(Y_train, i, batch_size);
    //         nn.train(batch_input,batch_target, learning_rate);
    //     }
    //
    //     if (epoch == 7 || epoch == 20)   learning_rate *= 0.1;
    //     std::cout << "EPOCH: " << epoch << " TOTAL COST: " << mse.cross_entropy(nn.predict(X_train), Y_train) << std::endl;;
    //
    // }


    // l1->save_weights("l1");
    // l2->save_weights("l2");
    // l3->save_weights("l3");
    //
    l1->load_weights("l1");
    l2->load_weights("l2");
    l3->load_weights("l3");

    IDXReader labels_f, img_f;
    std::vector<float> test_labels = labels_f.load_mnist("t10k-labels.idx1-ubyte");
    Matrix X_test {10000,784, img_f.load_mnist("t10k-images.idx3-ubyte")};
    X_test = Utils::z_score_normalization(X_test);
    Matrix prediction = nn.predict(X_test);
    double counter = 0;
    for (int i = 0; i < test_labels.size(); i++) {
        int best_digit = 0;
        double highest_prob = -1.0;
        for (int j = 0; j < 10; j++) {
            // std::cout << j << ":" << prediction(i,j) << " ";
            double val = prediction(i, j);
            if (val > highest_prob) {
                highest_prob = val;
                best_digit = j;
            }
        }
        // std::cout << std::endl << "TARGET VALUE: " << test_labels[i];
        // std::cout <<  " PREDICTION: " << best_digit << (best_digit == test_labels[i] ? "(True)" : "(False)") << std::endl << std::endl;
        if (best_digit == test_labels[i]) ++counter;
    }
    std::cout << "Success rate: " << (counter/10000.0)*100.0<< "% ";


}

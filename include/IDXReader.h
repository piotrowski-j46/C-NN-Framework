//
// Created by piotr on 05.02.2026.
//

#ifndef MICRO_NN_FRAMEWORK_IDXREADER_H
#define MICRO_NN_FRAMEWORK_IDXREADER_H
#include <vector>


class IDXReader {
public:
    struct Label {
        int magic_number = 0;
        int count = 0;
        std::vector<int> vals;
    };

    struct Image : public Label {
        int rows;
        int cols;
    };

    std::vector<float> load_mnist(const std::string& filename);
private:
    std::vector<Label> images_;
    std::vector<float> data;
};


#endif //MICRO_NN_FRAMEWORK_IDXREADER_H
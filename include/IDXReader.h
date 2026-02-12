//
// Created by piotr on 05.02.2026.
//

#ifndef MICRO_NN_FRAMEWORK_IDXREADER_H
#define MICRO_NN_FRAMEWORK_IDXREADER_H
#include <vector>


class IDXReader {
public:
    std::vector<float> load_mnist(const std::string& filename);
private:
    std::vector<float> data;
};


#endif //MICRO_NN_FRAMEWORK_IDXREADER_H
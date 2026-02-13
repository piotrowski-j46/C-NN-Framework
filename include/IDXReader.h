//
// Created by piotr on 05.02.2026.
//

#ifndef MICRO_NN_FRAMEWORK_IDXREADER_H
#define MICRO_NN_FRAMEWORK_IDXREADER_H
#include <vector>


class IDXReader {
public:
    static std::vector<float> load_mnist(const std::string& filename);
};


#endif //MICRO_NN_FRAMEWORK_IDXREADER_H
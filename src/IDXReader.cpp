//
// Created by piotr on 05.02.2026.
//
#include <iostream>
#include <fstream>
#include "IDXReader.h"

#include <filesystem>
#include <regex>

#include "Utils.h"

std::vector<float> IDXReader::load_mnist(const std::string& filename) {
    /*
     * Some indices have to be skipped since IDX data has some control values.
     * See https://yann.lecun.org/exdb/mnist/index.html?utm_source=chatgpt.com for more (expired certificate)
     */
    int skipped_indices = 0;
    const std::filesystem::path path = Utils::get_path("Data", filename);
    std::ifstream MNISTdata(path, std::ios::binary | std::ios::ate);

    if (!MNISTdata.is_open()) {
        std::cerr << "File can't be opened!" << std::endl;
        return std::vector<float>{};
    }

    const std::regex idx1("((\\.+idx1-ubyte))"), idx3("(\\.+idx3-ubyte)");
    if (std::regex_search(filename, idx1)) skipped_indices = 8;
    else if (std::regex_search(filename, idx3)) skipped_indices = 16;
    else {
        std::cerr << "Unkown filetype! " << std::endl;
        return std::vector<float>{};
    }
    const std::streamsize size = MNISTdata.tellg();
    data.reserve(size);
    MNISTdata.seekg(0, std::ios::beg);

    MNISTdata.seekg(skipped_indices);

    unsigned char pixel;
    int counter = 0;
    while (MNISTdata.read((char*)&pixel, 1)) {
        if (skipped_indices == 16) data.push_back(pixel); // - 0.5f ??
        else data.push_back(pixel);
        ++counter;
    }
    return data;
}


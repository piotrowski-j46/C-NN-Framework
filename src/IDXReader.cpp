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
    const std::filesystem::path path = Utils::get_path("Data", filename);
    std::ifstream file(path, std::ios::binary | std::ios::ate);

    if (!file.is_open()) {
        std::cerr << "File can't be opened!" << std::endl;
        return std::vector<float>{};
    }

    int header_size = 0;

    if (filename.find("idx1-ubyte") != std::string::npos) header_size = 8;
    else if (filename.find("idx3-ubyte") != std::string::npos) header_size = 16;
    else {
        std::cerr << "Unknown filetype! " << std::endl;
        return std::vector<float>{};
    }

    const std::streamsize size = file.tellg();
    const std::streamsize data_size = size - header_size;

    std::vector<unsigned char> raw_buffer(data_size);

    file.seekg(0, std::ios::beg);

    file.seekg(header_size);

    if (!file.read(reinterpret_cast<char*>(raw_buffer.data()), data_size)) {
        std::cerr << "Error reading data!" << std::endl;
        return {};
    }


    std::vector<float> result;
    result.reserve(size);

    for (const unsigned char byte : raw_buffer) {
        result.push_back(byte);
    }
    return result;
}


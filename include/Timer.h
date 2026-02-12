//
// Created by piotr on 06.02.2026.
//

#ifndef MICRO_NN_FRAMEWORK_TIMER_H
#define MICRO_NN_FRAMEWORK_TIMER_H
#include <chrono>

class Timer {
public:

    void start_measure() {
        start = std::chrono::high_resolution_clock::now();
    }

    [[nodiscard]] std::chrono::microseconds end_measure() const {
        return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start);
    }
private:
    std::chrono::system_clock::time_point start;
};


#endif //MICRO_NN_FRAMEWORK_TIMER_H
//
// Created by MurphySL on 2020/9/14.
//

#ifndef WEAVESS_UTIL_H
#define WEAVESS_UTIL_H

#include <random>
#include <algorithm>

namespace weavess {

    static void GenRandom(std::mt19937 &rng, unsigned *addr, unsigned size, unsigned N) {
        for (unsigned i = 0; i < size; ++i) {
            addr[i] = rng() % (N - size);
        }
        std::sort(addr, addr + size);
        for (unsigned i = 1; i < size; ++i) {
            if (addr[i] <= addr[i - 1]) {
                addr[i] = addr[i - 1] + 1;
            }
        }
        unsigned off = rng() % N;
        for (unsigned i = 0; i < size; ++i) {
            addr[i] = (addr[i] + off) % N;
        }
    }
}

#endif //WEAVESS_UTIL_H

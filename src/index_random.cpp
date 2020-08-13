//
// Created by Murph on 2020/8/12.
//
#include <weavess/index_random.h>

namespace weavess {

    IndexRandom::IndexRandom(const size_t dimension, const size_t n) : Index(dimension, n, L2) {
        has_built = true;
    }

    IndexRandom::~IndexRandom() {}

    void IndexRandom::Build(size_t n, const float *data, const Parameters &parameters) {
        data_ = data;
        nd_ = n;

        // Do Nothing

        has_built = true;
    }

    void
    IndexRandom::Search(const float *query, const float *x, size_t k, const Parameters &parameters, unsigned *indices) {

        GenRandom(rng, indices, k, nd_);
    }

}

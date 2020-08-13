//
// Created by Murph on 2020/8/12.
//

#ifndef WEAVESS_INDEX_RANDOM_H
#define WEAVESS_INDEX_RANDOM_H

#include "index.h"
#include "util.h"

namespace weavess {

    class IndexRandom : public Index {
    public:
        IndexRandom(const size_t dimension, const size_t n);

        virtual ~IndexRandom();

        std::mt19937 rng;

        void Save(const char *filename) override {}

        void Load(const char *filename) override {}

        virtual void Build(size_t n, const float *data, const Parameters &parameters) override;

        virtual void Search(
                const float *query,
                const float *x,
                size_t k,
                const Parameters &parameters,
                unsigned *indices) override;

    };

}

#endif //WEAVESS_INDEX_RANDOM_H

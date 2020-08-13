//
// Created by Murph on 2020/8/12.
//
#include <weavess/index.h>

namespace weavess {
    Index::Index(const size_t dimension, const size_t n, Metric metric = L2)
            : dimension_ (dimension), nd_(n), has_built(false) {
        switch (metric) {
            case L2:
                distance_ = new DistanceL2();
                break;
            default:
                distance_ = new DistanceL2();
                break;
        }
    }

    Index::~Index() {}
}


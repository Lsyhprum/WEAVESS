//
// Created by Murph on 2020/8/24.
//
#include "weavess/index.h"
#include "weavess/index_builder.h"

namespace weavess {
    void IndexComponentLoad::LoadInner(Index *index, char *data_file, Parameters &parameters) {
        float *data = nullptr;
        unsigned n{};
        unsigned dim{};

        load_data(data_file, data, n, dim);

        index->data_ = data;
        index->param_ = parameters;
        index->n_ = n;
        index->dim_ = dim;
    }
}

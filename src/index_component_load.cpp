//
// Created by Murph on 2020/8/24.
//
#include "weavess/index.h"
#include "weavess/index_builder.h"

namespace weavess {
    void IndexComponentLoad::LoadInner(Index *index, char *data_file, char *query_file, char *ground_file, Parameters &parameters) {

        // base_data
        float *data = nullptr;
        unsigned n{};
        unsigned dim{};
        load_data(data_file, data, n, dim);
        index->data_ = data;
        index->n_ = n;
        index->dim_ = dim;

        // query_data
        float *query_data = nullptr;
        unsigned query_num{};
        unsigned query_dim{};
        load_data<float>(query_file, query_data, query_num, query_dim);
        index->query_data_ = query_data;
        index->query_num_ = query_num;
        index->query_dim_ = query_dim;

        // ground_data
        unsigned *ground_data = nullptr;
        unsigned ground_num{};
        unsigned ground_dim{};
        load_data<unsigned>(ground_file, ground_data, ground_num, ground_dim);
        index->ground_data_ = ground_data;
        index->ground_num_ = ground_num;
        index->ground_dim_ = ground_dim;

        index->param_ = parameters;
    }
}

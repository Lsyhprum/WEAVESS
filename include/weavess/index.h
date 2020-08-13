//
// Created by Murph on 2020/8/12.
//

#ifndef WEAVESS_INDEX_H
#define WEAVESS_INDEX_H

#include <weavess/parameters.h>
#include <weavess/distance.h>
#include <weavess/util.h>

namespace weavess {
    class Index {
    public:
        explicit Index(const size_t dimension, const size_t n, Metric metric);

        virtual ~Index();

        virtual void Build(size_t n, const float *data, const Parameters &parameters) = 0;

        virtual void Search(
                const float *query,
                const float *x,
                size_t k,
                const Parameters &parameters,
                unsigned *indices) = 0;

        virtual void Save(const char *filename) = 0;

        virtual void Load(const char *filename) = 0;

        inline bool HasBuilt() const { return has_built; }

        inline size_t GetDimension() const { return dimension_; };

        inline size_t GetSizeOfDataset() const { return nd_; }

        inline const float *GetDataset() const { return data_; }

    protected:
        const size_t dimension_;
        const float *data_;
        size_t nd_;
        bool has_built;
        Distance *distance_;
        char *opt_graph_;
        size_t node_size;
        size_t data_len;
        size_t neighbor_len;
        size_t dist_cout = 0;
    };
}

#endif //WEAVESS_INDEX_H

//
// Created by Murph on 2020/8/14.
//
#include <cassert>
#include <algorithm>
#include "weavess/index_graph.h"
#include "weavess/util.h"

namespace weavess {

    IndexGraph::IndexGraph(const size_t dim, const size_t n) : Index(dim, n){
        has_built = true;

        assert(n_ != 0);
        assert(dim_ != 0);
    }

    IndexGraph::~IndexGraph() {}

    void IndexGraph::Build(const float *data, const Parameters &parameters) {
        unsigned K = parameters.Get<unsigned >("K");

        for(int i = 0; i < n_; i ++){
            for(int j = 0; j < n_; j ++){
                float dist = 0;

                if(i != j)
                    dist = distance_->compare(data_ + dim_ * i, data_ + dim_ * j, (unsigned)dim_);

                graph_[i].pool.emplace_back(Neighbor(i, dist));
            }

            std::sort(graph_[i].pool.begin(), graph_[i].pool.end());
            std::vector<int> knn;
            for(int a = 0; a < K; a ++)
                knn.emplace_back(graph_[i].pool[a].id);
            final_graph.emplace_back(knn);
        }
    }

    void IndexGraph::Search() {}

    void IndexGraph::Load(const char *filename) {}

    void IndexGraph::Save(const char *filename) {}

    void IndexGraph::InitGraph() {
        graph_.reserve(n_);
        for(unsigned i = 0; i < n_; i ++){
            graph_.emplace_back(NeighborPool());
        }
    }
}


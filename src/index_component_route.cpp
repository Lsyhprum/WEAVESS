//
// Created by Murph on 2020/8/29.
//
#include "weavess/index_builder.h"

namespace weavess {
    void IndexComponentRouteGreedy::RouteInner(const unsigned query_id, const unsigned K, unsigned *indices) {
        const auto L = index_->param_.get<unsigned>("L_search");

        std::vector<char> flags(index_->n_);

        int k=0;
        while(k < (int)L) {
            int nk = L;

            if (index_->retset[k].flag) {
                index_->retset[k].flag = false;
                unsigned n = index_->retset[k].id;

                for (unsigned m = 0; m < index_->final_graph_[n].size(); ++m) {
                    unsigned id = index_->final_graph_[n][m];
                    if(flags[id])continue;
                    flags[id] = 1;
                    float dist = index_->distance_->compare(index_->query_data_ + index_->query_dim_ * query_id, index_->data_ + index_->dim_ * id, (unsigned)index_->dim_);
                    if(dist >= index_->retset[L-1].distance)continue;
                    Neighbor nn(id, dist, true);
                    int r = InsertIntoPool(index_->retset.data(), L, nn);

                    //if(L+1 < retset.size()) ++L;
                    if(r < nk)nk=r;
                }
                //lock to here
            }
            if(nk <= k)k = nk;
            else ++k;
        }
        for(size_t i=0; i < K; i++){
            indices[i] = index_->retset[i].id;
        }
    }
}
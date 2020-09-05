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

    void IndexComponentRouteGuided::RouteInner(const unsigned query_id, const unsigned K, unsigned *indices) {
        const auto L = index_->param_.get<unsigned>("L_search");

        int k = 0;
        while (k < (int)L) {
            int nk = L;

            if (index_->retset[k].flag) {
                index_->retset[k].flag = false;
                unsigned n = index_->retset[k].id;

                unsigned div_dim_ = index_->Tn[n].div_dim;
                unsigned left_len = index_->Tn[n].left.size();
                unsigned right_len = index_->Tn[n].right.size();

                unsigned MaxM;
                bool left_flag = true;
                if(index_->query_data_ + index_->query_dim_ * query_id + div_dim_ < index_->data_ + index_->dim_ * n + div_dim_){
                    MaxM = left_len; //左子树邻居的个数
                }
                else {
                    MaxM = right_len; //右子树邻居的个数
                    left_flag = false;
                }
                for (unsigned m = 0; m < MaxM; ++m) {
                    unsigned id = index_->Tn[n].left[m];
                    if(!left_flag) id = index_->Tn[n].right[m];

                    // flag 重写
//                    if (flags[id]) continue;
//                    flags[id] = 1;
                    float *data = (float *)(index_->data_ + index_->dim_ * id);
                    float norm = *data;
                    data++;
                    float dist =
                            index_->distance_->compare(index_->query_data_ + index_->query_dim_ * query_id, data, (unsigned)index_->dim_); // 20-9-5 norm ?
                    //dist_cout++;
                    if (dist >= index_->retset[L - 1].distance) continue;
                    Neighbor nn(id, dist, true);
                    int r = InsertIntoPool(index_->retset.data(), L, nn);

                    // if(L+1 < index_->retset.size()) ++L;
                    if (r < nk) nk = r;
                }
            }
            if (nk <= k)
                k = nk;
            else
                ++k;
        }
        for (size_t i = 0; i < K; i++) {
            indices[i] = index_->retset[i].id;
        }
    }
}
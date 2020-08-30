//
// Created by Murph on 2020/8/24.
//
#include "weavess/index_builder.h"

namespace weavess {

    void IndexComponentSearchRandom::SearchInner(const unsigned int query, size_t K, unsigned int *indices, unsigned int &distcount) {

    }

    void IndexComponentSearchNSG::SearchInner(const unsigned int query, size_t K, unsigned int *indices, unsigned int &distcount) {
//        const unsigned L = parameters.get<unsigned>("L_search");
//        std::vector<Neighbor> retset(L + 1);
//        std::vector<unsigned> init_ids(L);
//        boost::dynamic_bitset<> flags{index_->n_, 0};
//        // std::mt19937 rng(rand());
//        // GenRandom(rng, init_ids.data(), L, (unsigned) index_->n_);
//
//        unsigned tmp_l = 0;
//        for (; tmp_l < L && tmp_l < index_->final_graph_[index_->ep_].size(); tmp_l++) {
//            init_ids[tmp_l] = index_->final_graph_[index_->ep_][tmp_l];
//            flags[init_ids[tmp_l]] = true;
//        }
//
//        while (tmp_l < L) {
//            unsigned id = rand() % index_->n_;
//            if (flags[id]) continue;
//            flags[id] = true;
//            init_ids[tmp_l] = id;
//            tmp_l++;
//        }
//
//        for (unsigned i = 0; i < init_ids.size(); i++) {
//            unsigned id = init_ids[i];
//            float dist =
//                    index_->distance_->compare(index_->data_ + index_->dim_ * id, query, (unsigned)index_->dim_);
//            retset[i] = Neighbor(id, dist, true);
//            // flags[id] = true;
//        }
//
//        std::sort(retset.begin(), retset.begin() + L);
//        int k = 0;
//        while (k < (int)L) {
//            int nk = L;
//
//            if (retset[k].flag) {
//                retset[k].flag = false;
//                unsigned n = retset[k].id;
//
//                for (unsigned m = 0; m < index_->final_graph_[n].size(); ++m) {
//                    unsigned id = index_->final_graph_[n][m];
//                    if (flags[id]) continue;
//                    flags[id] = 1;
//                    float dist =
//                            index_->distance_->compare(query, index_->data_ + index_->dim_ * id, (unsigned)index_->dim_);
//                    if (dist >= retset[L - 1].distance) continue;
//                    Neighbor nn(id, dist, true);
//                    int r = InsertIntoPool(retset.data(), L, nn);
//
//                    if (r < nk) nk = r;
//                }
//            }
//            if (nk <= k)
//                k = nk;
//            else
//                ++k;
//        }
//        for (size_t i = 0; i < K; i++) {
//            indices[i] = retset[i].id;
//        }
    }
}

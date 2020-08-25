//
// Created by Murph on 2020/8/24.
//
#include "weavess/index_builder.h"

namespace weavess {
    void IndexComponentInitRandom::InitInner() {
        std::cout << index_->n_ << std::endl;
        std::cout << index_->param_.ToString() << std::endl;
        const unsigned L = index_->param_.get<unsigned>("L");
        const unsigned S = index_->param_.get<unsigned>("S");

        index_->graph_.reserve(index_->n_);
        std::mt19937 rng(rand());
        for (unsigned i = 0; i < index_->n_; i++) {
            index_->graph_.push_back(nhood(L, S, rng, (unsigned) index_->n_));
        }

#pragma omp parallel for
        for (unsigned i = 0; i < index_->n_; i++) {
            std::vector<unsigned> tmp(S + 1);

            weavess::GenRandom(rng, tmp.data(), S + 1, index_->n_);

            for (unsigned j = 0; j < S; j++) {
                unsigned id = tmp[j];

                if (id == i)continue;
                float dist = index_->distance_->compare(index_->data_ + i * index_->dim_,
                                                        index_->data_ + id * index_->dim_,
                                                        (unsigned) index_->dim_);

                index_->graph_[i].pool.push_back(Neighbor(id, dist, true));
            }
            std::make_heap(index_->graph_[i].pool.begin(), index_->graph_[i].pool.end());
            index_->graph_[i].pool.reserve(L);
        }
    }

//    void IndexComponentInitKDTree::InitInner() {
//        float* mean_ = new float[index_->dim_];
//        float* var_ = new float[index_->dim_];
//        memset(mean_,0,index_->dim_*sizeof(float));
//        memset(var_,0,index_->dim_*sizeof(float));
//
//        /* Compute mean values.  Only the first SAMPLE_NUM values need to be
//          sampled to get a good estimate.
//         */
//        unsigned cnt = std::min((unsigned)index_->SAMPLE_NUM+1, count);
//        for (unsigned j = 0; j < cnt; ++j) {
//            const float* v = index_->data_ + indices[j] * index_->dim_;
//            for (size_t k=0; k<index_->dim_; ++k) {
//                mean_[k] += v[k];
//            }
//        }
//        float div_factor = float(1)/cnt;
//        for (size_t k=0; k<index_->dim_; ++k) {
//            mean_[k] *= div_factor;
//        }
//
//        /* Compute variances (no need to divide by count). */
//
//        for (unsigned j = 0; j < cnt; ++j) {
//            const float* v = index_->data_ + indices[j] * index_->dim_;
//            for (size_t k=0; k<index_->dim_; ++k) {
//                float dist = v[k] - mean_[k];
//                var_[k] += dist * dist;
//            }
//        }
//
//        /* Select one of the highest variance indices at random. */
//        cutdim = selectDivision(rng, var_);
//
//        cutval = mean_[cutdim];
//
//        unsigned lim1, lim2;
//
//        planeSplit(indices, count, cutdim, cutval, lim1, lim2);
//        //cut the subtree using the id which best balances the tree
//        if (lim1>count/2) index = lim1;
//        else if (lim2<count/2) index = lim2;
//        else index = count/2;
//
//        /* If either list is empty, it means that all remaining features
//         * are identical. Split in the middle to maintain a balanced tree.
//         */
//        if ((lim1==count)||(lim2==0)) index = count/2;
//        delete[] mean_;
//        delete[] var_;
//    }
}
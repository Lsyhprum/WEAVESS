//
// Created by Murph on 2020/8/29.
//
#include "weavess/index_builder.h"

namespace weavess {
    void IndexComponentEntryRand::EntryInner(const unsigned query_id) {
        const auto L = index_->param_.get<unsigned>("L_search");

        index_->retset.clear();
        index_->retset.reserve(L + 1);
        std::vector<unsigned> init_ids(L);
        std::mt19937 rng(rand());

        GenRandom(rng, init_ids.data(), L, (unsigned)index_->n_);

        std::vector<char> flags(index_->n_);
        memset(flags.data(), 0, index_->n_ * sizeof(char));
        for(unsigned i=0; i<L; i++){
            unsigned id = init_ids[i];
            float dist = index_->distance_->compare(index_->query_data_ + i * index_->query_dim_,
                                                    index_->data_ + id * index_->dim_,
                                                    (unsigned) index_->dim_);
            index_->retset[i]=Neighbor(id, dist, true);
        }
        std::sort(index_->retset.begin(), index_->retset.begin()+L);
    }

    void IndexComponentEntryKDT::EntryInner(unsigned int query_id) {
        const auto K = index_->param_.get<unsigned>("K");
        const auto L = index_->param_.get<unsigned>("L_search");
        const auto I = index_->param_.get<unsigned>("I_efanna");
        const auto TreeNum = index_->param_.get<unsigned>("nTrees");

        index_->retset.clear();
        index_->retset.reserve(L + 1);

        unsigned Nnode = L / K * TreeNum + 1;

        for(int i = 0; i < index_->tree_roots_.size(); i ++){
            Index::Node *root = index_->tree_roots_[i];

        }



    }

    void IndexComponentEntryKDT::DFSsearch(Index::Node* node, unsigned* indices, unsigned count, unsigned offset){
        //omp_set_lock(&rootlock);
        //std::cout<<node->treeid<<":"<<offset<<":"<<count<<std::endl;
        //omp_unset_lock(&rootlock);

        if(count <= index_->TNS){
            node->DivDim = -1;
            node->Lchild = nullptr;
            node->Rchild = nullptr;
            node->StartIdx = offset;
            node->EndIdx = offset + count;
            //add points

        }else{
            unsigned idx;
            unsigned cutdim;
            float cutval;
            meanSplit(rng, indices, count, idx, cutdim, cutval);
            node->DivDim = cutdim;
            node->DivVal = cutval;
            node->StartIdx = offset;
            node->EndIdx = offset + count;
            auto* nodeL = new Index::Node(); auto* nodeR = new Index::Node();
            node->Lchild = nodeL;
            nodeL->treeid = node->treeid;
            DFSbuild(nodeL, rng, indices, idx, offset);
            node->Rchild = nodeR;
            nodeR->treeid = node->treeid;
            DFSbuild(nodeR, rng, indices+idx, count-idx, offset+idx);
        }
    }
}
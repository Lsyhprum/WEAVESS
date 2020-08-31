//
// Created by Murph on 2020/8/29.
//
#include "weavess/index_builder.h"
#include <functional>

namespace weavess {
    // RAND
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


    // KDT
    void IndexComponentEntryKDT::EntryInner(unsigned int query_id) {
        const auto L = index_->param_.get<unsigned>("L_search");
        const auto TreeNum = index_->param_.get<unsigned>("nTrees");

        index_->retset.clear();
        //index_->retset.reserve(L + 1);

        unsigned Nnode = L / index_->TNS * TreeNum + 1;

        for(size_t i = 0; i < index_->tree_roots_.size(); i ++){
            std::stack<Index::Node*> st;
            Index::Node *root = index_->tree_roots_[i];
            SearchNearLeaf(root, st, root->treeid, query_id, Nnode, index_->retset);
        }

        std::sort(index_->retset.begin(), index_->retset.end());
        index_->retset.reserve(L+1);
    }

    void IndexComponentEntryKDT::SearchNearLeaf(Index::Node* node, std::stack<Index::Node*> &st, const size_t tree_id, const size_t query_id, size_t Nnode, std::vector<Neighbor> &retset){

        // 到达叶子节点
        if(node->Lchild == nullptr && node->Rchild == nullptr){
            node->visit = true;
            for(size_t i = node->StartIdx; i < node->EndIdx; i ++){
                unsigned nodeId = index_->LeafLists[tree_id][i];
                float dist = index_->distance_->compare(index_->query_data_ + query_id * index_->query_dim_,
                                                        index_->data_ + nodeId * index_->dim_,
                                                        (unsigned) index_->dim_);
                index_->retset.emplace_back(nodeId, dist, true);
            }
            Nnode -= 1;
            if(Nnode != 0){
                // 数量不够，返回上一次继续搜索
                Index::Node* parent = st.top(); st.pop();
                // 寻找未访问过的节点
                while(!st.empty() && parent->Lchild->visit && parent->Rchild->visit){
                    parent->visit = true;
                    st.pop();
                    parent = st.top();
                }
                st.pop();

                SearchNearLeaf(parent, st, tree_id, query_id, Nnode, retset);
            }
            return ;
        }

        const float *v = index_->query_data_ + query_id * index_->query_dim_;

        if(v[node->DivDim] < node->DivVal && !node->Lchild->visit){
            st.push(node);
            SearchNearLeaf(node->Lchild, st, tree_id, query_id, Nnode, retset);
        }else if(v[node->DivDim] >= node->DivVal && !node->Rchild->visit){
            st.push(node);
            SearchNearLeaf(node->Rchild, st, tree_id, query_id, Nnode, retset);
        }else if(v[node->DivDim] < node->DivVal && node->Lchild->visit){
            st.push(node);
            SearchNearLeaf(node->Rchild, st, tree_id, query_id, Nnode, retset);
        }else if(v[node->DivDim] >= node->DivVal && node->Rchild->visit){
            st.push(node);
            SearchNearLeaf(node->Lchild, st, tree_id, query_id, Nnode, retset);
        }

    }


    // MID
    void IndexComponentEntryCentroid::EntryInner(unsigned int query_id) {
        const unsigned L = index_->param_.get<unsigned>("L_search");
        index_->retset.clear();
        index_->retset.reserve(L + 1);
        std::vector<unsigned> init_ids(L);
        boost::dynamic_bitset<> flags{index_->n_, 0};
        // std::mt19937 rng(rand());
        // GenRandom(rng, init_ids.data(), L, (unsigned) index_->n_);

        unsigned tmp_l = 0;
        for (; tmp_l < L && tmp_l < index_->final_graph_[index_->ep_].size(); tmp_l++) {
            init_ids[tmp_l] = index_->final_graph_[index_->ep_][tmp_l];
            flags[init_ids[tmp_l]] = true;
        }

        while (tmp_l < L) {
            unsigned id = rand() % index_->n_;
            if (flags[id]) continue;
            flags[id] = true;
            init_ids[tmp_l] = id;
            tmp_l++;
        }

        for (unsigned i = 0; i < init_ids.size(); i++) {
            unsigned id = init_ids[i];
            float dist =
                    index_->distance_->compare(index_->data_ + index_->dim_ * id, index_->query_data_ + index_->query_dim_ * query_id, (unsigned)index_->dim_);
            index_->retset[i] = Neighbor(id, dist, true);
            // flags[id] = true;
        }

        std::sort(index_->retset.begin(), index_->retset.begin() + L);
    }
}
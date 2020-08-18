//
// Created by Murph on 2020/8/18.
//

#ifndef WEAVESS_INDEX_COMPONENT_INIT_H
#define WEAVESS_INDEX_COMPONENT_INIT_H

#include <omp.h>
#include <cassert>
#include "index_component.h"

namespace weavess {

    class IndexComponentInit : public IndexComponent {
    public:
        explicit IndexComponentInit(Index *index, Parameters param) : IndexComponent(index, std::move(param)) {}

        virtual void InitInner() = 0;
    };

    class IndexComponentInitRandom : public IndexComponentInit {
    public:
        explicit IndexComponentInitRandom(Index *index, Parameters param) : IndexComponentInit(index, std::move(param)) {}

        void InitInner() override {
            const unsigned L = param_.Get<unsigned>("L");
            const unsigned S = param_.Get<unsigned>("S");

            index_->graph_.reserve(index_->n_);
            std::mt19937 rng(rand());
            for (unsigned i = 0; i < index_->n_; i++) {
                index_->graph_.push_back(nhood(L, S, rng, (unsigned) index_->n_));
            }

#pragma omp parallel for
            for (unsigned i = 0; i < index_->n_; i++) {
                const float *query = index_->data_ + i * index_->dim_;
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
    };

    class IndexComponentInitKDTree : public IndexComponentInit {
    public:
        explicit IndexComponentInitKDTree(Index *index, Parameters param) : IndexComponentInit(index, std::move(param)) {}

        void meanSplit(std::mt19937& rng, unsigned* indices, unsigned count, unsigned& index, unsigned& cutdim, float& cutval){
            float* mean_ = new float[index_->dim_];
            float* var_ = new float[index_->dim_];
            memset(mean_,0,index_->dim_*sizeof(float));
            memset(var_,0,index_->dim_*sizeof(float));

            /* Compute mean values.  Only the first SAMPLE_NUM values need to be
              sampled to get a good estimate.
             */
            unsigned cnt = std::min((unsigned)index_->SAMPLE_NUM+1, count);
            for (unsigned j = 0; j < cnt; ++j) {
                const float* v = index_->data_ + indices[j] * index_->dim_;
                for (size_t k=0; k<index_->dim_; ++k) {
                    mean_[k] += v[k];
                }
            }
            float div_factor = float(1)/cnt;
            for (size_t k=0; k<index_->dim_; ++k) {
                mean_[k] *= div_factor;
            }

            /* Compute variances (no need to divide by count). */

            for (unsigned j = 0; j < cnt; ++j) {
                const float* v = index_->data_ + indices[j] * index_->dim_;
                for (size_t k=0; k<index_->dim_; ++k) {
                    float dist = v[k] - mean_[k];
                    var_[k] += dist * dist;
                }
            }

            /* Select one of the highest variance indices at random. */
            cutdim = selectDivision(rng, var_);

            cutval = mean_[cutdim];

            unsigned lim1, lim2;

            planeSplit(indices, count, cutdim, cutval, lim1, lim2);
            //cut the subtree using the id which best balances the tree
            if (lim1>count/2) index = lim1;
            else if (lim2<count/2) index = lim2;
            else index = count/2;

            /* If either list is empty, it means that all remaining features
             * are identical. Split in the middle to maintain a balanced tree.
             */
            if ((lim1==count)||(lim2==0)) index = count/2;
            delete[] mean_;
            delete[] var_;
        }

        void planeSplit(unsigned* indices, unsigned count, unsigned cutdim, float cutval, unsigned& lim1, unsigned& lim2){
            /* Move vector indices for left subtree to front of list. */
            int left = 0;
            int right = count-1;
            for (;; ) {
                const float* vl = index_->data_ + indices[left] * index_->dim_;
                const float* vr = index_->data_ + indices[right] * index_->dim_;
                while (left<=right && vl[cutdim]<cutval){
                    ++left;
                    vl = index_->data_ + indices[left] * index_->dim_;
                }
                while (left<=right && vr[cutdim]>=cutval){
                    --right;
                    vr = index_->data_ + indices[right] * index_->dim_;
                }
                if (left>right) break;
                std::swap(indices[left], indices[right]); ++left; --right;
            }
            lim1 = left;//lim1 is the id of the leftmost point <= cutval
            right = count-1;
            for (;; ) {
                const float* vl = index_->data_ + indices[left] *index_->dim_;
                const float* vr = index_->data_ + indices[right] *index_->dim_;
                while (left<=right && vl[cutdim]<=cutval){
                    ++left;
                    vl = index_->data_ + indices[left] *index_->dim_;
                }
                while (left<=right && vr[cutdim]>cutval){
                    --right;
                    vr = index_->data_ + indices[right] *index_->dim_;
                }
                if (left>right) break;
                std::swap(indices[left], indices[right]); ++left; --right;
            }
            lim2 = left;//lim2 is the id of the leftmost point >cutval
        }

        int selectDivision(std::mt19937& rng, float* v){
            int num = 0;
            size_t topind[index_->RAND_DIM];

            //Create a list of the indices of the top index_->RAND_DIM values.
            for (size_t i = 0; i <index_->dim_; ++i) {
                if ((num < index_->RAND_DIM)||(v[i] > v[topind[num-1]])) {
                    // Put this element at end of topind.
                    if (num < index_->RAND_DIM) {
                        topind[num++] = i;            // Add to list.
                    }
                    else {
                        topind[num-1] = i;         // Replace last element.
                    }
                    // Bubble end value down to right location by repeated swapping. sort the varience in decrease order
                    int j = num - 1;
                    while (j > 0  &&  v[topind[j]] > v[topind[j-1]]) {
                        std::swap(topind[j], topind[j-1]);
                        --j;
                    }
                }
            }
            // Select a random integer in range [0,num-1], and return that index.
            int rnd = rng()%num;
            return (int)topind[rnd];
        }

        void DFSbuild(Node* node, std::mt19937& rng, unsigned* indices, unsigned count, unsigned offset){
            //omp_set_lock(&rootlock);
            //std::cout<<node->treeid<<":"<<offset<<":"<<count<<std::endl;
            //omp_unset_lock(&rootlock);

            if(count <= index_->TNS){
                node->DivDim = -1;
                node->Lchild = NULL;
                node->Rchild = NULL;
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
                Node* nodeL = new Node(); Node* nodeR = new Node();
                node->Lchild = nodeL;
                nodeL->treeid = node->treeid;
                DFSbuild(nodeL, rng, indices, idx, offset);
                node->Rchild = nodeR;
                nodeR->treeid = node->treeid;
                DFSbuild(nodeR, rng, indices+idx, count-idx, offset+idx);
            }
        }

        void DFStest(unsigned level, unsigned dim, Node* node){
            if(node->Lchild !=NULL){
                DFStest(++level, node->DivDim, node->Lchild);
                //if(level > 15)
                std::cout<<"dim: "<<node->DivDim<<"--cutval: "<<node->DivVal<<"--S: "<<node->StartIdx<<"--E: "<<node->EndIdx<<" TREE: "<<node->treeid<<std::endl;
                if(node->Lchild->Lchild ==NULL){
                    std::vector<unsigned>& tmp = index_->LeafLists[node->treeid];
                    for(unsigned i = node->Rchild->StartIdx; i < node->Rchild->EndIdx; i++){
                        const float* tmpfea =index_->data_ + tmp[i] *index_->dim_+ node->DivDim;
                        std::cout<< *tmpfea <<" ";
                    }
                    std::cout<<std::endl;
                }
            }
            else if(node->Rchild !=NULL){
                DFStest(++level, node->DivDim, node->Rchild);
            }
            else{
                std::cout<<"dim: "<<dim<<std::endl;
                std::vector<unsigned>& tmp = index_->LeafLists[node->treeid];
                for(unsigned i = node->StartIdx; i < node->EndIdx; i++){
                    const float* tmpfea =index_->data_ + tmp[i] *index_->dim_+ dim;
                    std::cout<< *tmpfea <<" ";
                }
                std::cout<<std::endl;
            }
        }

        Node* SearchToLeaf(Node* node, size_t id){
            if(node->Lchild != NULL && node->Rchild !=NULL){
                const float* v = index_->data_ + id * index_->dim_;
                if(v[node->DivDim] < node->DivVal)
                    return SearchToLeaf(node->Lchild, id);
                else
                    return SearchToLeaf(node->Rchild, id);
            }
            else
                return node;
        }

        void mergeSubGraphs(size_t treeid, Node* node){

            if(node->Lchild != NULL && node->Rchild != NULL){
                mergeSubGraphs(treeid, node->Lchild);
                mergeSubGraphs(treeid, node->Rchild);

                size_t numL = node->Lchild->EndIdx - node->Lchild->StartIdx;
                size_t numR = node->Rchild->EndIdx - node->Rchild->StartIdx;
                size_t start,end;
                Node * root;
                if(numL < numR){
                    root = node->Rchild;
                    start = node->Lchild->StartIdx;
                    end = node->Lchild->EndIdx;
                }else{
                    root = node->Lchild;
                    start = node->Rchild->StartIdx;
                    end = node->Rchild->EndIdx;
                }

                for(;start < end; start++){

                    size_t feature_id = index_->LeafLists[treeid][start];

                    Node* leaf = SearchToLeaf(root, feature_id);
                    for(size_t i = leaf->StartIdx; i < leaf->EndIdx; i++){
                        size_t tmpfea = index_->LeafLists[treeid][i];
                        float dist = index_->distance_->compare(index_->data_ + tmpfea * index_->dim_, index_->data_ + feature_id * index_->dim_, index_->dim_);

                        {LockGuard guard(index_->graph_[tmpfea].lock);
                            if(index_->knn_graph[tmpfea].size() < index_->K || dist < index_->knn_graph[tmpfea].begin()->distance){
                                Candidate c1(feature_id, dist);
                                index_->knn_graph[tmpfea].insert(c1);
                                if(index_->knn_graph[tmpfea].size() > index_->K)
                                    index_->knn_graph[tmpfea].erase(index_->knn_graph[tmpfea].begin());
                            }
                        }

                        {LockGuard guard(index_->graph_[feature_id].lock);
                            if(index_->knn_graph[feature_id].size() < index_->K || dist < index_->knn_graph[feature_id].begin()->distance){
                                Candidate c1(tmpfea, dist);
                                index_->knn_graph[feature_id].insert(c1);
                                if(index_->knn_graph[feature_id].size() > index_->K)
                                    index_->knn_graph[feature_id].erase(index_->knn_graph[feature_id].begin());

                            }
                        }
                    }
                }
            }
        }

        void getMergeLevelNodeList(Node* node, size_t treeid, int deepth){
            if(node->Lchild != NULL && node->Rchild != NULL && deepth < index_->ml){
                deepth++;
                getMergeLevelNodeList(node->Lchild, treeid, deepth);
                getMergeLevelNodeList(node->Rchild, treeid, deepth);
            }else if(deepth == index_->ml){
                index_->mlNodeList.push_back(std::make_pair(node,treeid));
            }else{
                index_->error_flag = true;
                if(deepth < index_->max_deepth)index_->max_deepth = deepth;
            }
        }

        void InitInner() {
            //initial
            unsigned N = index_->n_;
            unsigned seed = 1998;

            index_->graph_.resize(N);
            index_->knn_graph.resize(N);

            //build tree
            unsigned TreeNum = param_.Get<unsigned>("nTrees");
            unsigned TreeNumBuild = param_.Get<unsigned>("nTrees");
            index_->ml = param_.Get<unsigned>("mLevel");
            index_->K = param_.Get<unsigned>("K");

            std::vector<int> indices(N);
            index_->LeafLists.resize(TreeNum);
            std::vector<Node *> ActiveSet;
            std::vector<Node *> NewSet;
            for (unsigned i = 0; i < (unsigned) TreeNum; i++) {
                Node *node = new Node;
                node->DivDim = -1;
                node->Lchild = NULL;
                node->Rchild = NULL;
                node->StartIdx = 0;
                node->EndIdx = N;
                node->treeid = i;
                index_->tree_roots_.push_back(node);
                ActiveSet.push_back(node);
            }

#pragma omp parallel for
            for (unsigned i = 0; i < N; i++)indices[i] = i;
#pragma omp parallel for
            for (unsigned i = 0; i < (unsigned) TreeNum; i++) {
                std::vector<unsigned> &myids = index_->LeafLists[i];
                myids.resize(N);
                std::copy(indices.begin(), indices.end(), myids.begin());
                std::random_shuffle(myids.begin(), myids.end());
            }
            omp_init_lock(&index_->tree_roots_);
            while (!ActiveSet.empty() && ActiveSet.size() < 1100) {
#pragma omp parallel for
                for (unsigned i = 0; i < ActiveSet.size(); i++) {
                    Node *node = ActiveSet[i];
                    unsigned mid;
                    unsigned cutdim;
                    float cutval;
                    std::mt19937 rng(seed ^ omp_get_thread_num());
                    std::vector<unsigned> &myids = index_->LeafLists[node->treeid];

                    meanSplit(rng, &myids[0] + node->StartIdx, node->EndIdx - node->StartIdx, mid, cutdim, cutval);

                    node->DivDim = cutdim;
                    node->DivVal = cutval;
                    //node->StartIdx = offset;
                    //node->EndIdx = offset + count;
                    Node *nodeL = new Node();
                    Node *nodeR = new Node();
                    nodeR->treeid = nodeL->treeid = node->treeid;
                    nodeL->StartIdx = node->StartIdx;
                    nodeL->EndIdx = node->StartIdx + mid;
                    nodeR->StartIdx = nodeL->EndIdx;
                    nodeR->EndIdx = node->EndIdx;
                    node->Lchild = nodeL;
                    node->Rchild = nodeR;
                    omp_set_lock(&index_->rootlock);
                    if (mid > index_->K)NewSet.push_back(nodeL);
                    if (nodeR->EndIdx - nodeR->StartIdx > index_->K)NewSet.push_back(nodeR);
                    omp_unset_lock(&index_->rootlock);
                }
                ActiveSet.resize(NewSet.size());
                std::copy(NewSet.begin(), NewSet.end(), ActiveSet.begin());
                NewSet.clear();
            }

#pragma omp parallel for
            for (unsigned i = 0; i < ActiveSet.size(); i++) {
                Node *node = ActiveSet[i];
                //omp_set_lock(&index_->rootlock);
                //std::cout<<i<<":"<<node->EndIdx-node->StartIdx<<std::endl;
                //omp_unset_lock(&index_->rootlock);
                std::mt19937 rng(seed ^ omp_get_thread_num());
                std::vector<unsigned> &myids = index_->LeafLists[node->treeid];
                DFSbuild(node, rng, &myids[0] + node->StartIdx, node->EndIdx - node->StartIdx, node->StartIdx);
            }
            //DFStest(0,0,index_->tree_roots_[0]);
            std::cout << "build tree completed" << std::endl;

            for (size_t i = 0; i < (unsigned) TreeNumBuild; i++) {
                getMergeLevelNodeList(index_->tree_roots_[i], i, 0);
            }

            std::cout << "merge node list size: " << index_->mlNodeList.size() << std::endl;
            if (index_->error_flag) {
                std::cout << "merge level deeper than tree, max merge deepth is " << index_->max_deepth - 1 << std::endl;
            }

#pragma omp parallel for
            for (size_t i = 0; i < index_->mlNodeList.size(); i++) {
                mergeSubGraphs(index_->mlNodeList[i].second, index_->mlNodeList[i].first);
            }

            std::cout << "merge tree completed" << std::endl;

            index_->final_graph_.reserve(index_->n_);
            std::mt19937 rng(seed ^ omp_get_thread_num());
            std::set<unsigned> result;
            for (unsigned i = 0; i < index_->n_; i++) {
                std::vector<unsigned> tmp;
                typename Index::CandidateHeap::reverse_iterator it = index_->knn_graph[i].rbegin();
                for (; it != index_->knn_graph[i].rend(); it++) {
                    tmp.push_back(it->row_id);
                }
                if (tmp.size() < index_->K) {
                    //std::cout << "node "<< i << " only has "<< tmp.size() <<" neighbors!" << std::endl;
                    result.clear();
                    size_t vlen = tmp.size();
                    for (size_t j = 0; j < vlen; j++) {
                        result.insert(tmp[j]);
                    }
                    while (result.size() < index_->K) {
                        unsigned id = rng() % N;
                        result.insert(id);
                    }
                    tmp.clear();
                    std::set<unsigned>::iterator it;
                    for (it = result.begin(); it != result.end(); it++) {
                        tmp.push_back(*it);
                    }
                    //std::copy(result.begin(),result.end(),tmp.begin());
                }
                tmp.reserve(index_->K);
                index_->final_graph_.push_back(tmp);
            }

            std::vector<nhood>().swap(index_->graph_);

            assert(index_->final_graph_.size() == index_->n_);

            const unsigned L = param_.Get<unsigned>("L");
            const unsigned S = param_.Get<unsigned>("S");

            index_->graph_.reserve(index_->n_);
            std::mt19937 rng2(rand());
            for (unsigned i = 0; i < index_->n_; i++) {
                index_->graph_.push_back(nhood(L, S, rng2, (unsigned) index_->n_));
            }
#pragma omp parallel for
            for (unsigned i = 0; i < index_->n_; i++) {
                auto& ids = index_->final_graph_[i];
                std::sort(ids.begin(), ids.end());

                size_t K = ids.size();

                for (unsigned j = 0; j < K; j++) {
                    unsigned id = ids[j];
                    if (id == i || (j>0 &&id == ids[j-1]))continue;
                    float dist = index_->distance_->compare(index_->data_ + i * index_->dim_, index_->data_ + id * index_->dim_, (unsigned) index_->dim_);
                    index_->graph_[i].pool.push_back(Neighbor(id, dist, true));
                }
                std::make_heap(index_->graph_[i].pool.begin(), index_->graph_[i].pool.end());
                index_->graph_[i].pool.reserve(L);
                std::vector<unsigned>().swap(ids);
            }
            Index::CompactGraph().swap(index_->final_graph_);
        }
    };

}

#endif //WEAVESS_INDEX_COMPONENT_INIT_H

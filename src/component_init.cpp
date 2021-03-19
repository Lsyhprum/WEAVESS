//
// Created by MurphySL on 2020/10/23.
//

#include "weavess/component.h"
#include <functional>


namespace weavess {

    void ComponentInitRandom::InitInner() {
        SetConfigs();

        unsigned range = index->getInitEdgesNum();

        index->getFinalGraph().resize(index->getBaseLen());

        std::mt19937 rng(rand());

#ifdef PARALLEL
#pragma omp parallel for
#endif
        for (unsigned i = 0; i < index->getBaseLen(); i++) {
            index->getFinalGraph()[i].reserve(range);

            std::vector<unsigned> tmp(range);

            GenRandom(rng, tmp.data(), range);

            for (unsigned j = 0; j < range; j++) {
                unsigned id = tmp[j];

                if (id == i) {
                    continue;
                }

                float dist = index->getDist()->compare(index->getBaseData() + i * index->getBaseDim(),
                                                       index->getBaseData() + id * index->getBaseDim(),
                                                       (unsigned) index->getBaseDim());

                index->getFinalGraph()[i].emplace_back(id, dist);
            }
            std::sort(index->getFinalGraph()[i].begin(), index->getFinalGraph()[i].end());
        }
    }

    void ComponentInitRandom::SetConfigs() {
        index->setInitEdgesNum(index->getParam().get<unsigned>("S"));
    }

    void ComponentInitRandom::GenRandom(std::mt19937 &rng, unsigned *addr, unsigned size) {
        unsigned N = index->getBaseLen();

        for (unsigned i = 0; i < size; ++i) {
            addr[i] = rng() % (N - size);
        }
        std::sort(addr, addr + size);

        for (unsigned i = 1; i < size; ++i) {
            if (addr[i] <= addr[i - 1]) {
                addr[i] = addr[i - 1] + 1;
            }
        }
        unsigned off = rng() % N;
        for (unsigned i = 0; i < size; ++i) {
            addr[i] = (addr[i] + off) % N;
        }
    }


    void ComponentInitKNNG::InitInner() {
        SetConfigs();

        unsigned range = index->getInitEdgesNum();

        index->getFinalGraph().resize(index->getBaseLen());

#ifdef PARALLEL
#pragma omp parallel for
#endif
        for (unsigned i = 0; i < index->getBaseLen(); i ++) {
            index->getFinalGraph()[i].resize(range);

            std::vector<Index::SimpleNeighbor> tmp;
            tmp.resize(index->getBaseLen() - 1);

            int pos = 0;
            for (unsigned j = 0; j < index->getBaseLen(); j ++) {
                if (i == j) continue;

                float dist = index->getDist()->compare(index->getBaseData() + index->getBaseDim() * i,
                                                       index->getBaseData() + index->getBaseDim() * j,
                                                       index->getBaseDim());
                tmp[pos ++] = Index::SimpleNeighbor(j, dist);
            }

            std::sort(tmp.begin(), tmp.end());

            for(int k = 0; k < index->getInitEdgesNum(); k ++) {
                index->getFinalGraph()[i][k] = tmp[k];
            }

            std::vector<Index::SimpleNeighbor>().swap(tmp);
        }
    }

    void ComponentInitKNNG::SetConfigs() {
        index->setInitEdgesNum(index->getParam().get<unsigned>("S"));
    }



    void ComponentInitKDTree::InitInner() {
        SetConfigs();

        unsigned seed = 1998;
        const auto TreeNum = index->nTrees;
        const auto TreeNumBuild = index->nTrees;

        index->graph_.resize(index->getBaseLen());
        index->knn_graph.resize(index->getBaseLen());

        std::vector<int> indices(index->getBaseLen());
        index->LeafLists.resize(TreeNum);
        std::vector<Index::EFANNA::Node *> ActiveSet;
        std::vector<Index::EFANNA::Node *> NewSet;
        for (unsigned i = 0; i < (unsigned) TreeNum; i++) {
            auto *node = new Index::EFANNA::Node;
            node->DivDim = -1;
            node->Lchild = nullptr;
            node->Rchild = nullptr;
            node->StartIdx = 0;
            node->EndIdx = index->getBaseLen();
            node->treeid = i;
            index->tree_roots_.push_back(node);
            ActiveSet.push_back(node);
        }

#pragma omp parallel for
        for (unsigned i = 0; i < index->getBaseLen(); i++)indices[i] = i;
#pragma omp parallel for
        for (unsigned i = 0; i < (unsigned) TreeNum; i++) {
            std::vector<unsigned> &myids = index->LeafLists[i];
            myids.resize(index->getBaseLen());
            std::copy(indices.begin(), indices.end(), myids.begin());
            std::random_shuffle(myids.begin(), myids.end());
        }
        omp_init_lock(&index->rootlock);
        while (!ActiveSet.empty() && ActiveSet.size() < 1100) {
#pragma omp parallel for
            for (unsigned i = 0; i < ActiveSet.size(); i++) {
                Index::EFANNA::Node *node = ActiveSet[i];
                unsigned mid;
                unsigned cutdim;
                float cutval;
                std::mt19937 rng(seed ^ omp_get_thread_num());
                std::vector<unsigned> &myids = index->LeafLists[node->treeid];

                meanSplit(rng, &myids[0] + node->StartIdx, node->EndIdx - node->StartIdx, mid, cutdim, cutval);

                node->DivDim = cutdim;
                node->DivVal = cutval;
                //node->StartIdx = offset;
                //node->EndIdx = offset + count;
                auto *nodeL = new Index::EFANNA::Node();
                auto *nodeR = new Index::EFANNA::Node();
                nodeR->treeid = nodeL->treeid = node->treeid;
                nodeL->StartIdx = node->StartIdx;
                nodeL->EndIdx = node->StartIdx + mid;
                nodeR->StartIdx = nodeL->EndIdx;
                nodeR->EndIdx = node->EndIdx;
                node->Lchild = nodeL;
                node->Rchild = nodeR;
                omp_set_lock(&index->rootlock);
                if (mid > index->getInitEdgesNum())NewSet.push_back(nodeL);
                if (nodeR->EndIdx - nodeR->StartIdx > index->getInitEdgesNum())NewSet.push_back(nodeR);
                omp_unset_lock(&index->rootlock);
            }
            ActiveSet.resize(NewSet.size());
            std::copy(NewSet.begin(), NewSet.end(), ActiveSet.begin());
            NewSet.clear();
        }

#pragma omp parallel for
        for (unsigned i = 0; i < ActiveSet.size(); i++) {
            Index::EFANNA::Node *node = ActiveSet[i];
            //omp_set_lock(&rootlock);
            //std::cout<<i<<":"<<node->EndIdx-node->StartIdx<<std::endl;
            //omp_unset_lock(&rootlock);
            std::mt19937 rng(seed ^ omp_get_thread_num());
            std::vector<unsigned> &myids = index->LeafLists[node->treeid];
            DFSbuild(node, rng, &myids[0] + node->StartIdx, node->EndIdx - node->StartIdx, node->StartIdx);
        }
        //DFStest(0,0,tree_roots_[0]);
        std::cout << "build tree completed" << std::endl;

        for (size_t i = 0; i < (unsigned) TreeNumBuild; i++) {
            getMergeLevelNodeList(index->tree_roots_[i], i, 0);
        }

        std::cout << "merge node list size: " << index->mlNodeList.size() << std::endl;
        if (index->error_flag) {
            std::cout << "merge level deeper than tree, max merge deepth is " << index->max_deepth - 1 << std::endl;
        }

#pragma omp parallel for
        for (size_t i = 0; i < index->mlNodeList.size(); i++) {
            mergeSubGraphs(index->mlNodeList[i].second, index->mlNodeList[i].first);
        }

        std::cout << "merge tree completed" << std::endl;

        index->getFinalGraph().resize(index->getBaseLen());
        std::mt19937 rng(seed ^ omp_get_thread_num());
        std::set<Index::SimpleNeighbor> result;
        for (unsigned i = 0; i < index->getBaseLen(); i++) {
            std::vector<Index::SimpleNeighbor> tmp;
            typename Index::CandidateHeap::reverse_iterator it = index->knn_graph[i].rbegin();
            for (; it != index->knn_graph[i].rend(); it++) {
                float dist = index->getDist()->compare(index->getBaseData() + index->getBaseDim() * i,
                                                       index->getBaseData() + index->getBaseDim() * it->row_id,
                                                       index->getBaseDim());
                tmp.push_back(Index::SimpleNeighbor(it->row_id, dist));
            }
            if (tmp.size() < index->getInitEdgesNum()) {
                //std::cout << "node "<< i << " only has "<< tmp.size() <<" neighbors!" << std::endl;
                result.clear();
                size_t vlen = tmp.size();
                for (size_t j = 0; j < vlen; j++) {
                    result.insert(tmp[j]);
                }
                while (result.size() < index->getInitEdgesNum()) {
                    unsigned id = rng() % index->getBaseLen();
                    float dist = index->getDist()->compare(index->getBaseData() + index->getBaseDim() * i,
                                                           index->getBaseData() + index->getBaseDim() * id,
                                                           index->getBaseDim());
                    result.insert(Index::SimpleNeighbor(id, dist));
                }
                tmp.clear();
                std::set<Index::SimpleNeighbor>::iterator it;
                for (it = result.begin(); it != result.end(); it++) {
                    tmp.push_back(*it);
                }
                //std::copy(result.begin(),result.end(),tmp.begin());
            }
            tmp.reserve(index->getInitEdgesNum());
            index->getFinalGraph()[i] = tmp;
        }
        std::vector<Index::nhood>().swap(index->graph_);
    }

    void ComponentInitKDTree::SetConfigs() {
        index->mLevel = index->getParam().get<unsigned>("mLevel");
        index->nTrees = index->getParam().get<unsigned>("nTrees");

        index->setInitEdgesNum(index->getParam().get<unsigned>("S"));
    }

    void ComponentInitKDTree::meanSplit(std::mt19937 &rng, unsigned *indices, unsigned count, unsigned &index1,
                                     unsigned &cutdim, float &cutval) {
        float *mean_ = new float[index->getBaseDim()];
        float *var_ = new float[index->getBaseDim()];
        memset(mean_, 0, index->getBaseDim() * sizeof(float));
        memset(var_, 0, index->getBaseDim() * sizeof(float));

        /* Compute mean values.  Only the first SAMPLE_NUM values need to be
          sampled to get a good estimate.
         */
        unsigned cnt = std::min((unsigned) index->SAMPLE_NUM + 1, count);
        for (unsigned j = 0; j < cnt; ++j) {
            const float *v = index->getBaseData() + indices[j] * index->getBaseDim();
            for (size_t k = 0; k < index->getBaseDim(); ++k) {
                mean_[k] += v[k];
            }
        }
        float div_factor = float(1) / cnt;
        for (size_t k = 0; k < index->getBaseDim(); ++k) {
            mean_[k] *= div_factor;
        }

        /* Compute variances (no need to divide by count). */

        for (unsigned j = 0; j < cnt; ++j) {
            const float *v = index->getBaseData() + indices[j] * index->getBaseDim();
            for (size_t k = 0; k < index->getBaseDim(); ++k) {
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
        if (lim1 > count / 2) index1 = lim1;
        else if (lim2 < count / 2) index1 = lim2;
        else index1 = count / 2;

        /* If either list is empty, it means that all remaining features
         * are identical. Split in the middle to maintain a balanced tree.
         */
        if ((lim1 == count) || (lim2 == 0)) index1 = count / 2;
        delete[] mean_;
        delete[] var_;
    }

    int ComponentInitKDTree::selectDivision(std::mt19937 &rng, float *v) {
        int num = 0;
        size_t topind[index->RAND_DIM];

        //Create a list of the indices of the top index->RAND_DIM values.
        for (size_t i = 0; i < index->getBaseDim(); ++i) {
            if ((num < index->RAND_DIM) || (v[i] > v[topind[num - 1]])) {
                // Put this element at end of topind.
                if (num < index->RAND_DIM) {
                    topind[num++] = i;            // Add to list.
                } else {
                    topind[num - 1] = i;         // Replace last element.
                }
                // Bubble end value down to right location by repeated swapping. sort the varience in decrease order
                int j = num - 1;
                while (j > 0 && v[topind[j]] > v[topind[j - 1]]) {
                    std::swap(topind[j], topind[j - 1]);
                    --j;
                }
            }
        }
        // Select a random integer in range [0,num-1], and return that index.
        int rnd = rng() % num;
        return (int) topind[rnd];
    }

    void ComponentInitKDTree::planeSplit(unsigned *indices, unsigned count, unsigned cutdim, float cutval, unsigned &lim1,
                                      unsigned &lim2) {
        /* Move vector indices for left subtree to front of list. */
        int left = 0;
        int right = count - 1;
        for (;;) {
            const float *vl = index->getBaseData() + indices[left] * index->getBaseDim();
            const float *vr = index->getBaseData() + indices[right] * index->getBaseDim();
            while (left <= right && vl[cutdim] < cutval) {
                ++left;
                vl = index->getBaseData() + indices[left] * index->getBaseDim();
            }
            while (left <= right && vr[cutdim] >= cutval) {
                --right;
                vr = index->getBaseData() + indices[right] * index->getBaseDim();
            }
            if (left > right) break;
            std::swap(indices[left], indices[right]);
            ++left;
            --right;
        }
        lim1 = left;//lim1 is the id of the leftmost point <= cutval
        right = count - 1;
        for (;;) {
            const float *vl = index->getBaseData() + indices[left] * index->getBaseDim();
            const float *vr = index->getBaseData() + indices[right] * index->getBaseDim();
            while (left <= right && vl[cutdim] <= cutval) {
                ++left;
                vl = index->getBaseData() + indices[left] * index->getBaseDim();
            }
            while (left <= right && vr[cutdim] > cutval) {
                --right;
                vr = index->getBaseData() + indices[right] * index->getBaseDim();
            }
            if (left > right) break;
            std::swap(indices[left], indices[right]);
            ++left;
            --right;
        }
        lim2 = left;//lim2 is the id of the leftmost point >cutval
    }

    void ComponentInitKDTree::DFSbuild(Index::EFANNA::Node *node, std::mt19937 &rng, unsigned *indices, unsigned count,
                                    unsigned offset) {
        //omp_set_lock(&rootlock);
        //std::cout<<node->treeid<<":"<<offset<<":"<<count<<std::endl;
        //omp_unset_lock(&rootlock);

        if (count <= index->TNS) {
            node->DivDim = -1;
            node->Lchild = nullptr;
            node->Rchild = nullptr;
            node->StartIdx = offset;
            node->EndIdx = offset + count;
            //add points

        } else {
            unsigned idx;
            unsigned cutdim;
            float cutval;
            meanSplit(rng, indices, count, idx, cutdim, cutval);
            node->DivDim = cutdim;
            node->DivVal = cutval;
            node->StartIdx = offset;
            node->EndIdx = offset + count;
            auto *nodeL = new Index::EFANNA::Node();
            auto *nodeR = new Index::EFANNA::Node();
            node->Lchild = nodeL;
            nodeL->treeid = node->treeid;
            DFSbuild(nodeL, rng, indices, idx, offset);
            node->Rchild = nodeR;
            nodeR->treeid = node->treeid;
            DFSbuild(nodeR, rng, indices + idx, count - idx, offset + idx);
        }
    }

    void ComponentInitKDTree::getMergeLevelNodeList(Index::EFANNA::Node *node, size_t treeid, unsigned deepth) {
        auto ml = index->getParam().get<unsigned>("mLevel");
        if (node->Lchild != nullptr && node->Rchild != nullptr && deepth < ml) {
            deepth++;
            getMergeLevelNodeList(node->Lchild, treeid, deepth);
            getMergeLevelNodeList(node->Rchild, treeid, deepth);
        } else if (deepth == ml) {
            index->mlNodeList.emplace_back(node, treeid);
        } else {
            index->error_flag = true;
            if (deepth < index->max_deepth)index->max_deepth = deepth;
        }
    }

    void ComponentInitKDTree::mergeSubGraphs(size_t treeid, Index::EFANNA::Node *node) {

        if (node->Lchild != nullptr && node->Rchild != nullptr) {
            mergeSubGraphs(treeid, node->Lchild);
            mergeSubGraphs(treeid, node->Rchild);

            size_t numL = node->Lchild->EndIdx - node->Lchild->StartIdx;
            size_t numR = node->Rchild->EndIdx - node->Rchild->StartIdx;
            size_t start, end;
            Index::EFANNA::Node *root;
            if (numL < numR) {
                root = node->Rchild;
                start = node->Lchild->StartIdx;
                end = node->Lchild->EndIdx;
            } else {
                root = node->Lchild;
                start = node->Rchild->StartIdx;
                end = node->Rchild->EndIdx;
            }

            //std::cout << start << " " << end << std::endl;

            for (; start < end; start++) {

                size_t feature_id = index->LeafLists[treeid][start];

                Index::EFANNA::Node *leaf = SearchToLeaf(root, feature_id);
                for (size_t i = leaf->StartIdx; i < leaf->EndIdx; i++) {
                    size_t tmpfea = index->LeafLists[treeid][i];
                    float dist = index->getDist()->compare(index->getBaseData() + tmpfea * index->getBaseDim(),
                                                           index->getBaseData() + feature_id * index->getBaseDim(),
                                                           index->getBaseDim());

                    {
                        Index::LockGuard guard(index->graph_[tmpfea].lock);
                        if (index->knn_graph[tmpfea].size() < index->getInitEdgesNum() || dist < index->knn_graph[tmpfea].begin()->distance) {
                            Index::Candidate c1(feature_id, dist);
                            index->knn_graph[tmpfea].insert(c1);
                            if (index->knn_graph[tmpfea].size() > index->getInitEdgesNum())
                                index->knn_graph[tmpfea].erase(index->knn_graph[tmpfea].begin());
                        }
                    }

                    {
                        Index::LockGuard guard(index->graph_[feature_id].lock);
                        if (index->knn_graph[feature_id].size() < index->getInitEdgesNum() ||
                            dist < index->knn_graph[feature_id].begin()->distance) {
                            Index::Candidate c1(tmpfea, dist);
                            index->knn_graph[feature_id].insert(c1);
                            if (index->knn_graph[feature_id].size() > index->getInitEdgesNum())
                                index->knn_graph[feature_id].erase(index->knn_graph[feature_id].begin());

                        }
                    }
                }
            }
        }
    }

    Index::EFANNA::Node *ComponentInitKDTree::SearchToLeaf(Index::EFANNA::Node *node, size_t id) {
        if (node->Lchild != nullptr && node->Rchild != nullptr) {
            const float *v = index->getBaseData() + id * index->getBaseDim();
            if (v[node->DivDim] < node->DivVal)
                return SearchToLeaf(node->Lchild, id);
            else
                return SearchToLeaf(node->Rchild, id);
        } else
            return node;
    }



    void ComponentInitKDT::InitInner() {
        omp_init_lock(&index->rootlock);
        SetConfigs();

        unsigned seed = 1998;

        index->graph_.resize(index->getBaseLen());
        index->knn_graph.resize(index->getBaseLen());

        const auto TreeNum = index->getParam().get<unsigned>("nTrees");
        const auto TreeNumBuild = index->getParam().get<unsigned>("nTrees");
        const auto K = index->getParam().get<unsigned>("K");

        std::vector<int> indices(index->getBaseLen());
        index->LeafLists.resize(TreeNum);
        std::vector<Index::EFANNA::Node *> ActiveSet;
        std::vector<Index::EFANNA::Node *> NewSet;
        for (unsigned i = 0; i < (unsigned) TreeNum; i++) {
            auto *node = new Index::EFANNA::Node;
            node->DivDim = -1;
            node->Lchild = nullptr;
            node->Rchild = nullptr;
            node->StartIdx = 0;
            node->EndIdx = index->getBaseLen();
            node->treeid = i;
            index->tree_roots_.push_back(node);
            ActiveSet.push_back(node);
        }

#pragma omp parallel for
        for (unsigned i = 0; i < index->getBaseLen(); i++)indices[i] = i;
#pragma omp parallel for
        for (unsigned i = 0; i < (unsigned) TreeNum; i++) {
            std::vector<unsigned> &myids = index->LeafLists[i];
            myids.resize(index->getBaseLen());
            std::copy(indices.begin(), indices.end(), myids.begin());
            std::random_shuffle(myids.begin(), myids.end());
        }
        // omp_init_lock(&index->rootlock);
        while (!ActiveSet.empty() && ActiveSet.size() < 1100) {
#pragma omp parallel for
            for (unsigned i = 0; i < ActiveSet.size(); i++) {
                Index::EFANNA::Node *node = ActiveSet[i];
                unsigned mid;
                unsigned cutdim;
                float cutval;
                std::mt19937 rng(seed ^ omp_get_thread_num());
                std::vector<unsigned> &myids = index->LeafLists[node->treeid];

                meanSplit(rng, &myids[0] + node->StartIdx, node->EndIdx - node->StartIdx, mid, cutdim, cutval);

                node->DivDim = cutdim;
                node->DivVal = cutval;
                //node->StartIdx = offset;
                //node->EndIdx = offset + count;
                auto *nodeL = new Index::EFANNA::Node();
                auto *nodeR = new Index::EFANNA::Node();
                nodeR->treeid = nodeL->treeid = node->treeid;
                nodeL->StartIdx = node->StartIdx;
                nodeL->EndIdx = node->StartIdx + mid;
                nodeR->StartIdx = nodeL->EndIdx;
                nodeR->EndIdx = node->EndIdx;
                node->Lchild = nodeL;
                node->Rchild = nodeR;
                omp_set_lock(&index->rootlock);
                if (mid > K)NewSet.push_back(nodeL);
                if (nodeR->EndIdx - nodeR->StartIdx > K)NewSet.push_back(nodeR);
                omp_unset_lock(&index->rootlock);
            }
            ActiveSet.resize(NewSet.size());
            std::copy(NewSet.begin(), NewSet.end(), ActiveSet.begin());
            NewSet.clear();
        }

#pragma omp parallel for
        for (unsigned i = 0; i < ActiveSet.size(); i++) {
            Index::EFANNA::Node *node = ActiveSet[i];
            //omp_set_lock(&rootlock);
            //std::cout<<i<<":"<<node->EndIdx-node->StartIdx<<std::endl;
            //omp_unset_lock(&rootlock);
            std::mt19937 rng(seed ^ omp_get_thread_num());
            std::vector<unsigned> &myids = index->LeafLists[node->treeid];
            DFSbuild(node, rng, &myids[0] + node->StartIdx, node->EndIdx - node->StartIdx, node->StartIdx);
        }
        //DFStest(0,0,tree_roots_[0]);
        std::cout << "build tree completed" << std::endl;

        for (size_t i = 0; i < (unsigned) TreeNumBuild; i++) {
            getMergeLevelNodeList(index->tree_roots_[i], i, 0);
        }

        std::cout << "merge node list size: " << index->mlNodeList.size() << std::endl;
        if (index->error_flag) {
            std::cout << "merge level deeper than tree, max merge deepth is " << index->max_deepth - 1 << std::endl;
        }

#pragma omp parallel for
        for (size_t i = 0; i < index->mlNodeList.size(); i++) {
            mergeSubGraphs(index->mlNodeList[i].second, index->mlNodeList[i].first);
        }

        std::cout << "merge tree completed" << std::endl;

        index->getFinalGraph().resize(index->getBaseLen());
        std::mt19937 rng(seed ^ omp_get_thread_num());
        std::set<Index::SimpleNeighbor> result;
        for (unsigned i = 0; i < index->getBaseLen(); i++) {
            std::vector<std::vector<unsigned>> level_tmp;
            std::vector<Index::SimpleNeighbor> tmp;
            typename Index::CandidateHeap::reverse_iterator it = index->knn_graph[i].rbegin();
            for (; it != index->knn_graph[i].rend(); it++) {
                float dist = index->getDist()->compare(index->getBaseData() + index->getBaseDim() * i,
                                                       index->getBaseData() + index->getBaseDim() * it->row_id,
                                                       index->getBaseDim());
                tmp.push_back(Index::SimpleNeighbor(it->row_id, dist));
            }
            if (tmp.size() < K) {
                //std::cout << "node "<< i << " only has "<< tmp.size() <<" neighbors!" << std::endl;
                result.clear();
                size_t vlen = tmp.size();
                for (size_t j = 0; j < vlen; j++) {
                    result.insert(tmp[j]);
                }
                while (result.size() < K) {
                    unsigned id = rng() % index->getBaseLen();
                    float dist = index->getDist()->compare(index->getBaseData() + index->getBaseDim() * i,
                                                           index->getBaseData() + index->getBaseDim() * id,
                                                           index->getBaseDim());
                    result.insert(Index::SimpleNeighbor(id, dist));
                }
                tmp.clear();
                std::set<Index::SimpleNeighbor>::iterator it;
                for (it = result.begin(); it != result.end(); it++) {
                    tmp.push_back(*it);
                }
                //std::copy(result.begin(),result.end(),tmp.begin());
            }
            tmp.reserve(K);
            index->getFinalGraph()[i] = tmp;
        }
        std::vector<Index::nhood>().swap(index->graph_);
    }

    void ComponentInitKDT::SetConfigs() {
        index->mLevel = index->getParam().get<unsigned>("mLevel");
        index->nTrees = index->getParam().get<unsigned>("nTrees");
    }

    void ComponentInitKDT::meanSplit(std::mt19937 &rng, unsigned *indices, unsigned count, unsigned &index1,
                                     unsigned &cutdim, float &cutval) {
        float *mean_ = new float[index->getBaseDim()];
        float *var_ = new float[index->getBaseDim()];
        memset(mean_, 0, index->getBaseDim() * sizeof(float));
        memset(var_, 0, index->getBaseDim() * sizeof(float));

        /* Compute mean values.  Only the first SAMPLE_NUM values need to be
          sampled to get a good estimate.
         */
        unsigned cnt = std::min((unsigned) index->SAMPLE_NUM + 1, count);
        for (unsigned j = 0; j < cnt; ++j) {
            const float *v = index->getBaseData() + indices[j] * index->getBaseDim();
            for (size_t k = 0; k < index->getBaseDim(); ++k) {
                mean_[k] += v[k];
            }
        }
        float div_factor = float(1) / cnt;
        for (size_t k = 0; k < index->getBaseDim(); ++k) {
            mean_[k] *= div_factor;
        }

        /* Compute variances (no need to divide by count). */

        for (unsigned j = 0; j < cnt; ++j) {
            const float *v = index->getBaseData() + indices[j] * index->getBaseDim();
            for (size_t k = 0; k < index->getBaseDim(); ++k) {
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
        if (lim1 > count / 2) index1 = lim1;
        else if (lim2 < count / 2) index1 = lim2;
        else index1 = count / 2;

        /* If either list is empty, it means that all remaining features
         * are identical. Split in the middle to maintain a balanced tree.
         */
        if ((lim1 == count) || (lim2 == 0)) index1 = count / 2;
        delete[] mean_;
        delete[] var_;
    }

    void ComponentInitKDT::planeSplit(unsigned *indices, unsigned count, unsigned cutdim, float cutval, unsigned &lim1,
                                      unsigned &lim2) {
        /* Move vector indices for left subtree to front of list. */
        int left = 0;
        int right = count - 1;
        for (;;) {
            const float *vl = index->getBaseData() + indices[left] * index->getBaseDim();
            const float *vr = index->getBaseData() + indices[right] * index->getBaseDim();
            while (left <= right && vl[cutdim] < cutval) {
                ++left;
                vl = index->getBaseData() + indices[left] * index->getBaseDim();
            }
            while (left <= right && vr[cutdim] >= cutval) {
                --right;
                vr = index->getBaseData() + indices[right] * index->getBaseDim();
            }
            if (left > right) break;
            std::swap(indices[left], indices[right]);
            ++left;
            --right;
        }
        lim1 = left;//lim1 is the id of the leftmost point <= cutval
        right = count - 1;
        for (;;) {
            const float *vl = index->getBaseData() + indices[left] * index->getBaseDim();
            const float *vr = index->getBaseData() + indices[right] * index->getBaseDim();
            while (left <= right && vl[cutdim] <= cutval) {
                ++left;
                vl = index->getBaseData() + indices[left] * index->getBaseDim();
            }
            while (left <= right && vr[cutdim] > cutval) {
                --right;
                vr = index->getBaseData() + indices[right] * index->getBaseDim();
            }
            if (left > right) break;
            std::swap(indices[left], indices[right]);
            ++left;
            --right;
        }
        lim2 = left;//lim2 is the id of the leftmost point >cutval
    }

    int ComponentInitKDT::selectDivision(std::mt19937 &rng, float *v) {
        int num = 0;
        size_t topind[index->RAND_DIM];

        //Create a list of the indices of the top index->RAND_DIM values.
        for (size_t i = 0; i < index->getBaseDim(); ++i) {
            if ((num < index->RAND_DIM) || (v[i] > v[topind[num - 1]])) {
                // Put this element at end of topind.
                if (num < index->RAND_DIM) {
                    topind[num++] = i;            // Add to list.
                } else {
                    topind[num - 1] = i;         // Replace last element.
                }
                // Bubble end value down to right location by repeated swapping. sort the varience in decrease order
                int j = num - 1;
                while (j > 0 && v[topind[j]] > v[topind[j - 1]]) {
                    std::swap(topind[j], topind[j - 1]);
                    --j;
                }
            }
        }
        // Select a random integer in range [0,num-1], and return that index.
        int rnd = rng() % num;
        return (int) topind[rnd];
    }

    void ComponentInitKDT::DFSbuild(Index::EFANNA::Node *node, std::mt19937 &rng, unsigned *indices, unsigned count,
                                    unsigned offset) {
        //omp_set_lock(&rootlock);
        //std::cout<<node->treeid<<":"<<offset<<":"<<count<<std::endl;
        //omp_unset_lock(&rootlock);

        if (count <= index->TNS) {
            node->DivDim = -1;
            node->Lchild = nullptr;
            node->Rchild = nullptr;
            node->StartIdx = offset;
            node->EndIdx = offset + count;
            //add points

        } else {
            unsigned idx;
            unsigned cutdim;
            float cutval;
            meanSplit(rng, indices, count, idx, cutdim, cutval);
            node->DivDim = cutdim;
            node->DivVal = cutval;
            node->StartIdx = offset;
            node->EndIdx = offset + count;
            auto *nodeL = new Index::EFANNA::Node();
            auto *nodeR = new Index::EFANNA::Node();
            node->Lchild = nodeL;
            nodeL->treeid = node->treeid;
            DFSbuild(nodeL, rng, indices, idx, offset);
            node->Rchild = nodeR;
            nodeR->treeid = node->treeid;
            DFSbuild(nodeR, rng, indices + idx, count - idx, offset + idx);
        }
    }

    void ComponentInitKDT::DFStest(unsigned level, unsigned dim, Index::EFANNA::Node *node) {
        if (node->Lchild != nullptr) {
            DFStest(++level, node->DivDim, node->Lchild);
            //if(level > 15)
            std::cout << "dim: " << node->DivDim << "--cutval: " << node->DivVal << "--S: " << node->StartIdx << "--E: "
                      << node->EndIdx << " TREE: " << node->treeid << std::endl;
            if (node->Lchild->Lchild == nullptr) {
                std::vector<unsigned> &tmp = index->LeafLists[node->treeid];
                for (unsigned i = node->Rchild->StartIdx; i < node->Rchild->EndIdx; i++) {
                    const float *tmpfea = index->getBaseData() + tmp[i] * index->getBaseDim() + node->DivDim;
                    std::cout << *tmpfea << " ";
                }
                std::cout << std::endl;
            }
        } else if (node->Rchild != nullptr) {
            DFStest(++level, node->DivDim, node->Rchild);
        } else {
            std::cout << "dim: " << dim << std::endl;
            std::vector<unsigned> &tmp = index->LeafLists[node->treeid];
            for (unsigned i = node->StartIdx; i < node->EndIdx; i++) {
                const float *tmpfea = index->getBaseData() + tmp[i] * index->getBaseDim() + dim;
                std::cout << *tmpfea << " ";
            }
            std::cout << std::endl;
        }
    }

    void ComponentInitKDT::getMergeLevelNodeList(Index::EFANNA::Node *node, size_t treeid, unsigned deepth) {
        auto ml = index->getParam().get<unsigned>("mLevel");
        if (node->Lchild != nullptr && node->Rchild != nullptr && deepth < ml) {
            deepth++;
            getMergeLevelNodeList(node->Lchild, treeid, deepth);
            getMergeLevelNodeList(node->Rchild, treeid, deepth);
        } else if (deepth == ml) {
            index->mlNodeList.emplace_back(node, treeid);
        } else {
            index->error_flag = true;
            if (deepth < index->max_deepth)index->max_deepth = deepth;
        }
    }

    Index::EFANNA::Node *ComponentInitKDT::SearchToLeaf(Index::EFANNA::Node *node, size_t id) {
        if (node->Lchild != nullptr && node->Rchild != nullptr) {
            const float *v = index->getBaseData() + id * index->getBaseDim();
            if (v[node->DivDim] < node->DivVal)
                return SearchToLeaf(node->Lchild, id);
            else
                return SearchToLeaf(node->Rchild, id);
        } else
            return node;
    }

    void ComponentInitKDT::mergeSubGraphs(size_t treeid, Index::EFANNA::Node *node) {
        auto K = index->getParam().get<unsigned>("K");

        if (node->Lchild != nullptr && node->Rchild != nullptr) {
            mergeSubGraphs(treeid, node->Lchild);
            mergeSubGraphs(treeid, node->Rchild);

            size_t numL = node->Lchild->EndIdx - node->Lchild->StartIdx;
            size_t numR = node->Rchild->EndIdx - node->Rchild->StartIdx;
            size_t start, end;
            Index::EFANNA::Node *root;
            if (numL < numR) {
                root = node->Rchild;
                start = node->Lchild->StartIdx;
                end = node->Lchild->EndIdx;
            } else {
                root = node->Lchild;
                start = node->Rchild->StartIdx;
                end = node->Rchild->EndIdx;
            }

            //std::cout << start << " " << end << std::endl;

            for (; start < end; start++) {

                size_t feature_id = index->LeafLists[treeid][start];

                Index::EFANNA::Node *leaf = SearchToLeaf(root, feature_id);
                for (size_t i = leaf->StartIdx; i < leaf->EndIdx; i++) {
                    size_t tmpfea = index->LeafLists[treeid][i];
                    float dist = index->getDist()->compare(index->getBaseData() + tmpfea * index->getBaseDim(),
                                                           index->getBaseData() + feature_id * index->getBaseDim(),
                                                           index->getBaseDim());

                    {
                        Index::LockGuard guard(index->graph_[tmpfea].lock);
                        if (index->knn_graph[tmpfea].size() < K || dist < index->knn_graph[tmpfea].begin()->distance) {
                            Index::Candidate c1(feature_id, dist);
                            index->knn_graph[tmpfea].insert(c1);
                            if (index->knn_graph[tmpfea].size() > K)
                                index->knn_graph[tmpfea].erase(index->knn_graph[tmpfea].begin());
                        }
                    }

                    {
                        Index::LockGuard guard(index->graph_[feature_id].lock);
                        if (index->knn_graph[feature_id].size() < K ||
                            dist < index->knn_graph[feature_id].begin()->distance) {
                            Index::Candidate c1(tmpfea, dist);
                            index->knn_graph[feature_id].insert(c1);
                            if (index->knn_graph[feature_id].size() > K)
                                index->knn_graph[feature_id].erase(index->knn_graph[feature_id].begin());

                        }
                    }
                }
            }
        }
    }



    void ComponentInitFANNG::InitInner() {
        SetConfigs();

        init();

        // graph_ -> final_graph
        index->getFinalGraph().resize(index->getBaseLen());
#ifdef PARALLEL
#pragma omp parallel for
#endif
        for (unsigned i = 0; i < index->getBaseLen(); i++) {
            std::vector<Index::SimpleNeighbor> tmp;

            for (int j = 0; j < index->L; j++) {
                tmp.push_back(Index::SimpleNeighbor(index->graph_[i].pool[j].id, index->graph_[i].pool[j].distance));
            }

//            for (auto &j : index->graph_[i].pool)
//                tmp.push_back(Index::SimpleNeighbor(j.id, j.distance));

            index->getFinalGraph()[i] = tmp;

            std::vector<Index::Neighbor>().swap(index->graph_[i].pool);
            std::vector<unsigned>().swap(index->graph_[i].nn_new);
            std::vector<unsigned>().swap(index->graph_[i].nn_old);
            std::vector<unsigned>().swap(index->graph_[i].rnn_new);
            std::vector<unsigned>().swap(index->graph_[i].rnn_new);
        }

        std::vector<Index::nhood>().swap(index->graph_);
    }

    void ComponentInitFANNG::SetConfigs() {
        index->L = index->getParam().get<unsigned>("L");
    }

    void ComponentInitFANNG::init() {
        index->graph_.resize(index->getBaseLen());
#ifdef PARALLEL
#pragma omp parallel for
#endif
        for (unsigned i = 0; i < index->getBaseLen(); i++) {
            std::vector<Index::Neighbor> tmp;
            for (unsigned j = 0; j < index->getBaseLen(); j++) {
                if (i == j) continue;

                float dist = index->getDist()->compare(index->getBaseData() + index->getBaseDim() * i,
                                                       index->getBaseData() + index->getBaseDim() * j,
                                                       index->getBaseDim());
                tmp.emplace_back(Index::Neighbor(j, dist, true));
            }
            std::make_heap(tmp.begin(), tmp.end(), std::greater<Index::Neighbor>());
            index->graph_[i].pool.reserve(index->L);
            for (unsigned j = 0; j < index->L; j++) {
                index->graph_[i].pool.emplace_back(tmp[0]);
                std::pop_heap(tmp.begin(), tmp.end(), std::greater<Index::Neighbor>());
                tmp.pop_back();
            }
        }
    }


    void ComponentInitRand::InitInner() {
        SetConfigs();

        index->graph_.resize(index->getBaseLen());
        std::mt19937 rng(rand());

#pragma omp parallel for
        for (unsigned i = 0; i < index->getBaseLen(); i++) {
            std::vector<unsigned> tmp(index->L);

            weavess::GenRandom(rng, tmp.data(), index->L, index->getBaseLen());

            for (unsigned j = 0; j < index->L; j++) {
                unsigned id = tmp[j];

                if (id == i)continue;
                float dist = index->getDist()->compare(index->getBaseData() + i * index->getBaseDim(),
                                                       index->getBaseData() + id * index->getBaseDim(),
                                                       (unsigned) index->getBaseDim());

                index->graph_[i].pool.emplace_back(id, dist, true);
            }
            std::make_heap(index->graph_[i].pool.begin(), index->graph_[i].pool.end());
            index->graph_[i].pool.reserve(index->L);
        }

        index->getFinalGraph().resize(index->getBaseLen());
        for (unsigned i = 0; i < index->getBaseLen(); i++) {
            std::vector<Index::SimpleNeighbor> tmp;

            std::sort(index->graph_[i].pool.begin(), index->graph_[i].pool.end());

            for (auto &j : index->graph_[i].pool) {
                tmp.push_back(Index::SimpleNeighbor(j.id, j.distance));
            }

            index->getFinalGraph()[i] = tmp;

            std::vector<Index::Neighbor>().swap(index->graph_[i].pool);
            std::vector<unsigned>().swap(index->graph_[i].nn_new);
            std::vector<unsigned>().swap(index->graph_[i].nn_old);
            std::vector<unsigned>().swap(index->graph_[i].rnn_new);
            std::vector<unsigned>().swap(index->graph_[i].rnn_new);
        }

        std::vector<Index::nhood>().swap(index->graph_);
    }

    void ComponentInitRand::SetConfigs() {
        index->L = index->getParam().get<unsigned>("L");
    }





    // IEH
    void ComponentInitIEH::InitInner() {
        std::string train_argv = index->getParam().get<std::string>("train");
        std::string test_argv = index->getParam().get<std::string>("test");
        std::string func_argv = index->getParam().get<std::string>("func");
        std::string basecode_argv = index->getParam().get<std::string>("basecode");
        std::string knntable_argv = index->getParam().get<std::string>("knntable");

        Index::Codes basecode;

        std::cout << "max support code length 32 and search hamming radius no greater than 1" << std::endl;
        LoadHashFunc(&func_argv[0], index->func);
        std::cout << "load hash function complete" << std::endl;
        LoadBaseCode(&basecode_argv[0], basecode);
        std::cout << "load base data code complete" << std::endl;
        LoadData(&train_argv[0], index->train);
        std::cout << "load base data complete" << std::endl;
        LoadData(&test_argv[0], index->test);
        std::cout << "load query data complete" << std::endl;

        // init hash
        BuildHashTable(index->UpperBits, index->LowerBits, basecode, index->tb);
        std::cout << "build hash table complete" << std::endl;

        // init knn graph
        LoadKnnTable(&knntable_argv[0], index->knntable);
        std::cout << "load knn graph complete" << std::endl;

        // init test
        QueryToCode(index->test, index->func, index->querycode);
        std::cout << "convert query code complete" << std::endl;
    }

    void ComponentInitIEH::LoadData(char *filename, Index::Matrix &dataset) {
        std::ifstream in(filename, std::ios::binary);
        if (!in.is_open()) {
            std::cout << "open file error" << std::endl;
            exit(-1);
        }
        unsigned int dim;
        in.read((char *) &dim, 4);
        std::cout << "data dimension: " << dim << std::endl;
        in.seekg(0, std::ios::end);
        std::ios::pos_type ss = in.tellg();
        size_t fsize = (size_t) ss;
        unsigned int num = fsize / (dim + 1) / 4;
        in.seekg(0, std::ios::beg);
        for (size_t i = 0; i < num; i++) {
            in.seekg(4, std::ios::cur);
            std::vector<float> vtmp(dim);
            vtmp.clear();
            for (size_t j = 0; j < dim; j++) {
                float tmp;
                in.read((char *) &tmp, 4);
                vtmp.push_back(tmp);
            }
            dataset.push_back(vtmp);
        }//cout<<dataset.size()<<endl;
        in.close();
    }

    void StringSplit(std::string src, std::vector<std::string> &des) {
        int start = 0;
        int end = 0;
        for (size_t i = 0; i < src.length(); i++) {
            if (src[i] == ' ') {
                end = i;
                //if(end>start)cout<<start<<" "<<end<<" "<<src.substr(start,end-start)<<endl;
                des.push_back(src.substr(start, end - start));
                start = i + 1;
            }
        }
    }

    void ComponentInitIEH::LoadHashFunc(char *filename, Index::Matrix &func) {
        std::ifstream in(filename);
        char buf[MAX_ROWSIZE];

        while (!in.eof()) {
            in.getline(buf, MAX_ROWSIZE);
            std::string strtmp(buf);
            std::vector<std::string> strs;
            StringSplit(strtmp, strs);
            if (strs.size() < 2)continue;
            std::vector<float> ftmp;
            for (size_t i = 0; i < strs.size(); i++) {
                float f = atof(strs[i].c_str());
                ftmp.push_back(f);
                //cout<<f<<" ";
            }//cout<<endl;
            //cout<<strtmp<<endl;
            func.push_back(ftmp);
        }//cout<<func.size()<<endl;
        in.close();
    }

    void ComponentInitIEH::LoadBaseCode(char *filename, Index::Codes &base) {
        std::ifstream in(filename);
        char buf[MAX_ROWSIZE];
        //int cnt = 0;
        while (!in.eof()) {
            in.getline(buf, MAX_ROWSIZE);
            std::string strtmp(buf);
            std::vector<std::string> strs;
            StringSplit(strtmp, strs);
            if (strs.size() < 2)continue;
            unsigned int codetmp = 0;
            for (size_t i = 0; i < strs.size(); i++) {
                unsigned int c = atoi(strs[i].c_str());
                codetmp = codetmp << 1;
                codetmp += c;

            }//if(cnt++ > 999998){cout<<strs.size()<<" "<<buf<<" "<<codetmp<<endl;}
            base.push_back(codetmp);
        }//cout<<base.size()<<endl;
        in.close();
    }

    void ComponentInitIEH::BuildHashTable(int upbits, int lowbits, Index::Codes base, Index::HashTable &tb) {
        tb.clear();
        for (int i = 0; i < (1 << upbits); i++) {
            Index::HashBucket emptyBucket;
            tb.push_back(emptyBucket);
        }
        for (size_t i = 0; i < base.size(); i++) {
            unsigned int idx1 = base[i] >> lowbits;
            unsigned int idx2 = base[i] - (idx1 << lowbits);
            if (tb[idx1].find(idx2) != tb[idx1].end()) {
                tb[idx1][idx2].push_back(i);
            } else {
                std::vector<unsigned int> v;
                v.push_back(i);
                tb[idx1].insert(make_pair(idx2, v));
            }
        }
    }

    void ComponentInitIEH::LoadKnnTable(char *filename, std::vector<Index::CandidateHeap2> &tb) {
        std::ifstream in(filename, std::ios::binary);
        in.seekg(0, std::ios::end);
        std::ios::pos_type ss = in.tellg();
        size_t fsize = (size_t) ss;
        int dim;
        in.seekg(0, std::ios::beg);
        in.read((char *) &dim, sizeof(int));
        size_t num = fsize / (dim + 1) / 4;
        std::cout << "load graph " << num << " " << dim << std::endl;
        in.seekg(0, std::ios::beg);
        tb.clear();
        for (size_t i = 0; i < num; i++) {
            Index::CandidateHeap2 heap;
            in.read((char *) &dim, sizeof(int));
            for (int j = 0; j < dim; j++) {
                int id;
                in.read((char *) &id, sizeof(int));
                Index::Candidate2<float> can(id, -1);
                heap.insert(can);
            }
            tb.push_back(heap);

        }
        in.close();
    }

    bool MatrixMultiply(Index::Matrix A, Index::Matrix B, Index::Matrix &C) {
        if (A.size() == 0 || B.size() == 0) {
            std::cout << "matrix a or b size 0" << std::endl;
            return false;
        } else if (A[0].size() != B.size()) {
            std::cout << "--error: matrix a, b dimension not agree" << std::endl;
            std::cout << "A" << A.size() << " * " << A[0].size() << std::endl;
            std::cout << "B" << B.size() << " * " << B[0].size() << std::endl;
            return false;
        }
        for (size_t i = 0; i < A.size(); i++) {
            std::vector<float> tmp;
            for (size_t j = 0; j < B[0].size(); j++) {
                float fnum = 0;
                for (size_t k = 0; k < B.size(); k++)fnum += A[i][k] * B[k][j];
                tmp.push_back(fnum);
            }
            C.push_back(tmp);
        }
        return true;
    }

    void ComponentInitIEH::QueryToCode(Index::Matrix query, Index::Matrix func, Index::Codes &querycode) {
        Index::Matrix Z;
        if (!MatrixMultiply(query, func, Z)) { return; }
        for (size_t i = 0; i < Z.size(); i++) {
            unsigned int codetmp = 0;
            for (size_t j = 0; j < Z[0].size(); j++) {
                if (Z[i][j] > 0) {
                    codetmp = codetmp << 1;
                    codetmp += 1;
                } else {
                    codetmp = codetmp << 1;
                    codetmp += 0;
                }
            }
            //if(i<3)cout<<codetmp<<endl;
            querycode.push_back(codetmp);
        }//cout<<querycode.size()<<endl;
    }


    // NSW
    void ComponentInitNSW::InitInner() {
        SetConfigs();

        index->nodes_.resize(index->getBaseLen());
        Index::HnswNode *first = new Index::HnswNode(0, 0, index->NN_, index->NN_);
        index->nodes_[0] = first;
        index->enterpoint_ = first;
#pragma omp parallel
        {
            auto *visited_list = new Index::VisitedList(index->getBaseLen());
#pragma omp for schedule(dynamic, 128)
            for (size_t i = 1; i < index->getBaseLen(); ++i) {
                auto *qnode = new Index::HnswNode(i, 0, index->NN_, index->NN_);
                index->nodes_[i] = qnode;
                InsertNode(qnode, visited_list);
            }
            delete visited_list;
        }
    }

    void ComponentInitNSW::SetConfigs() {
        index->NN_ = index->getParam().get<unsigned>("NN");
        index->ef_construction_ = index->getParam().get<unsigned>("ef_construction");
        index->n_threads_ = index->getParam().get<unsigned>("n_threads");
    }

    void ComponentInitNSW::InsertNode(Index::HnswNode *qnode, Index::VisitedList *visited_list) {
        Index::HnswNode *enterpoint = index->enterpoint_;

        std::priority_queue<Index::FurtherFirst> result;
        std::priority_queue<Index::CloserFirst> tmp;

        // CANDIDATE
        SearchAtLayer(qnode, enterpoint, 0, visited_list, result);

        while (!result.empty()) {
            tmp.push(Index::CloserFirst(result.top().GetNode(), result.top().GetDistance()));
            result.pop();
        }

        int pos = 0;
        while (!tmp.empty() && pos < index->NN_) {
            auto *top_node = tmp.top().GetNode();
            tmp.pop();
            Link(top_node, qnode, 0);
            pos++;
        }
    }

    void ComponentInitNSW::SearchAtLayer(Index::HnswNode *qnode, Index::HnswNode *enterpoint, int level,
                                         Index::VisitedList *visited_list,
                                         std::priority_queue<Index::FurtherFirst> &result) {
        // TODO: check Node 12bytes => 8bytes
        std::priority_queue<Index::CloserFirst> candidates;
        float d = index->getDist()->compare(index->getBaseData() + qnode->GetId() * index->getBaseDim(),
                                            index->getBaseData() + enterpoint->GetId() * index->getBaseDim(),
                                            index->getBaseDim());
        result.emplace(enterpoint, d);
        candidates.emplace(enterpoint, d);

        visited_list->Reset();
        visited_list->MarkAsVisited(enterpoint->GetId());

        while (!candidates.empty()) {
            const Index::CloserFirst &candidate = candidates.top();
            float lower_bound = result.top().GetDistance();
            if (candidate.GetDistance() > lower_bound)
                break;

            Index::HnswNode *candidate_node = candidate.GetNode();
            std::unique_lock<std::mutex> lock(candidate_node->GetAccessGuard());
            const std::vector<Index::HnswNode *> &neighbors = candidate_node->GetFriends(level);
            candidates.pop();
            for (const auto &neighbor : neighbors) {
                int id = neighbor->GetId();
                if (visited_list->NotVisited(id)) {
                    visited_list->MarkAsVisited(id);
                    d = index->getDist()->compare(index->getBaseData() + qnode->GetId() * index->getBaseDim(),
                                                  index->getBaseData() + neighbor->GetId() * index->getBaseDim(),
                                                  index->getBaseDim());
                    if (result.size() < index->ef_construction_ || result.top().GetDistance() > d) {
                        result.emplace(neighbor, d);
                        candidates.emplace(neighbor, d);
                        if (result.size() > index->ef_construction_)
                            result.pop();
                    }
                }
            }
        }
    }

    void ComponentInitNSW::Link(Index::HnswNode *source, Index::HnswNode *target, int level) {
        source->AddFriends(target, true);
        target->AddFriends(source, true);
    }


    // HNSW
    void ComponentInitHNSW::InitInner() {
        SetConfigs();

        Build(false);

//        for(int i = 0; i < index->nodes_.size(); i ++) {
//            std::cout << "node id : " << i << std::endl;
//            std::cout << "node level : " << index->nodes_[i]->GetLevel() << std::endl;
//            for(int level = index->nodes_[i]->GetLevel(); level >= 0; level --) {
//                std::cout << "level : " << level << std::endl;
//                for(int j = 0; j < index->nodes_[i]->GetFriends(level).size(); j ++) {
//                    std::cout << index->nodes_[i]->GetFriends(level)[j]->GetId() << " ";
//                }
//                std::cout << std::endl;
//            }
//        }
    }

    void ComponentInitHNSW::SetConfigs() {
        index->max_m_ = index->getParam().get<unsigned>("max_m");
        index->m_ = index->max_m_;
        index->max_m0_ = index->getParam().get<unsigned>("max_m0");
        auto ef_construction_ = index->getParam().get<unsigned>("ef_construction");
        if (ef_construction_ > 0) index->ef_construction_ = ef_construction_;
        index->n_threads_ = index->getParam().get<unsigned>("n_threads");
        index->mult = index->getParam().get<int>("mult");
        index->level_mult_ = index->mult > 0 ? index->mult : (1 / log(1.0 * index->m_));
    }

    void ComponentInitHNSW::Build(bool reverse) {
        index->nodes_.resize(index->getBaseLen());
        int level = GetRandomNodeLevel();
        auto *first = new Index::HnswNode(0, level, index->max_m_, index->max_m0_);
        index->nodes_[0] = first;
        index->max_level_ = level;
        index->enterpoint_ = first;
#pragma omp parallel
        {
            auto *visited_list = new Index::VisitedList(index->getBaseLen());
#pragma omp for schedule(dynamic, 128)
            for (size_t i = 1; i < index->getBaseLen(); ++i) {
                int level = GetRandomNodeLevel();
                auto *qnode = new Index::HnswNode(i, level, index->max_m_, index->max_m0_);
                index->nodes_[i] = qnode;
                InsertNode(qnode, visited_list);
            }

            delete visited_list;
        }
    }

    int ComponentInitHNSW::GetRandomSeedPerThread() {
        int tid = omp_get_thread_num();
        int g_seed = 17;
        for (int i = 0; i <= tid; ++i)
            g_seed = 214013 * g_seed + 2531011;
        return (g_seed >> 16) & 0x7FFF;
    }

    int ComponentInitHNSW::GetRandomNodeLevel() {
        static thread_local std::mt19937 rng(GetRandomSeedPerThread());
        static thread_local std::uniform_real_distribution<double> uniform_distribution(0.0, 1.0);
        double r = uniform_distribution(rng);

        if (r < std::numeric_limits<double>::epsilon())
            r = 1.0;
        return (int) (-log(r) * index->level_mult_);
    }

    void ComponentInitHNSW::InsertNode(Index::HnswNode *qnode, Index::VisitedList *visited_list) {
        int cur_level = qnode->GetLevel();
        std::unique_lock<std::mutex> max_level_lock(index->max_level_guard_, std::defer_lock);
        if (cur_level > index->max_level_)
            max_level_lock.lock();

        int max_level_copy = index->max_level_;
        Index::HnswNode *enterpoint = index->enterpoint_;

        if (cur_level < max_level_copy) {
            Index::HnswNode *cur_node = enterpoint;

            float d = index->getDist()->compare(index->getBaseData() + qnode->GetId() * index->getBaseDim(),
                                                index->getBaseData() + cur_node->GetId() * index->getBaseDim(),
                                                index->getBaseDim());
            float cur_dist = d;
            for (auto i = max_level_copy; i > cur_level; --i) {
                bool changed = true;
                while (changed) {
                    changed = false;
                    std::unique_lock<std::mutex> local_lock(cur_node->GetAccessGuard());
                    const std::vector<Index::HnswNode *> &neighbors = cur_node->GetFriends(i);

                    for (auto iter = neighbors.begin(); iter != neighbors.end(); ++iter) {
                        d = index->getDist()->compare(index->getBaseData() + qnode->GetId() * index->getBaseDim(),
                                                      index->getBaseData() + (*iter)->GetId() * index->getBaseDim(),
                                                      index->getBaseDim());

                        if (d < cur_dist) {
                            cur_dist = d;
                            cur_node = *iter;
                            changed = true;
                        }
                    }
                }
            }
            enterpoint = cur_node;
        }

        // PRUNE
        ComponentPrune *a = new ComponentPruneHeuristic(index);

        for (auto i = std::min(max_level_copy, cur_level); i >= 0; --i) {
            std::priority_queue<Index::FurtherFirst> result;
            SearchAtLayer(qnode, enterpoint, i, visited_list, result);

            a->Hnsw2Neighbor(qnode->GetId(), index->m_, result);

            while (!result.empty()) {
                auto *top_node = result.top().GetNode();
                result.pop();
                Link(top_node, qnode, i);
                Link(qnode, top_node, i);
            }
        }
        if (cur_level > index->enterpoint_->GetLevel()) {
            index->enterpoint_ = qnode;
            index->max_level_ = cur_level;
        }
    }

    void ComponentInitHNSW::SearchAtLayer(Index::HnswNode *qnode, Index::HnswNode *enterpoint, int level,
                                          Index::VisitedList *visited_list,
                                          std::priority_queue<Index::FurtherFirst> &result) {
        // TODO: check Node 12bytes => 8bytes
        std::priority_queue<Index::CloserFirst> candidates;
        float d = index->getDist()->compare(index->getBaseData() + qnode->GetId() * index->getBaseDim(),
                                            index->getBaseData() + enterpoint->GetId() * index->getBaseDim(),
                                            index->getBaseDim());
        result.emplace(enterpoint, d);
        candidates.emplace(enterpoint, d);

        visited_list->Reset();
        visited_list->MarkAsVisited(enterpoint->GetId());

        while (!candidates.empty()) {
            const Index::CloserFirst &candidate = candidates.top();
            float lower_bound = result.top().GetDistance();
            if (candidate.GetDistance() > lower_bound)
                break;

            Index::HnswNode *candidate_node = candidate.GetNode();
            std::unique_lock<std::mutex> lock(candidate_node->GetAccessGuard());
            const std::vector<Index::HnswNode *> &neighbors = candidate_node->GetFriends(level);
            candidates.pop();

            for (const auto &neighbor : neighbors) {
                int id = neighbor->GetId();
                if (visited_list->NotVisited(id)) {
                    visited_list->MarkAsVisited(id);
                    d = index->getDist()->compare(index->getBaseData() + qnode->GetId() * index->getBaseDim(),
                                                  index->getBaseData() + neighbor->GetId() * index->getBaseDim(),
                                                  index->getBaseDim());
                    if (result.size() < index->ef_construction_ || result.top().GetDistance() > d) {
                        result.emplace(neighbor, d);
                        candidates.emplace(neighbor, d);
                        if (result.size() > index->ef_construction_)
                            result.pop();
                    }
                }
            }
        }
    }

    void ComponentInitHNSW::Link(Index::HnswNode *source, Index::HnswNode *target, int level) {
        std::unique_lock<std::mutex> lock(source->GetAccessGuard());
        std::vector<Index::HnswNode *> &neighbors = source->GetFriends(level);
        neighbors.push_back(target);
        bool shrink = (level > 0 && neighbors.size() > source->GetMaxM()) ||
                      (level <= 0 && neighbors.size() > source->GetMaxM0());
        if (!shrink) return;

        std::priority_queue<Index::FurtherFirst> tempres;
        for (const auto &neighbor : neighbors) {
            float tmp = index->getDist()->compare(index->getBaseData() + source->GetId() * index->getBaseDim(),
                                                  index->getBaseData() + neighbor->GetId() * index->getBaseDim(),
                                                  index->getBaseDim());
            tempres.push(Index::FurtherFirst(neighbor, tmp));
        }

        // PRUNE
        ComponentPrune *a = new ComponentPruneHeuristic(index);
        a->Hnsw2Neighbor(source->GetId(), tempres.size() - 1, tempres);

        neighbors.clear();
        while (!tempres.empty()) {
            neighbors.emplace_back(tempres.top().GetNode());
            tempres.pop();
        }
        std::priority_queue<Index::FurtherFirst>().swap(tempres);
    }


    // ANNG
    void ComponentInitANNG::InitInner() {
        SetConfigs();

        index->nodes_.resize(index->getBaseLen());
        auto *first = new Index::HnswNode(0, 0, index->NN_, index->NN_);
        index->nodes_[0] = first;
        index->enterpoint_ = first;
        std::vector<unsigned> obj = {0};
        MakeVPTree(obj);

#pragma omp parallel
        {
            auto *visited_list = new Index::VisitedList(index->getBaseLen());
#pragma omp for schedule(dynamic, 128)
            for (size_t i = 1; i < index->getBaseLen(); ++i) {
                auto *qnode = new Index::HnswNode(i, 0, index->NN_, index->NN_);
                index->nodes_[i] = qnode;
                InsertNode(qnode, visited_list);
#pragma omp critical
                Insert(i);
            }
            delete visited_list;
        }

        index->getFinalGraph().resize(index->getBaseLen());

        for(int i = 0; i < index->nodes_.size(); i ++) {
            for(auto node : index->nodes_[i]->GetFriends(0)) {
                float dist = index->getDist()->compare(index->getBaseData() + index->getBaseDim() * i,
                                                       index->getBaseData() + index->getBaseDim() * node->GetId(),
                                                       index->getBaseDim());
                index->getFinalGraph()[i].emplace_back(node->GetId(), dist);
                //std::cout << index->getFinalGraph()[i].back().id << "|" << index->getFinalGraph()[i].back().distance << " ";
            }
            //std::cout << index->nodes_[i]->GetFriends(0).size() << std::endl;
        }

        std::vector<Index::HnswNode*>().swap(index->nodes_);
    }

    void ComponentInitANNG::SetConfigs() {
        index->NN_ = index->getParam().get<unsigned>("NN");
        index->ef_construction_ = index->getParam().get<unsigned>("ef_construction");
        index->n_threads_ = index->getParam().get<unsigned>("n_threads");
    }

    void ComponentInitANNG::InsertNode(Index::HnswNode *qnode, Index::VisitedList *visited_list) {
        Index::HnswNode *enterpoint = index->enterpoint_;

        std::priority_queue<Index::FurtherFirst> result;
        std::priority_queue<Index::CloserFirst> tmp;

        // CANDIDATE
        SearchAtLayer(qnode, enterpoint, 0, visited_list, result);
        //std::cout << qnode->GetId() << " " << result.size() << std::endl;

        while (!result.empty()) {
            tmp.push(Index::CloserFirst(result.top().GetNode(), result.top().GetDistance()));
            result.pop();
        }

        int pos = 0;
        while (!tmp.empty() && pos < index->NN_) {
            auto *top_node = tmp.top().GetNode();
            tmp.pop();
            Link(top_node, qnode, 0);
            pos++;
        }
    }

    void ComponentInitANNG::SearchAtLayer(Index::HnswNode *qnode, Index::HnswNode *enterpoint, int level,
                                         Index::VisitedList *visited_list,
                                         std::priority_queue<Index::FurtherFirst> &result) {

        float radius = static_cast<float>(FLT_MAX);
        std::priority_queue<Index::CloserFirst> candidates;

        float d = index->getDist()->compare(index->getBaseData() + qnode->GetId() * index->getBaseDim(),
                                            index->getBaseData() + enterpoint->GetId() * index->getBaseDim(),
                                            index->getBaseDim());
        result.emplace(enterpoint, d);
        candidates.emplace(enterpoint, d);

        visited_list->Reset();
        visited_list->MarkAsVisited(enterpoint->GetId());

        float explorationRadius = index->explorationCoefficient * radius;

        while (!candidates.empty()) {
            const Index::CloserFirst &candidate = candidates.top();
            if (candidate.GetDistance() > explorationRadius)
                break;

            Index::HnswNode *candidate_node = candidate.GetNode();
            std::unique_lock<std::mutex> lock(candidate_node->GetAccessGuard());
            const std::vector<Index::HnswNode *> &neighbors = candidate_node->GetFriends(level);
            candidates.pop();
            for (const auto &neighbor : neighbors) {
                int id = neighbor->GetId();
                if (visited_list->NotVisited(id)) {
                    visited_list->MarkAsVisited(id);
                    d = index->getDist()->compare(index->getBaseData() + qnode->GetId() * index->getBaseDim(),
                                                  index->getBaseData() + neighbor->GetId() * index->getBaseDim(),
                                                  index->getBaseDim());
                    //sc.distanceComputationCount++;
                    if (d <= explorationRadius){
                        candidates.emplace(neighbor, d);
                        if (d <= radius){
                            result.emplace(neighbor, d);
                            if (result.size() > index->ef_construction_){
                                if(result.top().GetDistance() >= d) {
                                    if(result.size() > index->ef_construction_) {
                                        result.pop();
                                    }
                                    radius = result.top().GetDistance();
                                    explorationRadius = index->explorationCoefficient * radius;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    void ComponentInitANNG::Link(Index::HnswNode *source, Index::HnswNode *target, int level) {
        source->AddFriends(target, true);
        target->AddFriends(source, true);
    }

    // Construct the tree from given objects set
    void ComponentInitANNG::MakeVPTree(const std::vector<unsigned>& objects)
    {
        index->vp_tree.m_root.reset();
// #pragma omp critical
#pragma omp parallel
        {
#pragma omp single
#pragma omp task
            index->vp_tree.m_root = MakeVPTree(objects, index->vp_tree.m_root);
        }
    }

    void ComponentInitANNG::InsertSplitLeafRoot(Index::VPNodePtr& root, const unsigned& new_value)
    {
        //std::cout << "	spit leaf root" << std::endl;
        // Split the root node if root is the leaf
        //
        Index::VPNodePtr s1(new Index::VPNodeType);
        Index::VPNodePtr s2(new Index::VPNodeType);

        // Set vantage point to root
        root->set_value(root->m_objects_list[0]);
        //root->m_objects_list.clear();

        //s1->AddObject(root->get_value());

        root->AddChild(0, s1);
        s1->set_parent(root);
        s1->set_leaf_node(true);

        root->AddChild(1, s2);
        s2->set_parent(root);
        s2->set_leaf_node(true);

        root->set_leaf_node(false);

        for(size_t c_pos = 0; c_pos < root->get_objects_count(); ++c_pos)
            Insert(root->m_objects_list[c_pos], root);

        root->m_objects_list.clear();

        Insert(new_value, root);

        //RedistributeAmongLeafNodes(root, new_value);

        //m_root->m_mu_list[0] = 0;
        //m_root->set_value(new_value); // Set Vantage Point
    }

    // Recursively collect data from subtree, and push them into S
    void ComponentInitANNG::CollectObjects(const Index::VPNodePtr& node, std::vector<unsigned>& S)
    {
        if(node->get_leaf_node())
            S.insert(S.end(), node->m_objects_list.begin(), node->m_objects_list.end() );
        else
        {
            for (size_t c_pos = 0; c_pos < node->get_branches_count(); ++c_pos)
                CollectObjects(node->m_child_list[c_pos], S);
        }
    }

    // (max{d(v, sj) | sj c SS1} + min{d(v, sj) | sj c SS2}) / 2
    const float ComponentInitANNG::MedianSumm(const std::vector<unsigned>& SS1, const std::vector<unsigned>& SS2, const unsigned& v) const
    {
        float c_max_distance = 0;
        float c_min_distance = 0;
        float c_current_distance = 0;

        if(!SS1.empty())
        {
            // max{d(v, sj) | sj c SSj}
            typename std::vector<unsigned>::const_iterator it_obj = SS1.begin();
            c_max_distance = index->getDist()->compare(index->getBaseData() + index->getBaseDim() * (*it_obj),
                                                       index->getBaseData() + index->getBaseDim() * v,
                                                       index->getBaseDim());
            ++it_obj;
            c_current_distance = c_max_distance;
            while(it_obj != SS1.end())
            {
                c_current_distance = index->getDist()->compare(index->getBaseData() + index->getBaseDim() * (*it_obj),
                                                               index->getBaseData() + index->getBaseDim() * v,
                                                               index->getBaseDim());
                if(c_current_distance > c_max_distance)
                    c_max_distance = c_current_distance;

                ++it_obj;
            }
        }

        if(!SS2.empty())
        {
            // min{d(v, sj) | sj c SSj}
            typename std::vector<unsigned>::const_iterator it_obj = SS2.begin();
            c_min_distance = index->getDist()->compare(index->getBaseData() + index->getBaseDim() * (*it_obj),
                                                       index->getBaseData() + index->getBaseDim() * v,
                                                       index->getBaseDim());
            ++it_obj;
            c_current_distance = c_min_distance;
            while(it_obj != SS2.end())
            {
                c_current_distance = index->getDist()->compare(index->getBaseData() + index->getBaseDim() * (*it_obj),
                                                               index->getBaseData() + index->getBaseDim() * v,
                                                               index->getBaseDim());
                if(c_current_distance < c_min_distance)
                    c_min_distance = c_current_distance;

                ++it_obj;
            }
        }

        return (c_max_distance + c_min_distance) / static_cast<float>(2);
    }

    // Calc the median value for object set
    float ComponentInitANNG::Median(const unsigned& value, const std::vector<unsigned>::const_iterator it_begin,
                                        const std::vector<unsigned>::const_iterator it_end)
    {
        std::vector<unsigned>::const_iterator it_obj = it_begin;
        float current_distance = 0;
        size_t count = 0;
        while(it_obj != it_end)
        {
            current_distance += index->getDist()->compare(index->getBaseData() + index->getBaseDim() * (*it_obj),
                                                          index->getBaseData() + index->getBaseDim() * value,
                                                          index->getBaseDim());
            ++it_obj;
            ++count;
        }
        return current_distance / static_cast<float>(count);
    }

    // If any sibling leaf node of L(eaf) is not full,
    // redistribute all objects under P(arent), among the leaf nodes
    void ComponentInitANNG::RedistributeAmongLeafNodes(const Index::VPNodePtr& parent_node, const unsigned& new_value)
    {
        //std::cout << "	redistribute among leaf nodes" << std::endl;
        // F - number of leaf nodes under P(arent)
        // F should be greater then 1
        //size_t F = parent_node->m_child_list.size();
        const size_t F = parent_node->get_branches_count();

        Index::VPNodePtr c_node;
        std::vector<unsigned> S; // Set of leaf objects + new one;

        // Create Set of whole objects from leaf nodes
        CollectObjects(parent_node, S);
        S.push_back(new_value);

        unsigned m_main_val = parent_node->get_value();
        Index *tmp_index = index;

        // Order the objects in S with respect to their distances from P's vantage point
        //Index::NGT::VPTree::ValueSorterType val_sorter(parent_node->get_value(), m_get_distance);
        std::sort(S.begin(), S.end(), [m_main_val, tmp_index](unsigned val1, unsigned val2) {
            float dist1 = tmp_index->getDist()->compare(tmp_index->getBaseData() + tmp_index->getBaseDim() * m_main_val,
                                                        tmp_index->getBaseData() + tmp_index->getBaseDim() * val1,
                                                        tmp_index->getBaseDim());
            float dist2 = tmp_index->getDist()->compare(tmp_index->getBaseData() + tmp_index->getBaseDim() * m_main_val,
                                                        tmp_index->getBaseData() + tmp_index->getBaseDim() * val2,
                                                        tmp_index->getBaseDim());
            return dist1 < dist2;
        });

        // Devide S into F groups of equal cardinality
        size_t c_whole_count = S.size();
        typename std::vector<unsigned>::const_iterator it_obj = S.begin();
        for (size_t c_pos = 0; c_pos < F; ++c_pos)
        {
            size_t c_equal_count = c_whole_count / (F - c_pos);
            c_whole_count -= c_equal_count;

            c_node = parent_node->m_child_list[c_pos];
            c_node->m_objects_list.clear();

            c_node->m_objects_list.insert(c_node->m_objects_list.begin(),
                                          it_obj, it_obj + c_equal_count);
            c_node->set_leaf_node(true);
            it_obj += c_equal_count;
        }

        // Update the boundary distance values
        for (size_t c_pos = 0; c_pos < F - 1; ++c_pos)
        {
            const std::vector<unsigned>& SS1 = parent_node->m_child_list[c_pos]->m_objects_list;
            const std::vector<unsigned>& SS2 = parent_node->m_child_list[c_pos + 1]->m_objects_list;

            parent_node->m_mu_list[c_pos] = MedianSumm(SS1, SS2, parent_node->get_value());
        }

    }

    // If L(eaf) node has a P(arent) node and P has room for one more child,
    // split the leaf node L
    void ComponentInitANNG::SplitLeafNode(const Index::VPNodePtr& parent_node, const size_t child_id, const unsigned& new_value)
    {
        //std::cout << "	split leaf node" << std::endl;
        // F - number of leaf nodes under P(arent)
        //
        const size_t F = parent_node->get_branches_count();
        const size_t k = child_id;

        assert(child_id < parent_node->m_child_list.size());

        Index::VPNodePtr c_leaf_node = parent_node->m_child_list[child_id];

        std::vector<unsigned> S = c_leaf_node->m_objects_list; // Set of leaf objects + new one
        S.push_back(new_value);

        unsigned m_main_val = parent_node->get_value();
        Index *tmp_index = index;

        // Order the objects in S with respect to their distances from P's vantage point
        //NGT::VPTree::ValueSorterType val_sorter(parent_node->get_value(), m_get_distance);
        std::sort(S.begin(), S.end(), [m_main_val, tmp_index](unsigned val1, unsigned val2) {
            float dist1 = tmp_index->getDist()->compare(tmp_index->getBaseData() + tmp_index->getBaseDim() * m_main_val,
                                                        tmp_index->getBaseData() + tmp_index->getBaseDim() * val1,
                                                        tmp_index->getBaseDim());
            float dist2 = tmp_index->getDist()->compare(tmp_index->getBaseData() + tmp_index->getBaseDim() * m_main_val,
                                                        tmp_index->getBaseData() + tmp_index->getBaseDim() * val2,
                                                        tmp_index->getBaseDim());
            return dist1 < dist2;
        });

        // Divide S into 2 groups of equal cardinality
        Index::VPNodePtr ss1_node(new Index::VPNodeType);
        Index::VPNodePtr ss2_node = c_leaf_node;
        ss2_node->m_objects_list.clear();

        parent_node->AddChild(parent_node->get_branches_count(), ss1_node);
        ss1_node->set_parent(parent_node);

        size_t c_half_count = S.size() / 2;
        for(size_t c_pos = 0; c_pos < S.size(); ++c_pos)
        {
            if(c_pos < c_half_count)
                ss1_node->AddObject(S[c_pos]);
            else
                ss2_node->AddObject(S[c_pos]);
        }

        // insertion/shift process
        for(size_t c_pos = F-2; c_pos >= k; --c_pos)
        {
            parent_node->m_mu_list[c_pos+1] = parent_node->m_mu_list[c_pos];
            if(!c_pos) // !!! hack :(
                break;
        }

        const std::vector<unsigned>& SS1 = ss1_node->m_objects_list;
        const std::vector<unsigned>& SS2 = ss2_node->m_objects_list;
        parent_node->m_mu_list[k] = MedianSumm(SS1, SS2, parent_node->get_value());

        // !! --c_pos
        for(size_t c_pos = F-1; c_pos >= k+1; --c_pos)
            parent_node->m_child_list[c_pos + 1] = parent_node->m_child_list[c_pos];

        parent_node->m_child_list[k] = ss1_node;
        parent_node->m_child_list[k + 1] = ss2_node;
    }

    void ComponentInitANNG::Remove(const unsigned& query_value, const Index::VPNodePtr& node)
    {
        if(node->get_leaf_node())
            node->DeleteObject(query_value);
        else
        {
            for (size_t c_pos = 0; c_pos < node->get_branches_count(); ++c_pos)
                Remove(query_value, node->m_child_list[c_pos]);
        }
    }

    // 3.a. Redistribute, among the sibling subtrees
    void ComponentInitANNG::RedistributeAmongNonLeafNodes(const Index::VPNodePtr& parent_node, const size_t k_id,
                                                              const size_t k1_id, const unsigned& new_value)
    {
        //std::cout << "	redistribute among nodes(subtrees)" << std::endl;
        assert(k_id != k1_id);

        size_t num_k = index->vp_tree.get_object_count(parent_node->m_child_list[k_id]);
        size_t num_k1 = index->vp_tree.get_object_count(parent_node->m_child_list[k1_id]);

        size_t average = (num_k + num_k1) / 2;

        if(num_k > num_k1)
        {
            // Create Set of objects from leaf nodes K-th subtree
            std::vector<unsigned> S; // Set of leaf objects + new one;
            CollectObjects(parent_node->m_child_list[k_id], S);
            //S.push_back(new_value);

            unsigned m_main_val = parent_node->get_value();
            Index *tmp_index = index;

            //Index::ValueSorterType val_sorter(parent_node->get_value(), m_get_distance);
            std::sort(S.begin(), S.end(), [m_main_val, tmp_index](unsigned val1, unsigned val2) {
                float dist1 = tmp_index->getDist()->compare(tmp_index->getBaseData() + tmp_index->getBaseDim() * m_main_val,
                                                            tmp_index->getBaseData() + tmp_index->getBaseDim() * val1,
                                                            tmp_index->getBaseDim());
                float dist2 = tmp_index->getDist()->compare(tmp_index->getBaseData() + tmp_index->getBaseDim() * m_main_val,
                                                            tmp_index->getBaseData() + tmp_index->getBaseDim() * val2,
                                                            tmp_index->getBaseDim());
                return dist1 < dist2;
            });

            size_t w = num_k - average;

            std::vector<unsigned> SS1(S.begin(), S.begin() + num_k - w);
            std::vector<unsigned> SS2(S.begin() + num_k - w, S.end());

            SS1.push_back(new_value);

            typename std::vector<unsigned>::const_iterator it_obj = SS2.begin();
            for(;it_obj != SS2.end(); ++it_obj)
                Remove(*it_obj, parent_node->m_child_list[k_id]);

            parent_node->m_mu_list[k_id] = MedianSumm(SS1, SS2, parent_node->get_value());

            for(it_obj = SS2.begin(); it_obj != SS2.end(); ++it_obj)
                Insert(*it_obj, parent_node->m_child_list[k1_id]);

        }else
        {
            // Create Set of objects from leaf nodes K-th subtree
            std::vector<unsigned> S; // Set of leaf objects + new one;
            CollectObjects(parent_node->m_child_list[k1_id], S);
            //S.push_back(new_value);
            unsigned m_main_val = parent_node->get_value();
            Index *tmp_index = index;
            //ValueSorterType val_sorter(parent_node->get_value(), m_get_distance);
            std::sort(S.begin(), S.end(), [m_main_val, tmp_index](unsigned val1, unsigned val2) {
                float dist1 = tmp_index->getDist()->compare(tmp_index->getBaseData() + tmp_index->getBaseDim() * m_main_val,
                                                            tmp_index->getBaseData() + tmp_index->getBaseDim() * val1,
                                                            tmp_index->getBaseDim());
                float dist2 = tmp_index->getDist()->compare(tmp_index->getBaseData() + tmp_index->getBaseDim() * m_main_val,
                                                            tmp_index->getBaseData() + tmp_index->getBaseDim() * val2,
                                                            tmp_index->getBaseDim());
                return dist1 < dist2;
            });

            size_t w = num_k1 - average;

            std::vector<unsigned> SS1(S.begin(), S.begin() + w);
            std::vector<unsigned> SS2(S.begin() + w, S.end());
            SS2.push_back(new_value);

            typename std::vector<unsigned>::const_iterator it_obj = SS1.begin();
            for(;it_obj != SS1.end(); ++it_obj)
                Remove(*it_obj, parent_node->m_child_list[k1_id]);

            parent_node->m_mu_list[k_id] = MedianSumm(SS1, SS2, parent_node->get_value());

            for(it_obj = SS1.begin(); it_obj != SS1.end(); ++it_obj)
                Insert(*it_obj, parent_node->m_child_list[k_id]);
        }

        num_k = index->vp_tree.get_object_count(parent_node->m_child_list[k_id]);
        num_k1 = index->vp_tree.get_object_count(parent_node->m_child_list[k1_id]);

        num_k = 0;
    }

    void ComponentInitANNG::InsertSplitRoot(Index::VPNodePtr& root, const unsigned& new_value)
    {
        //std::cout << "	split root" << std::endl;
        // Split the root node into 2 new nodes s1 and s2 and insert new data
        // according to the strategy SplitLeafNode() or RedistributeAmongLeafNodes()

        Index::VPNodePtr new_root(new Index::VPNodeType);
        Index::VPNodePtr s2_node(new Index::VPNodeType);

        new_root->set_value(root->get_value());
        //new_root->set_value(new_value);
        new_root->AddChild(0, root);
        //new_root->AddChild(0, s2_node);

        root->set_parent(new_root);
        //s2_node->set_parent(new_root);
        //s2_node->set_leaf_node(true);

        root = new_root;

        //Insert(new_value, root);
        //Insert(new_value);

        SplitNonLeafNode(root, 0, new_value);
    }

    // Obtain the best vantage point for objects set
    const unsigned& ComponentInitANNG::SelectVP(const std::vector<unsigned>& objects) {
        assert(!objects.empty());

        return *objects.begin();
    }

    // Construct the tree from given objects set
    Index::VPNodePtr ComponentInitANNG::MakeVPTree(const std::vector<unsigned>& objects, const Index::VPNodePtr& parent){

        if(objects.empty())
            return Index::VPNodePtr(new Index::VPNodeType);

        Index::VPNodePtr new_node(new Index::VPNodeType);

        new_node->set_parent(parent);

        // Set the VP
        new_node->set_value(SelectVP(objects));

        if(objects.size() <= index->vp_tree.m_non_leaf_branching_factor * index->vp_tree.m_leaf_branching_factor)
        {
            for(size_t c_pos = 0; c_pos < index->vp_tree.m_leaf_branching_factor; ++c_pos)
            {
                new_node->AddChild(0, Index::VPNodePtr(new Index::VPNodeType));
                new_node->m_child_list[c_pos]->set_leaf_node(true);
                new_node->m_child_list[c_pos]->set_parent(new_node);
            }

            new_node->m_child_list[0]->m_objects_list.insert(new_node->m_child_list[0]->m_objects_list.begin(), objects.begin()+1, objects.end());

            RedistributeAmongLeafNodes(new_node, *objects.begin());

            return new_node;
        }

        // Init children
        new_node->AddChild(0, Index::VPNodePtr(new Index::VPNodeType));
        new_node->AddChild(0, Index::VPNodePtr(new Index::VPNodeType));

        float median = Median(new_node->get_value(), objects.begin(), objects.end());
        new_node->m_mu_list[0] = median;

        size_t objects_count = objects.size();
        if(median == 0)
            objects_count = 0;

        bool c_left = false;

        // 60% of size
        size_t reserved_memory = static_cast<size_t>(static_cast<double>(objects_count) * 0.6);
        std::vector<unsigned> s_left, s_right;
        s_left.reserve(reserved_memory);
        s_right.reserve(reserved_memory);

        typename std::vector<unsigned>::const_iterator it_obj = objects.begin();
        while(it_obj != objects.end())
        {
            float dist = index->getDist()->compare(index->getBaseData() + index->getBaseDim() * new_node->get_value(),
                                                   index->getBaseData() + index->getBaseDim() * (*it_obj),
                                                   index->getBaseDim());
            if(dist < new_node->m_mu_list[0] || (dist == 0  && !c_left))
            {
                s_left.push_back(*it_obj);
                c_left = true;
            }else
            {
                s_right.push_back(*it_obj);
                c_left = false;
            }
            ++it_obj;
        }

        size_t left_count = s_left.size();
        size_t right_count = s_right.size();

        // 8( for 2 only now
        new_node->set_branches_count(2);

        Index::VPNodePtr new_node_l(new Index::VPNodeType);
        Index::VPNodePtr new_node_r(new Index::VPNodeType);

        Index::VPNodePtr new_node_lc(new Index::VPNodeType);
        Index::VPNodePtr new_node_rc(new Index::VPNodeType);

#pragma omp task shared(new_node)
        new_node->m_child_list[0] = MakeVPTree(s_left, new_node);
#pragma omp task shared(new_node)
        new_node->m_child_list[1] = MakeVPTree(s_right, new_node);

#pragma omp taskwait

        return new_node;
    }

    // 3.b. If Parent-Parent node is not full, spleat non-leaf node
    void ComponentInitANNG::SplitNonLeafNode(const Index::VPNodePtr& parent_node, const size_t child_id, const unsigned& new_value)
    {
        //std::cout << "	split node" << std::endl;
        assert(child_id < parent_node->m_child_list.size());

        const size_t k = child_id;
        const size_t F = parent_node->get_branches_count();

        Index::VPNodePtr c_split_node = parent_node->m_child_list[child_id];
        Index::VPNodePtr c_node;

        std::vector<unsigned> S; // Set of leaf objects + new one;

        // Create Set of whole objects from leaf nodes at sub tree
        CollectObjects(c_split_node, S);
        S.push_back(new_value);

        // Order the objects in S with respect to their distances from P's vantage point
        unsigned m_main_val = parent_node->get_value();
        Index *tmp_index = index;
        //Index::ValueSorterType val_sorter(parent_node->get_value(), m_get_distance);
        std::sort(S.begin(), S.end(), [m_main_val, tmp_index](unsigned val1, unsigned val2) {
            float dist1 = tmp_index->getDist()->compare(tmp_index->getBaseData() + tmp_index->getBaseDim() * m_main_val,
                                                        tmp_index->getBaseData() + tmp_index->getBaseDim() * val1,
                                                        tmp_index->getBaseDim());
            float dist2 = tmp_index->getDist()->compare(tmp_index->getBaseData() + tmp_index->getBaseDim() * m_main_val,
                                                        tmp_index->getBaseData() + tmp_index->getBaseDim() * val2,
                                                        tmp_index->getBaseDim());
            return dist1 < dist2;
        });

        // Create free room for instance
        Index::VPNodePtr s2_node(new Index::VPNodeType);
        s2_node->set_parent(parent_node);
        parent_node->AddChild(0, s2_node);

        std::vector<unsigned> SS1(S.begin(), S.begin() + S.size() / 2);
        std::vector<unsigned> SS2(S.begin() + S.size() / 2, S.end());


        // Shift data at parent node

        if(F > 1)
            for(size_t c_pos = F-2; c_pos >= k; --c_pos)
            {
                parent_node->m_mu_list[c_pos+1] = parent_node->m_mu_list[c_pos];
                if(!c_pos) // !!! hack :(
                    break;
            }

        parent_node->m_mu_list[k] = MedianSumm(SS1, SS2, parent_node->get_value());
        for(size_t c_pos = F-1; c_pos >= k+1; --c_pos)
            parent_node->m_child_list[c_pos + 1] = parent_node->m_child_list[c_pos];


        // Construct new vp-tree
        Index::VPNodePtr ss1_node;
        Index::VPNodePtr ss2_node;

#pragma omp task shared(ss1_node)
        ss1_node = MakeVPTree(SS1, ss1_node);
#pragma omp task shared(ss2_node)
        ss2_node = MakeVPTree(SS2, ss2_node);

#pragma omp taskwait

        ss1_node->set_parent(parent_node);
        ss2_node->set_parent(parent_node);

        parent_node->m_child_list[k] = ss1_node;
        parent_node->m_child_list[k+1] = ss2_node;
    }

    void ComponentInitANNG::Insert(const unsigned& new_value, Index::VPNodePtr& root) {
        assert(root.get());
        // 4.  L is the empty root Node
        if(!root->get_branches_count() && !root->get_leaf_node()){
            // At first insertion make root as a leaf node
            //InsertSplitRoot(new_value)
            root->AddObject(new_value);
        }else {
            // Traverse the tree, choosing the subtree Si, until the L(eaf) node
            // is found

            size_t c_current_node_parent_id = 0;
            Index::VPNodePtr c_parent_node;
            // Go through the tree, searching leaf node
            Index::VPNodePtr c_current_node = c_parent_node = root;
            while (!c_current_node->get_leaf_node() && c_current_node.get()) {
                c_parent_node = c_current_node;
                // test all distances at node
                for (size_t c_pos = 0; c_pos < c_current_node->m_mu_list.size(); ++c_pos) {
                    // test new_value with node vantage point
                    float dist = index->getDist()->compare(index->getBaseData() + index->getBaseDim() * new_value,
                                                           index->getBaseData() + index->getBaseDim() * c_current_node->get_value(),
                                                           index->getBaseDim());
                    if (dist < c_current_node->m_mu_list[c_pos]) {
                        c_current_node = c_current_node->m_child_list[c_pos];
                        c_current_node_parent_id = c_pos;
                        break;
                    }
                }

                if (c_parent_node == c_current_node)
                    c_current_node = *c_current_node->m_child_list.rbegin();
            }

            // Assume c_current_node - Leaf node
            // Have found leaf node, analize ancestros

            // 0. If there is a room at L(eaf) node - insert data
            if (c_current_node->get_objects_count() < index->vp_tree.m_leaf_branching_factor) {
                c_current_node->AddObject(new_value);
                return;
            }

            // Second node - we split the root
            if (c_current_node == root &&
                c_current_node->get_objects_count() >= index->vp_tree.m_leaf_branching_factor) {
                InsertSplitLeafRoot(c_current_node, new_value);
                return;
            }

            // 1. If any sibling leaf node of L(eaf) is not full,
            // redistribute all objects under P(arent), among the leaf nodes
            // Analize sibling nodes
            for (size_t c_pos = 0; c_pos < c_parent_node->get_branches_count(); ++c_pos) {
                if (c_parent_node->m_child_list[c_pos]->get_objects_count() < index->vp_tree.m_leaf_branching_factor) {
                    RedistributeAmongLeafNodes(c_parent_node, new_value);
                    return;
                }
            }


            // 2. If Parent has a room for one more child - split the leaf node
            if (c_parent_node->get_branches_count() < index->vp_tree.m_non_leaf_branching_factor) {
                SplitLeafNode(c_parent_node, c_current_node_parent_id, new_value);
                return;
            }

            // 3.a. Redistribute, among the sibling subtrees
            Index::VPNodePtr c_ancestor = c_parent_node->m_parent;
            if (c_ancestor.get()) {
                // found an id of full leaf node parent
                size_t c_found_free_subtree_id = index->vp_tree.m_leaf_branching_factor;
                size_t c_full_subtree_id = index->vp_tree.m_leaf_branching_factor;
                for (size_t c_anc_pos = 0; c_anc_pos < c_ancestor->get_branches_count(); ++c_anc_pos)
                    if (c_ancestor->m_child_list[c_anc_pos] == c_current_node->m_parent)
                        c_full_subtree_id = c_anc_pos;

                //assert(c_full_subtree_id != m_leaf_branching_factor);

                if (c_full_subtree_id != index->vp_tree.m_leaf_branching_factor)
                    for (size_t c_anc_pos = 0; c_anc_pos < c_ancestor->get_branches_count(); ++c_anc_pos) {
                        Index::VPNodePtr c_parent = c_ancestor->m_child_list[c_anc_pos];

                        if (c_parent == c_current_node->m_parent)
                            continue;

                        for (size_t c_par_pos = 0; c_par_pos < c_parent->get_branches_count(); ++c_par_pos) {
                            if (c_parent->m_child_list[c_par_pos]->get_leaf_node() &&
                                c_parent->m_child_list[c_par_pos]->m_objects_list.size() < index->vp_tree.m_leaf_branching_factor) {
                                c_found_free_subtree_id = c_anc_pos;
                                break;
                            }
                        }
                        if (c_found_free_subtree_id < index->vp_tree.m_leaf_branching_factor) {
                            // Found free subtree - redistribute data
                            if (c_found_free_subtree_id > c_full_subtree_id)
                                RedistributeAmongNonLeafNodes(c_ancestor, c_full_subtree_id, c_found_free_subtree_id,
                                                              new_value);
                            else
                                RedistributeAmongNonLeafNodes(c_ancestor, c_found_free_subtree_id, c_full_subtree_id,
                                                              new_value);

                            Insert(new_value, c_ancestor);

                            return;
                        }
                    }
            }

            // 3.b. If Parent-Parent node is not full, spleat non-leaf node
            if (c_current_node->m_parent.get() && c_current_node->m_parent->m_parent.get()) {
                Index::VPNodePtr c_ancestor = c_current_node->m_parent->m_parent; // A
                Index::VPNodePtr c_current_parent = c_current_node->m_parent; // B

                size_t c_found_free_subtree_id = index->vp_tree.m_non_leaf_branching_factor;
                for (size_t c_pos = 0; c_pos < c_ancestor->get_branches_count(); ++c_pos)
                    if (c_ancestor->m_child_list[c_pos] == c_current_parent)
                        c_found_free_subtree_id = c_pos;

                if (c_found_free_subtree_id != index->vp_tree.m_non_leaf_branching_factor &&
                    c_ancestor->get_branches_count() < index->vp_tree.m_non_leaf_branching_factor) {
                    SplitNonLeafNode(c_ancestor, c_found_free_subtree_id, new_value);
                    return;
                }
            }

            // 4. Cannot find any ancestor, that is not full -> split the root
            // into two new nodes s1 and s2
            InsertSplitRoot(root, new_value);
        }
    }

    void ComponentInitANNG::Insert(const unsigned int &new_value) {
        //std::cout << "Insert data to vptree.root" << std::endl;

        Insert(new_value, index->vp_tree.m_root);
    }


    // SPTAG KDT
    void ComponentInitSPTAG_KDT::InitInner() {
        SetConfigs();

        BuildTrees();

        BuildGraph();

        // for(int i = 0; i < 10; i ++) {
        //     std::cout << "len : " << index->getFinalGraph()[i].size() << std::endl;
        //     for(int j = 0; j < index->getFinalGraph()[i].size(); j ++){
        //         std::cout << index->getFinalGraph()[i][j].id << "|" << index->getFinalGraph()[i][j].distance << " ";
        //     }
        //     std::cout << std::endl;
        // }
    }

    void ComponentInitSPTAG_KDT::SetConfigs() {
        index->numOfThreads = index->getParam().get<unsigned>("numOfThreads");

        index->m_iTreeNumber = index->getParam().get<unsigned>("KDTNumber");

        index->m_iTPTNumber = index->getParam().get<unsigned>("TPTNumber");

        index->m_iTPTLeafSize = index->getParam().get<unsigned>("TPTLeafSize");

        index->m_iNeighborhoodSize = index->getParam().get<unsigned>("NeighborhoodSize");

        index->m_iNeighborhoodScale = index->getParam().get<unsigned>("GraphNeighborhoodScale");

        index->m_iCEF = index->getParam().get<unsigned>("CEF");
    }

    void ComponentInitSPTAG_KDT::BuildTrees() {
        std::vector<int> localindices;

        localindices.resize(index->getBaseLen());
        for (int i = 0; i < localindices.size(); i++) localindices[i] = i;

        index->m_pKDTreeRoots.resize(index->m_iTreeNumber * localindices.size());
        index->m_pTreeStart.resize(index->m_iTreeNumber, 0);
#pragma omp parallel for num_threads(index->numOfThreads)
        for (int i = 0; i < index->m_iTreeNumber; i++) {
            // Sleep(i * 100);
            std::srand(clock());

            std::vector<int> pindices(localindices.begin(), localindices.end());
            std::random_shuffle(pindices.begin(), pindices.end());

            index->m_pTreeStart[i] = i * pindices.size();
            // std::cout << "Start to build KDTree " << i + 1 << std::endl;
            int iTreeSize = index->m_pTreeStart[i];

            DivideTree(pindices, 0, pindices.size() - 1, index->m_pTreeStart[i], iTreeSize);
            // std::cout << i + 1 << " KDTree built, " << iTreeSize - index->m_pTreeStart[i] << " " << pindices.size();
        }
    }

    int ComponentInitSPTAG_KDT::rand(int high, int low) {
        return low + (int) (float(high - low) * (std::rand() / (RAND_MAX + 1.0)));
    }

    int ComponentInitSPTAG_KDT::SelectDivisionDimension(const std::vector<float> &varianceValues) {
        // Record the top maximum variances
        std::vector<int> topind(index->m_numTopDimensionKDTSplit);
        int num = 0;
        // order the variances
        for (int i = 0; i < (int) varianceValues.size(); i++) {
            if (num < index->m_numTopDimensionKDTSplit || varianceValues[i] > varianceValues[topind[num - 1]]) {
                if (num < index->m_numTopDimensionKDTSplit) {
                    topind[num++] = i;
                } else {
                    topind[num - 1] = i;
                }
                int j = num - 1;
                // order the TOP_DIM variances
                while (j > 0 && varianceValues[topind[j]] > varianceValues[topind[j - 1]]) {
                    std::swap(topind[j], topind[j - 1]);
                    j--;
                }
            }
        }
        // randomly choose a dimension from TOP_DIM
        return topind[ComponentInitSPTAG_KDT::rand(num)];
    }

    void ComponentInitSPTAG_KDT::ChooseDivision(Index::KDTNode &node, const std::vector<int> &indices,
                                                const int first, const int last) {
        std::vector<float> meanValues(index->getBaseDim(), 0);
        std::vector<float> varianceValues(index->getBaseDim(), 0);
        int end = std::min(first + index->m_iSamples, last);
        int count = end - first + 1;
        // calculate the mean of each dimension
        for (int j = first; j <= end; j++) {
            const float *v = index->getBaseData() + index->getBaseDim() * indices[j];
            for (int k = 0; k < index->getBaseDim(); k++) {
                meanValues[k] += v[k];
            }
        }
        for (int k = 0; k < index->getBaseDim(); k++) {
            meanValues[k] /= count;
        }
        // calculate the variance of each dimension
        for (int j = first; j <= end; j++) {
            const float *v = index->getBaseData() + index->getBaseDim() * indices[j];
            for (int k = 0; k < index->getBaseDim(); k++) {
                float dist = v[k] - meanValues[k];
                varianceValues[k] += dist * dist;
            }
        }
        // choose the split dimension as one of the dimension inside TOP_DIM maximum variance
        node.split_dim = SelectDivisionDimension(varianceValues);
        // determine the threshold
        node.split_value = meanValues[node.split_dim];
    }

    int ComponentInitSPTAG_KDT::Subdivide(const Index::KDTNode &node, std::vector<int> &indices,
                                          const int first, const int last) {
        int i = first;
        int j = last;
        // decide which child one point belongs
        while (i <= j) {
            int ind = indices[i];
            const float *v = index->getBaseData() + index->getBaseDim() * ind;
            float val = v[node.split_dim];
            if (val < node.split_value) {
                i++;
            } else {
                std::swap(indices[i], indices[j]);
                j--;
            }
        }
        // if all the points in the node are equal,equally split the node into 2
        if ((i == first) || (i == last + 1)) {
            i = (first + last + 1) / 2;
        }
        return i;
    }

    void ComponentInitSPTAG_KDT::DivideTree(std::vector<int> &indices, int first,
                                            int last, int tree_index, int &iTreeSize) {
        ChooseDivision(index->m_pKDTreeRoots[tree_index], indices, first, last);
        int i = Subdivide(index->m_pKDTreeRoots[tree_index], indices, first, last);
        if (i - 1 <= first) {
            index->m_pKDTreeRoots[tree_index].left = -indices[first] - 1;
        } else {
            iTreeSize++;
            index->m_pKDTreeRoots[tree_index].left = iTreeSize;
            DivideTree(indices, first, i - 1, iTreeSize, iTreeSize);
        }
        if (last == i) {
            index->m_pKDTreeRoots[tree_index].right = -indices[last] - 1;
        } else {
            iTreeSize++;
            index->m_pKDTreeRoots[tree_index].right = iTreeSize;
            DivideTree(indices, i, last, iTreeSize, iTreeSize);
        }
    }

    bool ComponentInitSPTAG_KDT::Compare(const Index::SimpleNeighbor &lhs, const Index::SimpleNeighbor &rhs) {
        return ((lhs.distance < rhs.distance) || ((lhs.distance == rhs.distance) && (lhs.id < rhs.id)));
    }

    void ComponentInitSPTAG_KDT::PartitionByTptree(std::vector<int> &indices, const int first,
                                                   const int last,
                                                   std::vector<std::pair<int, int>> &leaves) {
        if (last - first <= index->m_iTPTLeafSize) {
            leaves.emplace_back(first, last);
        } else {
            std::vector<float> Mean(index->getBaseDim(), 0);

            int iIteration = 100;
            int end = std::min(first + index->m_iSamples, last);
            int count = end - first + 1;
            // calculate the mean of each dimension
            for (int j = first; j <= end; j++) {
                const float *v = index->getBaseData() + index->getBaseDim() * indices[j];
                for (int k = 0; k < index->getBaseDim(); k++) {
                    Mean[k] += v[k];
                }
            }
            for (int k = 0; k < index->getBaseDim(); k++) {
                Mean[k] /= count;
            }
            std::vector<Index::SimpleNeighbor> Variance;
            Variance.reserve(index->getBaseDim());
            for (int j = 0; j < index->getBaseDim(); j++) {
                Variance.emplace_back(j, 0.0f);
            }
            // calculate the variance of each dimension
            for (int j = first; j <= end; j++) {
                const float *v = index->getBaseData() + index->getBaseDim() * indices[j];
                for (int k = 0; k < index->getBaseDim(); k++) {
                    float dist = v[k] - Mean[k];
                    Variance[k].distance += dist * dist;
                }
            }
            std::sort(Variance.begin(), Variance.end(), ComponentInitSPTAG_KDT::Compare);
            std::vector<int> indexs(index->m_numTopDimensionTPTSplit);
            std::vector<float> weight(index->m_numTopDimensionTPTSplit), bestweight(index->m_numTopDimensionTPTSplit);
            float bestvariance = Variance[index->getBaseDim() - 1].distance;
            for (int i = 0; i < index->m_numTopDimensionTPTSplit; i++) {
                indexs[i] = Variance[index->getBaseDim() - 1 - i].id;
                bestweight[i] = 0;
            }
            bestweight[0] = 1;
            float bestmean = Mean[indexs[0]];

            std::vector<float> Val(count);
            for (int i = 0; i < iIteration; i++) {
                float sumweight = 0;
                for (int j = 0; j < index->m_numTopDimensionTPTSplit; j++) {
                    weight[j] = float(std::rand() % 10000) / 5000.0f - 1.0f;
                    sumweight += weight[j] * weight[j];
                }
                sumweight = sqrt(sumweight);
                for (int j = 0; j < index->m_numTopDimensionTPTSplit; j++) {
                    weight[j] /= sumweight;
                }
                float mean = 0;
                for (int j = 0; j < count; j++) {
                    Val[j] = 0;
                    const float *v = index->getBaseData() + index->getBaseDim() * indices[first + j];
                    for (int k = 0; k < index->m_numTopDimensionTPTSplit; k++) {
                        Val[j] += weight[k] * v[indexs[k]];
                    }
                    mean += Val[j];
                }
                mean /= count;
                float var = 0;
                for (int j = 0; j < count; j++) {
                    float dist = Val[j] - mean;
                    var += dist * dist;
                }
                if (var > bestvariance) {
                    bestvariance = var;
                    bestmean = mean;
                    for (int j = 0; j < index->m_numTopDimensionTPTSplit; j++) {
                        bestweight[j] = weight[j];
                    }
                }
            }
            int i = first;
            int j = last;
            // decide which child one point belongs
            while (i <= j) {
                float val = 0;
                const float *v = index->getBaseData() + index->getBaseDim() * indices[i];
                for (int k = 0; k < index->m_numTopDimensionTPTSplit; k++) {
                    val += bestweight[k] * v[indexs[k]];
                }
                if (val < bestmean) {
                    i++;
                } else {
                    std::swap(indices[i], indices[j]);
                    j--;
                }
            }
            // if all the points in the node are equal,equally split the node into 2
            if ((i == first) || (i == last + 1)) {
                i = (first + last + 1) / 2;
            }

            Mean.clear();
            Variance.clear();
            Val.clear();
            indexs.clear();
            weight.clear();
            bestweight.clear();

            PartitionByTptree(indices, first, i - 1, leaves);
            PartitionByTptree(indices, i, last, leaves);
        }
    }

    void ComponentInitSPTAG_KDT::AddNeighbor(int idx, float dist, int origin) {
        int size = index->m_iNeighborhoodSize - 1;

        if (dist < index->getFinalGraph()[origin][size].distance ||
            (dist == index->getFinalGraph()[origin][size].distance && idx < index->getFinalGraph()[origin][size].id)) {
            int nb;

            for (nb = 0; nb <= size && index->getFinalGraph()[origin][nb].id != idx; nb++);

            if (nb > size) {
                nb = size;
                while (nb > 0 && (dist < index->getFinalGraph()[origin][nb - 1].distance ||
                                  (dist == index->getFinalGraph()[origin][nb - 1].distance &&
                                   idx < index->getFinalGraph()[origin][nb - 1].id))) {
                    index->getFinalGraph()[origin][nb] = index->getFinalGraph()[origin][nb - 1];
                    nb--;
                }
                index->getFinalGraph()[origin][nb].distance = dist;
                index->getFinalGraph()[origin][nb].id = idx;
            }
        }
    }

    void ComponentInitSPTAG_KDT::BuildGraph() {
        index->m_iNeighborhoodSize = index->m_iNeighborhoodSize * index->m_iNeighborhoodScale;

        index->getFinalGraph().resize(index->getBaseLen());

        float MaxDist = (std::numeric_limits<float>::max)();
        int MaxId = (std::numeric_limits<int>::max)();

        for (int i = 0; i < index->getBaseLen(); i++) {
            index->getFinalGraph()[i].resize(index->m_iNeighborhoodSize);
            for (int j = 0; j < index->m_iNeighborhoodSize; j++) {
                Index::SimpleNeighbor neighbor(MaxId, MaxDist);
                index->getFinalGraph()[i][j] = neighbor;
            }
        }

        std::vector<std::vector<int>> TptreeDataIndices(index->m_iTPTNumber, std::vector<int>(index->getBaseLen()));
        std::vector<std::vector<std::pair<int, int>>> TptreeLeafNodes(index->m_iTPTNumber,
                                                                      std::vector<std::pair<int, int>>());

        auto t1 = std::chrono::high_resolution_clock::now();
        // std::cout << "Parallel TpTree Partition begin\n";
#pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < index->m_iTPTNumber; i++) {
            // Sleep(i * 100);
            std::srand(clock());
            for (int j = 0; j < index->getBaseLen(); j++) TptreeDataIndices[i][j] = j;
            std::random_shuffle(TptreeDataIndices[i].begin(), TptreeDataIndices[i].end());
            PartitionByTptree(TptreeDataIndices[i], 0, index->getBaseLen() - 1, TptreeLeafNodes[i]);
            // std::cout << "Finish Getting Leaves for Tree " << i << std::endl;
        }
        // std::cout << "Parallel TpTree Partition done\n";
        auto t2 = std::chrono::high_resolution_clock::now();
        // std::cout << "Build TPTree time (s): " << std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count()
                //   << std::endl;

        for (int i = 0; i < index->m_iTPTNumber; i++) {
#pragma omp parallel for schedule(dynamic)
            for (int j = 0; j < (int) TptreeLeafNodes[i].size(); j++) {
                int start_index = TptreeLeafNodes[i][j].first;
                int end_index = TptreeLeafNodes[i][j].second;
                if ((j * 5) % TptreeLeafNodes[i].size() == 0)
                    // std::cout << "Processing Tree " << i << " "
                            //   << static_cast<int>(j * 1.0 / TptreeLeafNodes[i].size() * 100) << std::endl;
                for (int x = start_index; x < end_index; x++) {
                    for (int y = x + 1; y <= end_index; y++) {
                        int p1 = TptreeDataIndices[i][x];
                        int p2 = TptreeDataIndices[i][y];
                        float dist = index->getDist()->compare(index->getBaseData() + index->getBaseDim() * p1,
                                                               index->getBaseData() + index->getBaseDim() * p2,
                                                               index->getBaseDim());

                        AddNeighbor(p2, dist, p1);
                        AddNeighbor(p1, dist, p2);
                    }
                }
            }
            TptreeDataIndices[i].clear();
            TptreeLeafNodes[i].clear();
        }
        TptreeDataIndices.clear();
        TptreeLeafNodes.clear();

        auto t3 = std::chrono::high_resolution_clock::now();
        // std::cout << "Process TPTree time (s): " << std::chrono::duration_cast<std::chrono::seconds>(t3 - t2).count()
                //   << std::endl;
    }



    /**
     * SPTAG_BKT
     */
    void ComponentInitSPTAG_BKT::InitInner() {
        SetConfigs();

        BuildTrees();

        BuildGraph();

//        for(int i = 0; i < 50; i ++) {
//            std::cout << "len : " << index->getFinalGraph()[i].size() << std::endl;
//            for(int j = 0; j < index->getFinalGraph()[i].size(); j ++){
//                std::cout << index->getFinalGraph()[i][j].id << "|" << index->getFinalGraph()[i][j].distance << " ";
//            }
//            std::cout << std::endl;
//        }
    }

    void ComponentInitSPTAG_BKT::SetConfigs() {
        index->numOfThreads = index->getParam().get<unsigned>("numOfThreads");

        index->m_iTreeNumber = index->getParam().get<unsigned>("BKTNumber");

        index->m_iBKTKmeansK = index->getParam().get<unsigned>("BKTKMeansK");

        index->m_iTPTNumber = index->getParam().get<unsigned>("TPTNumber");

        index->m_iTPTLeafSize = index->getParam().get<unsigned>("TPTLeafSize");

        index->m_iNeighborhoodSize = index->getParam().get<unsigned>("NeighborhoodSize");

        index->m_iNeighborhoodScale = index->getParam().get<unsigned>("GraphNeighborhoodScale");

        index->m_iCEF = index->getParam().get<unsigned>("CEF");
    }

    int ComponentInitSPTAG_BKT::rand(int high, int low) {
        return low + (int) (float(high - low) * (std::rand() / (RAND_MAX + 1.0)));
    }

    float ComponentInitSPTAG_BKT::KmeansAssign(std::vector<int> &indices, const int first,
                                               const int last, Index::KmeansArgs<float> &args,
                                               const bool updateCenters, float lambda) {
        float currDist = 0;
        int subsize = (last - first - 1) / args._T + 1;
        const float MaxDist = (std::numeric_limits<float>::max)();

//#pragma omp parallel for num_threads(args._T) shared(index->getBaseData(), indices) reduction(+:currDist)
        for (int tid = 0; tid < args._T; tid++) {
            int istart = first + tid * subsize;
            int iend = std::min(first + (tid + 1) * subsize, last);
            int *inewCounts = args.newCounts + tid * args._K;
            float *inewCenters = args.newCenters + tid * args._K * args._D;
            int *iclusterIdx = args.clusterIdx + tid * args._K;
            float *iclusterDist = args.clusterDist + tid * args._K;
            float idist = 0;
            for (int i = istart; i < iend; i++) {
                int clusterid = 0;
                float smallestDist = MaxDist;
                for (int k = 0; k < args._DK; k++) {
                    float dist = index->getDist()->compare(index->getBaseData() + index->getBaseDim() * indices[i],
                                                           args.centers + k * args._D, args._D) +
                                 lambda * args.counts[k];
                    if (dist > -MaxDist && dist < smallestDist) {
                        clusterid = k;
                        smallestDist = dist;
                    }
                }
                args.label[i] = clusterid;
                inewCounts[clusterid]++;
                idist += smallestDist;
                if (updateCenters) {
                    const float *v = index->getBaseData() + index->getBaseDim() * indices[i];
                    float *center = inewCenters + clusterid * args._D;
                    for (int j = 0; j < args._D; j++) center[j] += v[j];
                    if (smallestDist > iclusterDist[clusterid]) {
                        iclusterDist[clusterid] = smallestDist;
                        iclusterIdx[clusterid] = indices[i];
                    }
                } else {
                    if (smallestDist <= iclusterDist[clusterid]) {
                        iclusterDist[clusterid] = smallestDist;
                        iclusterIdx[clusterid] = indices[i];
                    }
                }
            }
            currDist += idist;
        }

        for (int i = 1; i < args._T; i++) {
            for (int k = 0; k < args._DK; k++)
                args.newCounts[k] += args.newCounts[i * args._K + k];
        }

        if (updateCenters) {
            for (int i = 1; i < args._T; i++) {
                float *currCenter = args.newCenters + i * args._K * args._D;
                for (size_t j = 0; j < ((size_t) args._DK) * args._D; j++) args.newCenters[j] += currCenter[j];

                for (int k = 0; k < args._DK; k++) {
                    if (args.clusterIdx[i * args._K + k] != -1 &&
                        args.clusterDist[i * args._K + k] > args.clusterDist[k]) {
                        args.clusterDist[k] = args.clusterDist[i * args._K + k];
                        args.clusterIdx[k] = args.clusterIdx[i * args._K + k];
                    }
                }
            }
        } else {
            for (int i = 1; i < args._T; i++) {
                for (int k = 0; k < args._DK; k++) {
                    if (args.clusterIdx[i * args._K + k] != -1 &&
                        args.clusterDist[i * args._K + k] <= args.clusterDist[k]) {
                        args.clusterDist[k] = args.clusterDist[i * args._K + k];
                        args.clusterIdx[k] = args.clusterIdx[i * args._K + k];
                    }
                }
            }
        }

        return currDist;
    }

    void ComponentInitSPTAG_BKT::InitCenters(std::vector<int> &indices, const int first,
                                             const int last, Index::KmeansArgs<float> &args, int samples,
                                             int tryIters) {
        const float MaxDist = (std::numeric_limits<float>::max)();

        int batchEnd = std::min(first + samples, last);
        float currDist, minClusterDist = MaxDist;
        for (int numKmeans = 0; numKmeans < tryIters; numKmeans++) {
            for (int k = 0; k < args._DK; k++) {
                int randid = ComponentInitSPTAG_BKT::rand(last, first);
                std::memcpy(args.centers + k * args._D, index->getBaseData() + index->getBaseDim() * indices[randid],
                            sizeof(float) * args._D);
            }
            args.ClearCounts();
            args.ClearDists(MaxDist);
            currDist = KmeansAssign(indices, first, batchEnd, args, false, 0);
            if (currDist < minClusterDist) {
                minClusterDist = currDist;
                memcpy(args.newTCenters, args.centers, sizeof(float) * args._K * args._D);
                memcpy(args.counts, args.newCounts, sizeof(int) * args._K);
            }
        }
    }

    float ComponentInitSPTAG_BKT::RefineCenters(Index::KmeansArgs<float> &args) {
        //std::cout << "RefineCenters" << std::endl;
//        for(int i = 0; i < args._DK; i ++)
//            std::cout << args.clusterIdx[i] << std::endl;
        int maxcluster = -1;
        int maxCount = 0;
        for (int k = 0; k < args._DK; k++) {
            if (args.clusterIdx[k] == -1) continue;
            float dist = index->getDist()->compare(index->getBaseData() + index->getBaseDim() * args.clusterIdx[k],
                                                   args.centers + k * args._D,
                                                   args._D);
            if (args.counts[k] > maxCount && args.newCounts[k] > 0 && dist > 1e-6) {
                maxcluster = k;
                maxCount = args.counts[k];
            }
        }

        if (maxcluster != -1 && (args.clusterIdx[maxcluster] < 0 || args.clusterIdx[maxcluster] >= index->getBaseLen()))
            std::cout << "maxcluster:" << maxcluster << "(" << args.newCounts[maxcluster] << ") Error dist:"
                      << args.clusterDist[maxcluster] << std::endl;

        float diff = 0;
        for (int k = 0; k < args._DK; k++) {
            float *TCenter = args.newTCenters + k * args._D;
            if (args.counts[k] == 0) {
                if (maxcluster != -1) {
                    //int nextid = Utils::rand_int(last, first);
                    //while (args.label[nextid] != maxcluster) nextid = Utils::rand_int(last, first);
                    int nextid = args.clusterIdx[maxcluster];
                    std::memcpy(TCenter, index->getBaseData() + index->getBaseDim() * nextid, sizeof(float) * args._D);
                } else {
                    std::memcpy(TCenter, args.centers + k * args._D, sizeof(float) * args._D);
                }
            } else {
                float *currCenters = args.newCenters + k * args._D;
                for (unsigned j = 0; j < args._D; j++) currCenters[j] /= args.counts[k];

                for (unsigned j = 0; j < args._D; j++) TCenter[j] = (float) (currCenters[j]);
            }
            diff += (index->getDist()->compare(args.centers + k * args._D, TCenter, args._D));
        }

        //std::cout << "RefineCenters end" << std::endl;
        return diff;
    }

    int ComponentInitSPTAG_BKT::KmeansClustering(std::vector<int> &indices, const int first,
                                                 const int last, Index::KmeansArgs<float> &args,
                                                 int samples) {
        const float MaxDist = (std::numeric_limits<float>::max)();

        //std::cout << "KmeansClustering" << std::endl;

        InitCenters(indices, first, last, args, samples, 3);

        int batchEnd = std::min(first + samples, last);
        float currDiff, currDist, minClusterDist = MaxDist;
        int noImprovement = 0;
        for (int iter = 0; iter < 100; iter++) {
            //std::cout << iter << std::endl;
            std::memcpy(args.centers, args.newTCenters, sizeof(float) * args._K * args._D);
            std::random_shuffle(indices.begin() + first, indices.begin() + last);

            args.ClearCenters();
            args.ClearCounts();
            args.ClearDists(-MaxDist);
            currDist = KmeansAssign(indices, first, batchEnd, args, true, 1 / (100.0f * (batchEnd - first)));

            std::memcpy(args.counts, args.newCounts, sizeof(int) * args._K);

            if (currDist < minClusterDist) {
                noImprovement = 0;
                minClusterDist = currDist;
            } else {
                noImprovement++;
            }
            currDiff = RefineCenters(args);
            if (currDiff < 1e-3 || noImprovement >= 5) break;
        }

        //std::cout << -1.2 << std::endl;

        args.ClearCounts();
        args.ClearDists(MaxDist);
        currDist = KmeansAssign(indices, first, last, args, false, 0);
        std::memcpy(args.counts, args.newCounts, sizeof(int) * args._K);

        //std::cout << -1.3 << std::endl;

        int numClusters = 0;
        for (int i = 0; i < args._K; i++) if (args.counts[i] > 0) numClusters++;

        if (numClusters <= 1) {
            return numClusters;
        }
        args.Shuffle(indices, first, last);
        return numClusters;
    }

    void ComponentInitSPTAG_BKT::BuildTrees() {
        struct BKTStackItem {
            int bkt_index, first, last;

            BKTStackItem(int index_, int first_, int last_) : bkt_index(index_), first(first_), last(last_) {}
        };
        std::stack<BKTStackItem> ss;

        std::vector<int> localindices;
        localindices.resize(index->getBaseLen());
        for (int i = 0; i < localindices.size(); i++) localindices[i] = i;

        Index::KmeansArgs<float> args(index->m_iBKTKmeansK, index->getBaseDim(), (int) localindices.size(),
                                      index->numOfThreads);

        index->m_pSampleCenterMap.clear();
        for (char i = 0; i < index->m_iTreeNumber; i++) {
            std::random_shuffle(localindices.begin(), localindices.end());

            index->m_pTreeStart.push_back((int) index->m_pBKTreeRoots.size());
            index->m_pBKTreeRoots.emplace_back((int) localindices.size());

            // std::cout << "Start to build BKTree " << i + 1 << std::endl;

            ss.push(BKTStackItem(index->m_pTreeStart[i], 0, (int) localindices.size()));
            while (!ss.empty()) {
                //std::cout << 1 << std::endl;
                BKTStackItem item = ss.top();
                ss.pop();
                int newBKTid = (int) index->m_pBKTreeRoots.size();
                index->m_pBKTreeRoots[item.bkt_index].childStart = newBKTid;
                if (item.last - item.first <= index->m_iBKTLeafSize) {
                    for (int j = item.first; j < item.last; j++) {
                        int cid = localindices[j];
                        index->m_pBKTreeRoots.emplace_back(cid);
                    }
                } else { // clustering the data into BKTKmeansK clusters
                    //std::cout << 3 << std::endl;
                    int numClusters = KmeansClustering(localindices, item.first, item.last, args, index->m_iSamples);
                    //std::cout << 3.5 << std::endl;
                    if (numClusters <= 1) {
                        int end = std::min(item.last + 1, (int) localindices.size());
                        std::sort(localindices.begin() + item.first, localindices.begin() + end);
                        index->m_pBKTreeRoots[item.bkt_index].centerid = localindices[item.first];
                        index->m_pBKTreeRoots[item.bkt_index].childStart = -index->m_pBKTreeRoots[item.bkt_index].childStart;
                        for (int j = item.first + 1; j < end; j++) {
                            int cid = localindices[j];
                            index->m_pBKTreeRoots.emplace_back(cid);
                            index->m_pSampleCenterMap[cid] = index->m_pBKTreeRoots[item.bkt_index].centerid;
                        }
                        index->m_pSampleCenterMap[-1 - index->m_pBKTreeRoots[item.bkt_index].centerid] = item.bkt_index;
                    } else {
                        for (int k = 0; k < index->m_iBKTKmeansK; k++) {
                            if (args.counts[k] == 0) continue;
                            int cid = localindices[item.first + args.counts[k] - 1];
                            index->m_pBKTreeRoots.emplace_back(cid);
                            if (args.counts[k] > 1)
                                ss.push(BKTStackItem(newBKTid++, item.first, item.first + args.counts[k] - 1));
                            item.first += args.counts[k];
                        }
                    }
                    //std::cout << 3.55 << std::endl;
                }
                //std::cout << 4 << std::endl;
                index->m_pBKTreeRoots[item.bkt_index].childEnd = (int) index->m_pBKTreeRoots.size();
            }
            //std::cout << 2 << std::endl;
            index->m_pBKTreeRoots.emplace_back(-1);
            //std::cout << i + 1 << " BKTree built, " << index->m_pBKTreeRoots.size() - index->m_pTreeStart[i] << localindices.size() << std::endl;
        }
    }

    bool ComponentInitSPTAG_BKT::Compare(const Index::SimpleNeighbor &lhs, const Index::SimpleNeighbor &rhs) {
        return ((lhs.distance < rhs.distance) || ((lhs.distance == rhs.distance) && (lhs.id < rhs.id)));
    }

    void ComponentInitSPTAG_BKT::PartitionByTptree(std::vector<int> &indices, const int first,
                                                   const int last,
                                                   std::vector<std::pair<int, int>> &leaves) {
        if (last - first <= index->m_iTPTLeafSize) {
            leaves.emplace_back(first, last);
        } else {
            std::vector<float> Mean(index->getBaseDim(), 0);

            int iIteration = 100;
            int end = std::min(first + index->m_iSamples, last);
            int count = end - first + 1;
            // calculate the mean of each dimension
            for (int j = first; j <= end; j++) {
                const float *v = index->getBaseData() + index->getBaseDim() * indices[j];
                for (int k = 0; k < index->getBaseDim(); k++) {
                    Mean[k] += v[k];
                }
            }
            for (int k = 0; k < index->getBaseDim(); k++) {
                Mean[k] /= count;
            }
            std::vector<Index::SimpleNeighbor> Variance;
            Variance.reserve(index->getBaseDim());
            for (int j = 0; j < index->getBaseDim(); j++) {
                Variance.emplace_back(j, 0.0f);
            }
            // calculate the variance of each dimension
            for (int j = first; j <= end; j++) {
                const float *v = index->getBaseData() + index->getBaseDim() * indices[j];
                for (int k = 0; k < index->getBaseDim(); k++) {
                    float dist = v[k] - Mean[k];
                    Variance[k].distance += dist * dist;
                }
            }
            std::sort(Variance.begin(), Variance.end(), ComponentInitSPTAG_BKT::Compare);
            std::vector<int> indexs(index->m_numTopDimensionTPTSplit);
            std::vector<float> weight(index->m_numTopDimensionTPTSplit), bestweight(index->m_numTopDimensionTPTSplit);
            float bestvariance = Variance[index->getBaseDim() - 1].distance;
            for (int i = 0; i < index->m_numTopDimensionTPTSplit; i++) {
                indexs[i] = Variance[index->getBaseDim() - 1 - i].id;
                bestweight[i] = 0;
            }
            bestweight[0] = 1;
            float bestmean = Mean[indexs[0]];

            std::vector<float> Val(count);
            for (int i = 0; i < iIteration; i++) {
                float sumweight = 0;
                for (int j = 0; j < index->m_numTopDimensionTPTSplit; j++) {
                    weight[j] = float(std::rand() % 10000) / 5000.0f - 1.0f;
                    sumweight += weight[j] * weight[j];
                }
                sumweight = sqrt(sumweight);
                for (int j = 0; j < index->m_numTopDimensionTPTSplit; j++) {
                    weight[j] /= sumweight;
                }
                float mean = 0;
                for (int j = 0; j < count; j++) {
                    Val[j] = 0;
                    const float *v = index->getBaseData() + index->getBaseDim() * indices[first + j];
                    for (int k = 0; k < index->m_numTopDimensionTPTSplit; k++) {
                        Val[j] += weight[k] * v[indexs[k]];
                    }
                    mean += Val[j];
                }
                mean /= count;
                float var = 0;
                for (int j = 0; j < count; j++) {
                    float dist = Val[j] - mean;
                    var += dist * dist;
                }
                if (var > bestvariance) {
                    bestvariance = var;
                    bestmean = mean;
                    for (int j = 0; j < index->m_numTopDimensionTPTSplit; j++) {
                        bestweight[j] = weight[j];
                    }
                }
            }
            int i = first;
            int j = last;
            // decide which child one point belongs
            while (i <= j) {
                float val = 0;
                const float *v = index->getBaseData() + index->getBaseDim() * indices[i];
                for (int k = 0; k < index->m_numTopDimensionTPTSplit; k++) {
                    val += bestweight[k] * v[indexs[k]];
                }
                if (val < bestmean) {
                    i++;
                } else {
                    std::swap(indices[i], indices[j]);
                    j--;
                }
            }
            // if all the points in the node are equal,equally split the node into 2
            if ((i == first) || (i == last + 1)) {
                i = (first + last + 1) / 2;
            }

            Mean.clear();
            Variance.clear();
            Val.clear();
            indexs.clear();
            weight.clear();
            bestweight.clear();

            PartitionByTptree(indices, first, i - 1, leaves);
            PartitionByTptree(indices, i, last, leaves);
        }
    }

    void ComponentInitSPTAG_BKT::AddNeighbor(int idx, float dist, int origin) {
        int size = index->m_iNeighborhoodSize - 1;

        if (dist < index->getFinalGraph()[origin][size].distance ||
            (dist == index->getFinalGraph()[origin][size].distance && idx < index->getFinalGraph()[origin][size].id)) {
            int nb;

            for (nb = 0; nb <= size && index->getFinalGraph()[origin][nb].id != idx; nb++);

            if (nb > size) {
                nb = size;
                while (nb > 0 && (dist < index->getFinalGraph()[origin][nb - 1].distance ||
                                  (dist == index->getFinalGraph()[origin][nb - 1].distance &&
                                   idx < index->getFinalGraph()[origin][nb - 1].id))) {
                    index->getFinalGraph()[origin][nb] = index->getFinalGraph()[origin][nb - 1];
                    nb--;
                }
                index->getFinalGraph()[origin][nb].distance = dist;
                index->getFinalGraph()[origin][nb].id = idx;
            }
        }
    }

    void ComponentInitSPTAG_BKT::BuildGraph() {
        index->m_iNeighborhoodSize = index->m_iNeighborhoodSize * index->m_iNeighborhoodScale;

        index->getFinalGraph().resize(index->getBaseLen());

        float MaxDist = (std::numeric_limits<float>::max)();
        int MaxId = (std::numeric_limits<int>::max)();

        for (int i = 0; i < index->getBaseLen(); i++) {
            index->getFinalGraph()[i].resize(index->m_iNeighborhoodSize);
            for (int j = 0; j < index->m_iNeighborhoodSize; j++) {
                Index::SimpleNeighbor neighbor(MaxId, MaxDist);
                index->getFinalGraph()[i][j] = neighbor;
            }
        }

        std::vector<std::vector<int>> TptreeDataIndices(index->m_iTPTNumber, std::vector<int>(index->getBaseLen()));
        std::vector<std::vector<std::pair<int, int>>> TptreeLeafNodes(index->m_iTPTNumber,
                                                                      std::vector<std::pair<int, int>>());

        // auto t1 = std::chrono::high_resolution_clock::now();
        // std::cout << "Parallel TpTree Partition begin\n";
#pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < index->m_iTPTNumber; i++) {
            // Sleep(i * 100);
            std::srand(clock());
            for (int j = 0; j < index->getBaseLen(); j++) TptreeDataIndices[i][j] = j;
            std::random_shuffle(TptreeDataIndices[i].begin(), TptreeDataIndices[i].end());
            PartitionByTptree(TptreeDataIndices[i], 0, index->getBaseLen() - 1, TptreeLeafNodes[i]);
            // std::cout << "Finish Getting Leaves for Tree " << i << std::endl;
        }
        // std::cout << "Parallel TpTree Partition done\n";
        // auto t2 = std::chrono::high_resolution_clock::now();
        // std::cout << "Build TPTree time (s): " << std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count()
                //   << std::endl;

        for (int i = 0; i < index->m_iTPTNumber; i++) {
#pragma omp parallel for schedule(dynamic)
            for (int j = 0; j < (int) TptreeLeafNodes[i].size(); j++) {
                int start_index = TptreeLeafNodes[i][j].first;
                int end_index = TptreeLeafNodes[i][j].second;
                if ((j * 5) % TptreeLeafNodes[i].size() == 0)
                    // std::cout << "Processing Tree " << i << " "
                            //   << static_cast<int>(j * 1.0 / TptreeLeafNodes[i].size() * 100) << std::endl;
                for (int x = start_index; x < end_index; x++) {
                    for (int y = x + 1; y <= end_index; y++) {
                        int p1 = TptreeDataIndices[i][x];
                        int p2 = TptreeDataIndices[i][y];
                        float dist = index->getDist()->compare(index->getBaseData() + index->getBaseDim() * p1,
                                                               index->getBaseData() + index->getBaseDim() * p2,
                                                               index->getBaseDim());

                        AddNeighbor(p2, dist, p1);
                        AddNeighbor(p1, dist, p2);
                    }
                }
            }
            TptreeDataIndices[i].clear();
            TptreeLeafNodes[i].clear();
        }
        TptreeDataIndices.clear();
        TptreeLeafNodes.clear();

        // auto t3 = std::chrono::high_resolution_clock::now();
        // std::cout << "Process TPTree time (s): " << std::chrono::duration_cast<std::chrono::seconds>(t3 - t2).count()
                //   << std::endl;
    }


    // HCNNG
    void ComponentInitHCNNG::InitInner() {

        // -- Hierarchical clustering --

        SetConfigs();

        int max_mst_degree = 3;

        std::vector<std::vector<Index::Edge> > G(index->getBaseLen());
        std::vector<omp_lock_t> locks(index->getBaseLen());

        for (int i = 0; i < index->getBaseLen(); i++) {
            omp_init_lock(&locks[i]);
            G[i].reserve(max_mst_degree * index->num_cl);
        }

        // printf("creating clusters...\n");
#pragma omp parallel for
        for (int i = 0; i < index->num_cl; i++) {
            int *idx_points = new int[index->getBaseLen()];
            for (int j = 0; j < index->getBaseLen(); j++)
                idx_points[j] = j;
            create_clusters(idx_points, 0, index->getBaseLen() - 1, G, index->minsize_cl, locks, max_mst_degree);
            // printf("end cluster %d\n", i);
            delete[] idx_points;
        }

        // printf("sorting...\n");
        sort_edges(G);
        // print_stats_graph(G);

        // G - > final_graph
        index->getFinalGraph().resize(index->getBaseLen());

        for (int i = 0; i < index->getBaseLen(); i++) {
            std::vector<Index::SimpleNeighbor> tmp;

            int degree = G[i].size();

            for (int j = 0; j < degree; j++) {
                tmp.emplace_back(G[i][j].v2, G[i][j].weight);
            }
            index->getFinalGraph()[i] = tmp;
        }

        // -- KD-tree --
        unsigned seed = 1998;

        const auto TreeNum = index->getParam().get<unsigned>("nTrees");
        const auto TreeNumBuild = index->getParam().get<unsigned>("nTrees");
        const auto K = index->getParam().get<unsigned>("K");

        std::vector<int> indices(index->getBaseLen());
        index->LeafLists.resize(TreeNum);
        std::vector<Index::EFANNA::Node *> ActiveSet;
        std::vector<Index::EFANNA::Node *> NewSet;
        for (unsigned i = 0; i < (unsigned) TreeNum; i++) {
            auto *node = new Index::EFANNA::Node;
            node->DivDim = -1;
            node->Lchild = nullptr;
            node->Rchild = nullptr;
            node->StartIdx = 0;
            node->EndIdx = index->getBaseLen();
            node->treeid = i;
            index->tree_roots_.push_back(node);
            ActiveSet.push_back(node);
        }

#pragma omp parallel for
        for (unsigned i = 0; i < index->getBaseLen(); i++)indices[i] = i;
#pragma omp parallel for
        for (unsigned i = 0; i < (unsigned) TreeNum; i++) {
            std::vector<unsigned> &myids = index->LeafLists[i];
            myids.resize(index->getBaseLen());
            std::copy(indices.begin(), indices.end(), myids.begin());
            std::random_shuffle(myids.begin(), myids.end());
        }
        omp_init_lock(&index->rootlock);
        while (!ActiveSet.empty() && ActiveSet.size() < 1100) {
#pragma omp parallel for
            for (unsigned i = 0; i < ActiveSet.size(); i++) {
                Index::EFANNA::Node *node = ActiveSet[i];
                unsigned mid;
                unsigned cutdim;
                float cutval;
                std::mt19937 rng(seed ^ omp_get_thread_num());
                std::vector<unsigned> &myids = index->LeafLists[node->treeid];

                meanSplit(rng, &myids[0] + node->StartIdx, node->EndIdx - node->StartIdx, mid, cutdim, cutval);

                node->DivDim = cutdim;
                node->DivVal = cutval;
                //node->StartIdx = offset;
                //node->EndIdx = offset + count;
                auto *nodeL = new Index::EFANNA::Node();
                auto *nodeR = new Index::EFANNA::Node();
                nodeR->treeid = nodeL->treeid = node->treeid;
                nodeL->StartIdx = node->StartIdx;
                nodeL->EndIdx = node->StartIdx + mid;
                nodeR->StartIdx = nodeL->EndIdx;
                nodeR->EndIdx = node->EndIdx;
                node->Lchild = nodeL;
                node->Rchild = nodeR;
                omp_set_lock(&index->rootlock);
                if (mid > K)NewSet.push_back(nodeL);
                if (nodeR->EndIdx - nodeR->StartIdx > K)NewSet.push_back(nodeR);
                omp_unset_lock(&index->rootlock);
            }
            ActiveSet.resize(NewSet.size());
            std::copy(NewSet.begin(), NewSet.end(), ActiveSet.begin());
            NewSet.clear();
        }

#pragma omp parallel for
        for (unsigned i = 0; i < ActiveSet.size(); i++) {
            Index::EFANNA::Node *node = ActiveSet[i];
            std::mt19937 rng(seed ^ omp_get_thread_num());
            std::vector<unsigned> &myids = index->LeafLists[node->treeid];
            DFSbuild(node, rng, &myids[0] + node->StartIdx, node->EndIdx - node->StartIdx, node->StartIdx);
        }
        //DFStest(0,0,tree_roots_[0]);
        std::cout << "build tree completed" << std::endl;

        for (size_t i = 0; i < (unsigned) TreeNumBuild; i++) {
            getMergeLevelNodeList(index->tree_roots_[i], i, 0);
        }

        std::cout << "merge node list size: " << index->mlNodeList.size() << std::endl;
        if (index->error_flag) {
            std::cout << "merge level deeper than tree, max merge deepth is " << index->max_deepth - 1 << std::endl;
        }

        std::cout << "merge tree completed" << std::endl;

        //  -- space partition tree -> guided search --
        build_tree();

//        for(int i = 0; i < 10; i ++) {
//            for(int j = 0; j < index->getFinalGraph()[i].size(); j ++) {
//                std::cout << index->getFinalGraph()[i][j].id << "|" << index->getFinalGraph()[i][j].distance << " ";
//            }
//            std::cout << std::endl;
//        }
    }

    void ComponentInitHCNNG::SetConfigs() {
        index->minsize_cl = (unsigned)std::sqrt(index->getBaseLen());
        index->num_cl = index->getParam().get<unsigned>("num_cl");

        // kd-tree
        index->mLevel = index->getParam().get<unsigned>("mLevel");
        index->nTrees = index->getParam().get<unsigned>("nTrees");
    }

    void ComponentInitHCNNG::build_tree() {
        index->Tn.resize(index->getBaseLen());

        for (size_t i = 0; i < index->getBaseLen(); i++) {
            auto size = index->getFinalGraph()[i].size();
            long long min_diff = 1e6, min_diff_dim = -1;
            for (size_t j = 0; j < index->getBaseDim(); j++) {
                int lnum = 0, rnum = 0;
                for (size_t k = 0; k < size; k++) {
                    if ((index->getBaseData() + index->getFinalGraph()[i][k].id * index->getBaseDim())[j] <
                        (index->getBaseData() + i * index->getBaseDim())[j]) {
                        lnum++;
                    } else {
                        rnum++;
                    }
                }
                long long diff = lnum - rnum;
                if (diff < 0) diff = -diff;
                if (diff < min_diff) {
                    min_diff = diff;
                    min_diff_dim = j;
                }
            }
            index->Tn[i].div_dim = min_diff_dim;
            for (size_t k = 0; k < size; k++) {
                if ((index->getBaseData() + index->getFinalGraph()[i][k].id * index->getBaseDim())[min_diff_dim] <
                    (index->getBaseData() + i * index->getBaseDim())[min_diff_dim]) {
                    index->Tn[i].left.push_back(index->getFinalGraph()[i][k].id);
                } else {
                    index->Tn[i].right.push_back(index->getFinalGraph()[i][k].id);
                }
            }
        }
    }

    int ComponentInitHCNNG::rand_int(const int &min, const int &max) {
        static thread_local std::mt19937 *generator = nullptr;
        if (!generator)
            generator = new std::mt19937(clock() + std::hash<std::thread::id>()(std::this_thread::get_id()));
        std::uniform_int_distribution<int> distribution(min, max);
        return distribution(*generator);
    }

    std::tuple<std::vector<std::vector<Index::Edge> >, float>
    kruskal(std::vector<Index::Edge> &edges, int N, int max_mst_degree) {
        sort(edges.begin(), edges.end());
        std::vector<std::vector<Index::Edge >> MST(N);
        auto *disjset = new Index::DisjointSet(N);
        float cost = 0;
        for (Index::Edge &e : edges) {
            if (disjset->find(e.v1) != disjset->find(e.v2) && MST[e.v1].size() < max_mst_degree &&
                MST[e.v2].size() < max_mst_degree) {
                MST[e.v1].push_back(e);
                MST[e.v2].push_back(Index::Edge(e.v2, e.v1, e.weight));
                disjset->_union(e.v1, e.v2);
                cost += e.weight;

            }
        }
        delete disjset;
        return make_tuple(MST, cost);
    }

    std::vector<std::vector<Index::Edge> >
    ComponentInitHCNNG::create_exact_mst(int *idx_points, int left, int right, int max_mst_degree) {
        int N = right - left + 1;
        if (N == 1) {
            index->xxx++;
            printf("%d\n", index->xxx);
        }
        float cost;
        std::vector<Index::Edge> full;
        std::vector<std::vector<Index::Edge> > mst;
        full.reserve(N * (N - 1));
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++)
                if (i != j) {
                    float dist = index->getDist()->compare(
                            index->getBaseData() + idx_points[left + i] * index->getBaseDim(),
                            index->getBaseData() + idx_points[left + j] * index->getBaseDim(),
                            (unsigned) index->getBaseDim());
                    full.emplace_back(i, j, dist);
                }

        }
        tie(mst, cost) = kruskal(full, N, max_mst_degree);
        return mst;
    }

    bool check_in_neighbors(int u, std::vector<Index::Edge> &neigh) {
        for (int i = 0; i < neigh.size(); i++) {
            if (neigh[i].v2 == u)
                return true;
        }
        return false;
    }

    void
    ComponentInitHCNNG::create_clusters(int *idx_points, int left, int right,
                                        std::vector<std::vector<Index::Edge> > &graph,
                                        int minsize_cl, std::vector<omp_lock_t> &locks, int max_mst_degree) {
        int num_points = right - left + 1;

        if (num_points < minsize_cl) {
            std::vector<std::vector<Index::Edge> > mst = create_exact_mst(idx_points, left, right, max_mst_degree);
            for (int i = 0; i < num_points; i++) {
                for (int j = 0; j < mst[i].size(); j++) {
                    omp_set_lock(&locks[idx_points[left + i]]);
                    if (!check_in_neighbors(idx_points[left + mst[i][j].v2], graph[idx_points[left + i]])) {
                        graph[idx_points[left + i]].push_back(
                                Index::Edge(idx_points[left + i], idx_points[left + mst[i][j].v2], mst[i][j].weight));
                    }

                    omp_unset_lock(&locks[idx_points[left + i]]);
                }
            }
        } else {
            int x = rand_int(left, right);
            int y = rand_int(left, right);
            while (y == x) y = rand_int(left, right);

            std::vector<std::pair<float, int> > dx(num_points);
            std::vector<std::pair<float, int> > dy(num_points);
            std::unordered_set<int> taken;
            for (int i = 0; i < num_points; i++) {
                dx[i] = std::make_pair(
                        index->getDist()->compare(index->getBaseData() + index->getBaseDim() * idx_points[x],
                                                  index->getBaseData() +
                                                  index->getBaseDim() * idx_points[left + i],
                                                  index->getBaseDim()), idx_points[left + i]);
                dy[i] = std::make_pair(
                        index->getDist()->compare(index->getBaseData() + index->getBaseDim() * idx_points[y],
                                                  index->getBaseData() +
                                                  index->getBaseDim() * idx_points[left + i],
                                                  index->getBaseDim()), idx_points[left + i]);
            }
            sort(dx.begin(), dx.end());
            sort(dy.begin(), dy.end());
            int i = 0, j = 0, turn = rand_int(0, 1), p = left, q = right;
            while (i < num_points || j < num_points) {
                if (turn == 0) {
                    if (i < num_points) {
                        if (not_in_set(dx[i].second, taken)) {
                            idx_points[p] = dx[i].second;
                            taken.insert(dx[i].second);
                            p++;
                            turn = (turn + 1) % 2;
                        }
                        i++;
                    } else {
                        turn = (turn + 1) % 2;
                    }
                } else {
                    if (j < num_points) {
                        if (not_in_set(dy[j].second, taken)) {
                            idx_points[q] = dy[j].second;
                            taken.insert(dy[j].second);
                            q--;
                            turn = (turn + 1) % 2;
                        }
                        j++;
                    } else {
                        turn = (turn + 1) % 2;
                    }
                }
            }

            dx.clear();
            dy.clear();
            taken.clear();
            std::vector<std::pair<float, int> >().swap(dx);
            std::vector<std::pair<float, int> >().swap(dy);

            create_clusters(idx_points, left, p - 1, graph, minsize_cl, locks, max_mst_degree);
            create_clusters(idx_points, p, right, graph, minsize_cl, locks, max_mst_degree);
        }
    }

    void ComponentInitHCNNG::sort_edges(std::vector<std::vector<Index::Edge> > &G) {
        int N = G.size();
#pragma omp parallel for
        for (int i = 0; i < N; i++)
            sort(G[i].begin(), G[i].end());
    }

    std::vector<int> get_sizeadj(std::vector<std::vector<Index::Edge> > &G) {
        std::vector<int> NE(G.size());
        for (int i = 0; i < G.size(); i++)
            NE[i] = G[i].size();
        return NE;
    }

    template<typename SomeType>
    float mean_v(std::vector<SomeType> a) {
        float s = 0;
        for (float x: a) s += x;
        return s / a.size();
    }

    template<typename SomeType>
    float sum_v(std::vector<SomeType> a) {
        float s = 0;
        for (float x: a) s += x;
        return s;
    }

    template<typename SomeType>
    float max_v(std::vector<SomeType> a) {
        float mx = a[0];
        for (float x: a) mx = std::max(mx, x);
        return mx;
    }

    template<typename SomeType>
    float min_v(std::vector<SomeType> a) {
        float mn = a[0];
        for (float x: a) mn = std::min(mn, x);
        return mn;
    }

    template<typename SomeType>
    float std_v(std::vector<SomeType> a) {
        float m = mean_v(a), s = 0, n = a.size();
        for (float x: a) s += (x - m) * (x - m);
        return sqrt(s / (n - 1));
    }

    void ComponentInitHCNNG::print_stats_graph(std::vector<std::vector<Index::Edge> > &G) {
        std::vector<int> sizeadj;
        sizeadj = get_sizeadj(G);
        printf("num edges:\t%.0lf\n", sum_v(sizeadj) / 2);
        printf("max degree:\t%.0lf\n", max_v(sizeadj));
        printf("min degree:\t%.0lf\n", min_v(sizeadj));
        printf("avg degree:\t%.2lf\n", mean_v(sizeadj));
        printf("std degree:\t%.2lf\n\n", std_v(sizeadj));
    }

    // kdt
    void ComponentInitHCNNG::meanSplit(std::mt19937 &rng, unsigned *indices, unsigned count, unsigned &index1,
                                       unsigned &cutdim, float &cutval) {
        float *mean_ = new float[index->getBaseDim()];
        float *var_ = new float[index->getBaseDim()];
        memset(mean_, 0, index->getBaseDim() * sizeof(float));
        memset(var_, 0, index->getBaseDim() * sizeof(float));

        /* Compute mean values.  Only the first SAMPLE_NUM values need to be
          sampled to get a good estimate.
         */
        unsigned cnt = std::min((unsigned) index->SAMPLE_NUM + 1, count);
        for (unsigned j = 0; j < cnt; ++j) {
            const float *v = index->getBaseData() + indices[j] * index->getBaseDim();
            for (size_t k = 0; k < index->getBaseDim(); ++k) {
                mean_[k] += v[k];
            }
        }
        float div_factor = float(1) / cnt;
        for (size_t k = 0; k < index->getBaseDim(); ++k) {
            mean_[k] *= div_factor;
        }

        /* Compute variances (no need to divide by count). */

        for (unsigned j = 0; j < cnt; ++j) {
            const float *v = index->getBaseData() + indices[j] * index->getBaseDim();
            for (size_t k = 0; k < index->getBaseDim(); ++k) {
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
        if (lim1 > count / 2) index1 = lim1;
        else if (lim2 < count / 2) index1 = lim2;
        else index1 = count / 2;

        /* If either list is empty, it means that all remaining features
         * are identical. Split in the middle to maintain a balanced tree.
         */
        if ((lim1 == count) || (lim2 == 0)) index1 = count / 2;
        delete[] mean_;
        delete[] var_;
    }

    void
    ComponentInitHCNNG::planeSplit(unsigned *indices, unsigned count, unsigned cutdim, float cutval, unsigned &lim1,
                                   unsigned &lim2) {
        /* Move vector indices for left subtree to front of list. */
        int left = 0;
        int right = count - 1;
        for (;;) {
            const float *vl = index->getBaseData() + indices[left] * index->getBaseDim();
            const float *vr = index->getBaseData() + indices[right] * index->getBaseDim();
            while (left <= right && vl[cutdim] < cutval) {
                ++left;
                vl = index->getBaseData() + indices[left] * index->getBaseDim();
            }
            while (left <= right && vr[cutdim] >= cutval) {
                --right;
                vr = index->getBaseData() + indices[right] * index->getBaseDim();
            }
            if (left > right) break;
            std::swap(indices[left], indices[right]);
            ++left;
            --right;
        }
        lim1 = left;//lim1 is the id of the leftmost point <= cutval
        right = count - 1;
        for (;;) {
            const float *vl = index->getBaseData() + indices[left] * index->getBaseDim();
            const float *vr = index->getBaseData() + indices[right] * index->getBaseDim();
            while (left <= right && vl[cutdim] <= cutval) {
                ++left;
                vl = index->getBaseData() + indices[left] * index->getBaseDim();
            }
            while (left <= right && vr[cutdim] > cutval) {
                --right;
                vr = index->getBaseData() + indices[right] * index->getBaseDim();
            }
            if (left > right) break;
            std::swap(indices[left], indices[right]);
            ++left;
            --right;
        }
        lim2 = left;//lim2 is the id of the leftmost point >cutval
    }

    int ComponentInitHCNNG::selectDivision(std::mt19937 &rng, float *v) {
        int num = 0;
        size_t topind[index->RAND_DIM];

        //Create a list of the indices of the top index->RAND_DIM values.
        for (size_t i = 0; i < index->getBaseDim(); ++i) {
            if ((num < index->RAND_DIM) || (v[i] > v[topind[num - 1]])) {
                // Put this element at end of topind.
                if (num < index->RAND_DIM) {
                    topind[num++] = i;            // Add to list.
                } else {
                    topind[num - 1] = i;         // Replace last element.
                }
                // Bubble end value down to right location by repeated swapping. sort the varience in decrease order
                int j = num - 1;
                while (j > 0 && v[topind[j]] > v[topind[j - 1]]) {
                    std::swap(topind[j], topind[j - 1]);
                    --j;
                }
            }
        }
        // Select a random integer in range [0,num-1], and return that index.
        int rnd = rng() % num;
        return (int) topind[rnd];
    }

    void ComponentInitHCNNG::DFSbuild(Index::EFANNA::Node *node, std::mt19937 &rng, unsigned *indices, unsigned count,
                                      unsigned offset) {
        //omp_set_lock(&rootlock);
        //std::cout<<node->treeid<<":"<<offset<<":"<<count<<std::endl;
        //omp_unset_lock(&rootlock);

        if (count <= index->TNS) {
            node->DivDim = -1;
            node->Lchild = nullptr;
            node->Rchild = nullptr;
            node->StartIdx = offset;
            node->EndIdx = offset + count;
            //add points

        } else {
            unsigned idx;
            unsigned cutdim;
            float cutval;
            meanSplit(rng, indices, count, idx, cutdim, cutval);
            node->DivDim = cutdim;
            node->DivVal = cutval;
            node->StartIdx = offset;
            node->EndIdx = offset + count;
            auto *nodeL = new Index::EFANNA::Node();
            auto *nodeR = new Index::EFANNA::Node();
            node->Lchild = nodeL;
            nodeL->treeid = node->treeid;
            DFSbuild(nodeL, rng, indices, idx, offset);
            node->Rchild = nodeR;
            nodeR->treeid = node->treeid;
            DFSbuild(nodeR, rng, indices + idx, count - idx, offset + idx);
        }
    }

    void ComponentInitHCNNG::getMergeLevelNodeList(Index::EFANNA::Node *node, size_t treeid, unsigned deepth) {
        auto ml = index->getParam().get<unsigned>("mLevel");
        if (node->Lchild != nullptr && node->Rchild != nullptr && deepth < ml) {
            deepth++;
            getMergeLevelNodeList(node->Lchild, treeid, deepth);
            getMergeLevelNodeList(node->Rchild, treeid, deepth);
        } else if (deepth == ml) {
            index->mlNodeList.emplace_back(node, treeid);
        } else {
            index->error_flag = true;
            if (deepth < index->max_deepth)index->max_deepth = deepth;
        }
    }

}

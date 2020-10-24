//
// Created by MurphySL on 2020/10/23.
//

#include "weavess/component.h"

namespace weavess {

    // NN-Descent
    void ComponentInitNNDescent::InitInner() {

        // L ITER S R
        SetConfigs();

        // 添加随机点作为近邻
        init();

        NNDescent();

        // graph_ -> final_graph
        index->getFinalGraph().resize(index->getBaseLen());
        for (unsigned i = 0; i < index->getBaseLen(); i++) {
            std::vector<Index::SimpleNeighbor> tmp;

            std::sort(index->graph_[i].pool.begin(), index->graph_[i].pool.end());

            for (auto & j : index->graph_[i].pool)
                tmp.push_back(Index::SimpleNeighbor(j.id, j.distance));

            index->getFinalGraph()[i] = tmp;

            // 内存释放
            std::vector<Index::Neighbor>().swap(index->graph_[i].pool);
            std::vector<unsigned>().swap(index->graph_[i].nn_new);
            std::vector<unsigned>().swap(index->graph_[i].nn_old);
            std::vector<unsigned>().swap(index->graph_[i].rnn_new);
            std::vector<unsigned>().swap(index->graph_[i].rnn_new);
        }

        // 内存释放
        std::vector<Index::nhood>().swap(index->graph_);
    }

    void ComponentInitNNDescent::SetConfigs() {
        index->L = index->getParam().get<unsigned>("L");
        index->S = index->getParam().get<unsigned>("S");
        index->R = index->getParam().get<unsigned>("R");
        index->ITER = index->getParam().get<unsigned>("ITER");
    }

    void ComponentInitNNDescent::init() {
        index->graph_.reserve(index->getBaseLen());
        std::mt19937 rng(rand());
        for (unsigned i = 0; i < index->getBaseLen(); i++) {
            index->graph_.emplace_back(Index::nhood(index->L, index->S, rng, (unsigned) index->getBaseLen()));
        }
#ifdef PARALLEL
#pragma omp parallel for
#endif
        for (unsigned i = 0; i < index->getBaseLen(); i++) {
            std::vector<unsigned> tmp(index->S + 1);

            weavess::GenRandom(rng, tmp.data(), index->S + 1, index->getBaseLen());

            for (unsigned j = 0; j < index->S; j++) {
                unsigned id = tmp[j];

                if (id == i)continue;
                float dist = index->getDist()->compare(index->getBaseData() + i * index->getBaseDim(),
                                                       index->getBaseData() + id * index->getBaseDim(),
                                                       (unsigned) index->getBaseDim());

                index->graph_[i].pool.emplace_back(Index::Neighbor(id, dist, true));
            }
            std::make_heap(index->graph_[i].pool.begin(), index->graph_[i].pool.end());
            index->graph_[i].pool.reserve(index->L);
        }
    }

    void ComponentInitNNDescent::NNDescent() {
        for (unsigned it = 0; it < index->ITER; it++) {
            std::cout << "NN-Descent iter: " << it << std::endl;

            join();

            update();
        }
    }

    void ComponentInitNNDescent::join() {
#ifdef PARALLEL
#pragma omp parallel for default(shared) schedule(dynamic, 100)
#endif
        for (unsigned n = 0; n < index->getBaseLen(); n++) {
            index->graph_[n].join([&](unsigned i, unsigned j) {
                if (i != j) {
                    float dist = index->getDist()->compare(index->getBaseData() + i * index->getBaseDim(),
                                                           index->getBaseData() + j * index->getBaseDim(),
                                                           index->getBaseDim());

                    index->graph_[i].insert(j, dist);
                    index->graph_[j].insert(i, dist);
                }
            });
        }
    }

    void ComponentInitNNDescent::update() {
#ifdef PARALLEL
#pragma omp parallel for
#endif
        for (unsigned i = 0; i < index->getBaseLen(); i++) {
            std::vector<unsigned>().swap(index->graph_[i].nn_new);
            std::vector<unsigned>().swap(index->graph_[i].nn_old);
            //std::vector<unsigned>().swap(graph_[i].rnn_new);
            //std::vector<unsigned>().swap(graph_[i].rnn_old);
            //graph_[i].nn_new.clear();
            //graph_[i].nn_old.clear();
            //graph_[i].rnn_new.clear();
            //graph_[i].rnn_old.clear();
        }
#ifdef PARALLEL
#pragma omp parallel for
#endif
        for (unsigned n = 0; n < index->getBaseLen(); ++n) {
            auto &nn = index->graph_[n];
            std::sort(nn.pool.begin(), nn.pool.end());
            if(nn.pool.size()>index->L)nn.pool.resize(index->L);
            nn.pool.reserve(index->L);
            unsigned maxl = std::min(nn.M + index->S, (unsigned) nn.pool.size());
            unsigned c = 0;
            unsigned l = 0;
            //std::sort(nn.pool.begin(), nn.pool.end());
            //if(n==0)std::cout << nn.pool[0].distance<<","<< nn.pool[1].distance<<","<< nn.pool[2].distance<< std::endl;
            while ((l < maxl) && (c < index->S)) {
                if (nn.pool[l].flag) ++c;
                ++l;
            }
            nn.M = l;
        }
#ifdef PARALLEL
#pragma omp parallel for
#endif
        for (unsigned n = 0; n < index->getBaseLen(); ++n) {
            auto &nnhd = index->graph_[n];
            auto &nn_new = nnhd.nn_new;
            auto &nn_old = nnhd.nn_old;
            for (unsigned l = 0; l < nnhd.M; ++l) {
                auto &nn = nnhd.pool[l];
                auto &nhood_o = index->graph_[nn.id];  // nn on the other side of the edge

                if (nn.flag) {
                    nn_new.push_back(nn.id);
                    if (nn.distance > nhood_o.pool.back().distance) {
                        Index::LockGuard guard(nhood_o.lock);
                        if(nhood_o.rnn_new.size() < index->R)nhood_o.rnn_new.push_back(n);
                        else{
                            unsigned int pos = rand() % index->R;
                            nhood_o.rnn_new[pos] = n;
                        }
                    }
                    nn.flag = false;
                } else {
                    nn_old.push_back(nn.id);
                    if (nn.distance > nhood_o.pool.back().distance) {
                        Index::LockGuard guard(nhood_o.lock);
                        if(nhood_o.rnn_old.size() < index->R)nhood_o.rnn_old.push_back(n);
                        else{
                            unsigned int pos = rand() % index->R;
                            nhood_o.rnn_old[pos] = n;
                        }
                    }
                }
            }
            std::make_heap(nnhd.pool.begin(), nnhd.pool.end());
        }
#ifdef PARALLEL
#pragma omp parallel for
#endif
        for (unsigned i = 0; i < index->getBaseLen(); ++i) {
            auto &nn_new = index->graph_[i].nn_new;
            auto &nn_old = index->graph_[i].nn_old;
            auto &rnn_new = index->graph_[i].rnn_new;
            auto &rnn_old = index->graph_[i].rnn_old;
            if (index->R && rnn_new.size() > index->R) {
                std::random_shuffle(rnn_new.begin(), rnn_new.end());
                rnn_new.resize(index->R);
            }
            nn_new.insert(nn_new.end(), rnn_new.begin(), rnn_new.end());
            if (index->R && rnn_old.size() > index->R) {
                std::random_shuffle(rnn_old.begin(), rnn_old.end());
                rnn_old.resize(index->R);
            }
            nn_old.insert(nn_old.end(), rnn_old.begin(), rnn_old.end());
            if(nn_old.size() > index->R * 2){nn_old.resize(index->R * 2);nn_old.reserve(index->R*2);}
            std::vector<unsigned>().swap(index->graph_[i].rnn_new);
            std::vector<unsigned>().swap(index->graph_[i].rnn_old);
        }
    }


    // RAND
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

            for (auto & j : index->graph_[i].pool){
                tmp.push_back(Index::SimpleNeighbor(j.id, j.distance));
            }

            index->getFinalGraph()[i] = tmp;

            // 内存释放
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


    // KDT
    void ComponentInitKDT::InitInner() {
        SetConfigs();

        unsigned seed = 1998;

        index->graph_.resize(index->getBaseLen());
        index->knn_graph.resize(index->getBaseLen());

        const auto TreeNum = index->getParam().get<unsigned>("nTrees");
        const auto TreeNumBuild = index->getParam().get<unsigned>("nTrees");
        const auto K = index->getParam().get<unsigned>("K");

        // 选择树根
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
        // 构建随机截断树
        while (!ActiveSet.empty() && ActiveSet.size() < 1100) {
#pragma omp parallel for
            for (unsigned i = 0; i < ActiveSet.size(); i++) {
                Index::EFANNA::Node *node = ActiveSet[i];
                unsigned mid;
                unsigned cutdim;
                float cutval;
                std::mt19937 rng(seed ^ omp_get_thread_num());
                std::vector<unsigned> &myids = index->LeafLists[node->treeid];

                // 根据特征值进行划分
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
            // 查找树根对应节点
            std::vector<unsigned> &myids = index->LeafLists[node->treeid];
            // 添加规定深度下的所有子节点
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


    // IEH
    void ComponentInitIEH::InitInner() {
        std::string func_argv = index->getParam().get<std::string>("func");
        std::string basecode_argv = index->getParam().get<std::string>("basecode");
        std::string knntable_argv = index->getParam().get<std::string>("knntable");

        Index::Matrix func;
        Index::Codes basecode;
        Index::Codes querycode;
        Index::Matrix train;
        Index::Matrix test;

        LoadHashFunc(&func_argv[0], func);
        LoadBaseCode(&basecode_argv[0], basecode);

        int UpperBits = 8;
        int LowerBits = 8; //change with code length:code length = up + low;
        Index::HashTable tb;
        BuildHashTable(UpperBits,LowerBits,basecode,tb);
        std::cout<<"build hash table complete"<<std::endl;

        QueryToCode(test, func, querycode);
        std::cout<<"convert query code complete"<<std::endl;
        std::vector<std::vector<int> > hashcands;
        HashTest(UpperBits, LowerBits, querycode, tb, hashcands);
        std::cout<<"hash candidates ready"<<std::endl;

        std::cout<<"initial finish : "<<std::endl;

        std::vector<Index::CandidateHeap2 > knntable;
        LoadKnnTable(&knntable_argv[0],knntable);
        std::cout<<"load knn graph complete"<<std::endl;

        //GNNS
        std::vector<Index::CandidateHeap2 > res;
        for(size_t i = 0; i < hashcands.size(); i++){
            Index::CandidateHeap2 cands;
            for(size_t j = 0; j < hashcands[i].size(); j++){
                int neighbor = hashcands[i][j];
                Index::Candidate2<float> c(neighbor, index->getDist()->compare(&test[i][0], &train[neighbor][0], test[i].size()));
                cands.insert(c);
                if(cands.size() > POOL_SIZE)cands.erase(cands.begin());
            }
            res.push_back(cands);
        }

        //iteration
        auto expand = index->getParam().get<unsigned>("expand");
        auto iterlimit = index->getParam().get<unsigned>("iterlimit");
        for(size_t i = 0; i < res.size(); i++){
            int niter = 0;
            while(niter++ < iterlimit){
                Index::CandidateHeap2::reverse_iterator it = res[i].rbegin();
                std::vector<int> ids;
                for(int j = 0; it != res[i].rend() && j < expand; it++, j++){
                    int neighbor = it->row_id;
                    Index::CandidateHeap2::reverse_iterator nnit = knntable[neighbor].rbegin();
                    for(int k = 0; nnit != knntable[neighbor].rend() && k < expand; nnit++, k++){
                        int nn = nnit->row_id;
                        ids.push_back(nn);
                    }
                }
                for(size_t j = 0; j < ids.size(); j++){
                    Index::Candidate2<float> c(ids[j], index->getDist()->compare(&test[i][0], &train[ids[j]][0], test[i].size()));
                    res[i].insert(c);
                    if(res[i].size() > POOL_SIZE)res[i].erase(res[i].begin());
                }
            }//cout<<i<<endl;
        }
        std::cout<<"GNNS complete "<<std::endl;
    }

    void StringSplit(std::string src, std::vector<std::string>& des){
        int start = 0;
        int end = 0;
        for(size_t i = 0; i < src.length(); i++){
            if(src[i]==' '){
                end = i;
                //if(end>start)cout<<start<<" "<<end<<" "<<src.substr(start,end-start)<<endl;
                des.push_back(src.substr(start,end-start));
                start = i+1;
            }
        }
    }

    void ComponentInitIEH::LoadHashFunc(char* filename, Index::Matrix& func){
        std::ifstream in(filename);
        char buf[MAX_ROWSIZE];

        while(!in.eof()){
            in.getline(buf,MAX_ROWSIZE);
            std::string strtmp(buf);
            std::vector<std::string> strs;
            StringSplit(strtmp,strs);
            if(strs.size()<2)continue;
            std::vector<float> ftmp;
            for(size_t i = 0; i < strs.size(); i++){
                float f = atof(strs[i].c_str());
                ftmp.push_back(f);
                //cout<<f<<" ";
            }//cout<<endl;
            //cout<<strtmp<<endl;
            func.push_back(ftmp);
        }//cout<<func.size()<<endl;
        in.close();
    }

    void ComponentInitIEH::LoadBaseCode(char* filename, Index::Codes& base){
        std::ifstream in(filename);
        char buf[MAX_ROWSIZE];
        //int cnt = 0;
        while(!in.eof()){
            in.getline(buf,MAX_ROWSIZE);
            std::string strtmp(buf);
            std::vector< std::string> strs;
            StringSplit(strtmp,strs);
            if(strs.size()<2)continue;
            unsigned int codetmp = 0;
            for(size_t i = 0; i < strs.size(); i++){
                unsigned int c = atoi(strs[i].c_str());
                codetmp = codetmp << 1;
                codetmp += c;

            }//if(cnt++ > 999998){cout<<strs.size()<<" "<<buf<<" "<<codetmp<<endl;}
            base.push_back(codetmp);
        }//cout<<base.size()<<endl;
        in.close();
    }

    void ComponentInitIEH::BuildHashTable(int upbits, int lowbits, Index::Codes base ,Index::HashTable& tb){
        tb.clear();
        for(int i = 0; i < (1 << upbits); i++){
            Index::HashBucket emptyBucket;
            tb.push_back(emptyBucket);
        }
        for(size_t i = 0; i < base.size(); i ++){
            unsigned int idx1 = base[i] >> lowbits;
            unsigned int idx2 = base[i] - (idx1 << lowbits);
            if(tb[idx1].find(idx2) != tb[idx1].end()){
                tb[idx1][idx2].push_back(i);
            }else{
                std::vector<unsigned int> v;
                v.push_back(i);
                tb[idx1].insert(make_pair(idx2,v));
            }
        }
    }

    bool MatrixMultiply(Index::Matrix A, Index::Matrix B, Index::Matrix& C){
        if(A.size() == 0 || B.size() == 0){std::cout<<"matrix a or b size 0"<<std::endl;return false;}
        else if(A[0].size() != B.size()){
            std::cout<<"--error: matrix a, b dimension not agree"<<std::endl;
            std::cout<<"A"<<A.size()<<" * "<<A[0].size()<<std::endl;
            std::cout<<"B"<<B.size()<<" * "<<B[0].size()<<std::endl;
            return false;
        }
        for(size_t i = 0; i < A.size(); i++){
            std::vector<float> tmp;
            for(size_t j = 0; j < B[0].size(); j++){
                float fnum = 0;
                for(size_t k=0; k < B.size(); k++)fnum += A[i][k] * B[k][j];
                tmp.push_back(fnum);
            }
            C.push_back(tmp);
        }
        return true;
    }

    void ComponentInitIEH::QueryToCode(Index::Matrix query, Index::Matrix func, Index::Codes& querycode){
        Index::Matrix Z;
        if(!MatrixMultiply(query, func, Z)){return;}
        for(size_t i = 0; i < Z.size(); i++){
            unsigned int codetmp = 0;
            for(size_t j = 0; j < Z[0].size(); j++){
                if(Z[i][j]>0){codetmp = codetmp << 1;codetmp += 1;}
                else {codetmp = codetmp << 1;codetmp += 0;}
            }
            //if(i<3)cout<<codetmp<<endl;
            querycode.push_back(codetmp);
        }//cout<<querycode.size()<<endl;
    }

    void ComponentInitIEH::HashTest(int upbits,int lowbits, Index::Codes querycode, Index::HashTable tb, std::vector<std::vector<int> >& cands){
        for(size_t i = 0; i < querycode.size(); i++){

            unsigned int idx1 = querycode[i] >> lowbits;
            unsigned int idx2 = querycode[i] - (idx1 << lowbits);
            Index::HashBucket::iterator bucket= tb[idx1].find(idx2);
            std::vector<int> canstmp;
            if(bucket != tb[idx1].end()){
                std::vector<unsigned int> vp = bucket->second;
                //cout<<i<<":"<<vp.size()<<endl;
                for(size_t j = 0; j < vp.size() && canstmp.size() < INIT_NUM; j++){
                    canstmp.push_back(vp[j]);
                }
            }


            if(HASH_RADIUS == 0){
                cands.push_back(canstmp);
                continue;
            }
            for(size_t j = 0; j < DEPTH; j++){
                unsigned int searchcode = querycode[i] ^ (1 << j);
                unsigned int idx1 = searchcode >> lowbits;
                unsigned int idx2 = searchcode - (idx1 << lowbits);
                Index::HashBucket::iterator bucket= tb[idx1].find(idx2);
                if(bucket != tb[idx1].end()){
                    std::vector<unsigned int> vp = bucket->second;
                    for(size_t k = 0; k < vp.size() && canstmp.size() < INIT_NUM; k++){
                        canstmp.push_back(vp[k]);
                    }
                }
            }
            cands.push_back(canstmp);
        }
    }

    void ComponentInitIEH::LoadKnnTable(char* filename, std::vector<Index::CandidateHeap2 >& tb){
        std::ifstream in(filename,std::ios::binary);
        in.seekg(0,std::ios::end);
        std::ios::pos_type ss = in.tellg();
        size_t fsize = (size_t)ss;
        int dim;
        in.seekg(0,std::ios::beg);
        in.read((char*)&dim, sizeof(int));
        size_t num = fsize / (dim+1) / 4;
        std::cout<<"load graph "<<num<<" "<<dim<< std::endl;
        in.seekg(0,std::ios::beg);
        tb.clear();
        for(size_t i = 0; i < num; i++){
            Index::CandidateHeap2 heap;
            in.read((char*)&dim, sizeof(int));
            for(int j =0; j < dim; j++){
                int id;
                in.read((char*)&id, sizeof(int));
                Index::Candidate2<float> can(id, -1);
                heap.insert(can);
            }
            tb.push_back(heap);

        }
        in.close();
    }

    // NSW
//    void ComponentInitNSW::InitInner() {
//        SetConfigs();
//
//        Build(false);
//
//        for (size_t i = 0; i < index->nodes_.size(); ++i) {
//            delete index->nodes_[i];
//        }
//        index->nodes_.clear();
//    }
//
//    void ComponentInitNSW::SetConfigs() {
//        index->NN_ = index->getParam().get<unsigned>("NN");
//        index->ef_construction_ = index->getParam().get<unsigned>("ef_construction");
//        index->n_threads_ = index->getParam().get<unsigned>("n_threads_");
//    }
//
//    void ComponentInitNSW::Build(bool reverse) {
//        index->nodes_.resize(index->getBaseLen());
//        Index::HnswNode *first = new Index::HnswNode(0, 0, index->NN_, index->NN_);
//        index->nodes_[0] = first;
//        index->enterpoint_ = first;
//#pragma omp parallel num_threads(index->n_threads_)
//        {
//            auto *visited_list = new Index::VisitedList(index->getBaseLen());
//            if (reverse) {
//#pragma omp for schedule(dynamic, 128)
//                for (size_t i = index->getBaseLen() - 1; i >= 1; --i) {
//                    auto *qnode = new Index::HnswNode(i, 0, index->NN_, index->NN_);
//                    index->nodes_[i] = qnode;
//                    InsertNode(qnode, visited_list);
//                }
//            } else {
//#pragma omp for schedule(dynamic, 128)
//                for (size_t i = 1; i < index->getBaseLen(); ++i) {
//                    std::cout << i << std::endl;
//                    auto *qnode = new Index::HnswNode(i, 0, index->NN_, index->NN_);
//                    index->nodes_[i] = qnode;
//                    InsertNode(qnode, visited_list);
//                }
//            }
//            delete visited_list;
//        }
//    }
//
//    void ComponentInitNSW::InsertNode(Index::HnswNode *qnode, Index::VisitedList *visited_list) {
//        Index::HnswNode *enterpoint = index->enterpoint_;
//
//        std::priority_queue<Index::FurtherFirst> result;
//
//        // CANDIDATE
//        SearchAtLayer(qnode, enterpoint, 0, visited_list, result);
//
//        while (result.size() > 0) {
//            auto *top_node = result.top().GetNode();
//            result.pop();
//            Link(top_node, qnode, 0);
//        }
//    }
//
//    void ComponentInitNSW::SearchAtLayer(Index::HnswNode *qnode, Index::HnswNode *enterpoint, int level,
//                                           Index::VisitedList *visited_list,
//                                           std::priority_queue<Index::FurtherFirst> &result) {
//        // TODO: check Node 12bytes => 8bytes
//        std::priority_queue<Index::CloserFirst> candidates;
//        float d = index->getDist()->compare(index->getBaseData() + qnode->GetId() * index->getBaseDim(),
//                                            index->getBaseData() + enterpoint->GetId() * index->getBaseDim(),
//                                            index->getBaseDim());
//        result.emplace(enterpoint, d);
//        candidates.emplace(enterpoint, d);
//
//        visited_list->Reset();
//        visited_list->MarkAsVisited(enterpoint->GetId());
//
//        while (!candidates.empty()) {
//            const Index::CloserFirst &candidate = candidates.top();
//            float lower_bound = result.top().GetDistance();
//            if (candidate.GetDistance() > lower_bound)
//                break;
//
//            Index::HnswNode *candidate_node = candidate.GetNode();
//            std::unique_lock<std::mutex> lock(candidate_node->GetAccessGuard());
//            const std::vector<Index::HnswNode *> &neighbors = candidate_node->GetFriends(level);
//            candidates.pop();
//            for (const auto &neighbor : neighbors) {
//                int id = neighbor->GetId();
//                if (visited_list->NotVisited(id)) {
//                    visited_list->MarkAsVisited(id);
//                    d = index->getDist()->compare(index->getBaseData() + qnode->GetId() * index->getBaseDim(),
//                                                  index->getBaseData() + neighbor->GetId() * index->getBaseDim(),
//                                                  index->getBaseDim());
//                    if (result.size() < index->ef_construction_ || result.top().GetDistance() > d) {
//                        result.emplace(neighbor, d);
//                        candidates.emplace(neighbor, d);
//                        if (result.size() > index->ef_construction_)
//                            result.pop();
//                    }
//                }
//            }
//        }
//    }
//
//    void ComponentInitNSW::Link(Index::HnswNode *source, Index::HnswNode *target, int level) {
//        source->AddFriends(target, true);
//        target->AddFriends(source, true);
//    }
}

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

            for (auto &j : index->graph_[i].pool) {
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
            if (nn.pool.size() > index->L)nn.pool.resize(index->L);
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
                        if (nhood_o.rnn_new.size() < index->R)nhood_o.rnn_new.push_back(n);
                        else {
                            unsigned int pos = rand() % index->R;
                            nhood_o.rnn_new[pos] = n;
                        }
                    }
                    nn.flag = false;
                } else {
                    nn_old.push_back(nn.id);
                    if (nn.distance > nhood_o.pool.back().distance) {
                        Index::LockGuard guard(nhood_o.lock);
                        if (nhood_o.rnn_old.size() < index->R)nhood_o.rnn_old.push_back(n);
                        else {
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
            if (nn_old.size() > index->R * 2) {
                nn_old.resize(index->R * 2);
                nn_old.reserve(index->R * 2);
            }
            std::vector<unsigned>().swap(index->graph_[i].rnn_new);
            std::vector<unsigned>().swap(index->graph_[i].rnn_old);
        }
    }


    // FANNG
    void ComponentInitFANNG::InitInner() {
        SetConfigs();

        init();

        // graph_ -> final_graph
        index->getFinalGraph().resize(index->getBaseLen());
        for (unsigned i = 0; i < index->getBaseLen(); i++) {
            std::vector<Index::SimpleNeighbor> tmp;

            std::sort(index->graph_[i].pool.begin(), index->graph_[i].pool.end());

            for (int j = 0; j < index->L; j++) {
                tmp.push_back(Index::SimpleNeighbor(index->graph_[i].pool[j].id, index->graph_[i].pool[j].distance));
            }

//            for (auto &j : index->graph_[i].pool)
//                tmp.push_back(Index::SimpleNeighbor(j.id, j.distance));

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

    void ComponentInitFANNG::SetConfigs() {
        index->L = index->getParam().get<unsigned>("L");
    }

    void ComponentInitFANNG::init() {
        index->graph_.resize(index->getBaseLen());
#ifdef PARALLEL
#pragma omp parallel for
#endif
        for (unsigned i = 0; i < index->getBaseLen(); i++) {
            for (unsigned j = 0; j < index->getBaseLen(); j++) {
                if (i == j) continue;

                float dist = index->getDist()->compare(index->getBaseData() + index->getBaseDim() * i,
                                                       index->getBaseData() + index->getBaseDim() * j,
                                                       index->getBaseDim());
                index->graph_[i].pool.emplace_back(Index::Neighbor(j, dist, true));
            }

            std::make_heap(index->graph_[i].pool.begin(), index->graph_[i].pool.end());
            index->graph_[i].pool.reserve(index->getBaseLen() - 1);
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

            for (auto &j : index->graph_[i].pool) {
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
#pragma omp parallel num_threads(index->n_threads_)
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
        index->n_threads_ = index->getParam().get<unsigned>("n_threads_");
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
        // 随机决定当前结点层数
        int level = GetRandomNodeLevel();
        // 必须提前插入结点
        auto *first = new Index::HnswNode(0, level, index->max_m_, index->max_m0_);
        index->nodes_[0] = first;
        index->max_level_ = level;
        index->enterpoint_ = first;
#pragma omp parallel num_threads(index->n_threads_)
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

        // 当前结点所达层数小于最大层数，逐步向下寻找
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
            // 贪婪算法在当前层获取近邻候选点
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

        Build();
    }

    void ComponentInitANNG::SetConfigs() {
        index->edgeSizeForCreation = index->getParam().get<unsigned>("edgeSizeForCreation");
        index->truncationThreshold = index->getParam().get<unsigned>("truncationThreshold");
        index->edgeSizeForSearch = index->getParam().get<unsigned>("edgeSizeForSearch");

        index->size = index->getParam().get<unsigned>("size");
    }

    void ComponentInitANNG::Build() {
        index->getFinalGraph().resize(index->getBaseLen());

        // 为插入操作提前计算距离
        for (unsigned idxi = 0; idxi < index->getBaseLen(); idxi++) {
            std::vector<Index::SimpleNeighbor> tmp;
            for (unsigned idxj = 0; idxj < idxi; idxj++) {
                float d = index->getDist()->compare(index->getBaseData() + idxi * index->getBaseDim(),
                                                    index->getBaseData() + idxj * index->getBaseDim(),
                                                    index->getBaseDim());
                tmp.emplace_back(idxj, d);
            }
            std::sort(tmp.begin(), tmp.end());
            if (tmp.size() > index->edgeSizeForCreation) {
                tmp.resize(index->edgeSizeForCreation);
            }

            index->getFinalGraph()[idxi] = tmp;
        }

        // 逐个进行插入操作
        for (unsigned i = 0; i < index->getBaseLen(); i++) {
            InsertNode(i);
        }
    }

    void ComponentInitANNG::InsertNode(unsigned id) {
        std::queue<unsigned> truncateQueue;

        for (unsigned i = 0; i < index->getFinalGraph()[id].size(); i++) {
            assert(index->getFinalGraph()[id][i].id != id);

            if (addEdge(index->getFinalGraph()[id][i].id, id, index->getFinalGraph()[id][i].distance)) {
                truncateQueue.push(index->getFinalGraph()[id][i].id);
            }
        }

        while (!truncateQueue.empty()) {
            unsigned tid = truncateQueue.front();
            truncateEdgesOptimally(tid, index->edgeSizeForCreation);
            truncateQueue.pop();
        }
    }

    bool ComponentInitANNG::addEdge(unsigned target, unsigned addID, float dist) {
        Index::SimpleNeighbor obj(addID, dist);

        auto ni = std::lower_bound(index->getFinalGraph()[target].begin(), index->getFinalGraph()[target].end(), obj);
        if ((ni != index->getFinalGraph()[target].end()) && (ni->id == addID)) {
            std::cout << "NGT::addEdge: already existed! " << ni->id << ":" << addID << std::endl;
        } else {
            index->getFinalGraph()[target].insert(ni, obj);
        }

        if (index->truncationThreshold != 0 && index->getFinalGraph()[target].size() > index->truncationThreshold) {
            return true;
        }
        return false;
    }

    void ComponentInitANNG::truncateEdgesOptimally(unsigned id, size_t truncationSize) {
        std::vector<Index::SimpleNeighbor> delNodes;
        size_t osize = index->getFinalGraph()[id].size();

        for (size_t i = truncationSize; i < osize; i++) {
            if (id == index->getFinalGraph()[id][i].id) {
                continue;
            }
            delNodes.push_back(index->getFinalGraph()[id][i]);
        }

        auto ri = index->getFinalGraph()[id].begin();
        ri += truncationSize;
        index->getFinalGraph()[id].erase(ri, index->getFinalGraph()[id].end());

        for (size_t i = 0; i < delNodes.size(); i++) {
            for (auto j = index->getFinalGraph()[delNodes[i].id].begin();
                 j != index->getFinalGraph()[delNodes[i].id].end(); j++) {
                if ((*j).id == id) {
                    index->getFinalGraph()[delNodes[i].id].erase(j);
                    break;
                }
            }
        }

        for (unsigned i = 0; i < delNodes.size(); i++) {
            std::vector<Index::SimpleNeighbor> pool;
            Search(id, delNodes[i].id, pool);

            Index::SimpleNeighbor nearest = pool.front();
            if (nearest.id != delNodes[i].id) {
                unsigned tid = delNodes[i].id;
                auto iter = std::lower_bound(index->getFinalGraph()[tid].begin(), index->getFinalGraph()[tid].end(),
                                             nearest);
                if ((*iter).id != nearest.id) {
                    index->getFinalGraph()[tid].insert(iter, nearest);
                }

                Index::SimpleNeighbor obj(tid, delNodes[i].distance);
                index->getFinalGraph()[nearest.id].push_back(obj);
                std::sort(index->getFinalGraph()[nearest.id].begin(), index->getFinalGraph()[nearest.id].end());
            }
        }
    }

    void ComponentInitANNG::Search(unsigned startId, unsigned query, std::vector<Index::SimpleNeighbor> &pool) {
        unsigned edgeSize = index->edgeSizeForSearch;
        float radius = 3.402823466e+38F;
        float explorationRadius = index->explorationCoefficient * radius;

        // 大顶堆
        std::priority_queue<Index::SimpleNeighbor, std::vector<Index::SimpleNeighbor>, std::less<Index::SimpleNeighbor>> result;
        std::priority_queue<Index::SimpleNeighbor, std::vector<Index::SimpleNeighbor>, std::greater<Index::SimpleNeighbor>> unchecked;
        std::unordered_set<unsigned> distanceChecked;

        float d = index->getDist()->compare(index->getBaseData() + index->getBaseDim() * startId,
                                            index->getBaseData() + index->getBaseDim() * query,
                                            index->getBaseDim());
        Index::SimpleNeighbor obj(startId, d);
//        unchecked.push(obj);      //bug here
//        result.push(obj);
//        distanceChecked.insert(startId);
//
//        while (!unchecked.empty()) {
//            Index::SimpleNeighbor target = unchecked.top();
//            unchecked.pop();
//
//            if (target.distance > explorationRadius) {
//                break;
//            }
//
//            if (index->getFinalGraph()[target.id].empty()) continue;
//            unsigned neighborSize = index->getFinalGraph()[target.id].size() < edgeSize ?
//                                    index->getFinalGraph()[target.id].size() : edgeSize;
//
//            for (int i = 0; i < neighborSize; i++) {
//                if (distanceChecked.find(index->getFinalGraph()[target.id][i].id) != distanceChecked.end())
//                    continue;
//
//                distanceChecked.insert(index->getFinalGraph()[target.id][i].id);
//                float dist = index->getDist()->compare(
//                        index->getBaseData() + index->getBaseDim() * index->getFinalGraph()[target.id][i].id,
//                        index->getBaseData() + index->getBaseDim() * query,
//                        index->getBaseDim());
//                if (dist <= explorationRadius) {
//                    unchecked.push(Index::SimpleNeighbor(index->getFinalGraph()[target.id][i].id, dist));
//                    if (dist <= radius) {
//                        result.push(Index::SimpleNeighbor(index->getFinalGraph()[target.id][i].id, dist));
//                        if (result.size() > index->size) {
//                            if (result.top().distance >= dist) {
//                                if (result.size() > index->size) {
//                                    result.pop();
//                                }
//                                radius = result.top().distance;
//                                explorationRadius = index->explorationCoefficient * radius;
//                            }
//                        }
//                    }
//                }
//            }
//        }
//
//        for (int i = 0; i < result.size(); i++) {
//            pool.push_back(result.top());
//            result.pop();
//        }
//        std::sort(pool.begin(), pool.end());
    }


    // ANNG_new_new
    void ComponentInitPANNG::InitInner() {
        SetConfigs();

        index->nodes_.resize(index->getBaseLen());
        Index::HnswNode *first = new Index::HnswNode(0, 0, index->NN_, index->NN_);
        index->nodes_[0] = first;
        index->enterpoint_ = first;
        std::vector<unsigned> obj = {0};
        MakeVPTree(obj);
#pragma omp parallel num_threads(index->n_threads_)
        {
            auto *visited_list = new Index::VisitedList(index->getBaseLen());
#pragma omp for schedule(dynamic, 128)
            for (size_t i = 1; i < index->getBaseLen(); ++i) {
                auto *qnode = new Index::HnswNode(i, 0, index->NN_, index->NN_);
                index->nodes_[i] = qnode;
                InsertNode(qnode, visited_list);
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
            }
        }
    }

    void ComponentInitPANNG::SetConfigs() {
        index->NN_ = index->getParam().get<unsigned>("NN");
        index->ef_construction_ = index->getParam().get<unsigned>("ef_construction");
        index->n_threads_ = index->getParam().get<unsigned>("n_threads_");
    }

    void ComponentInitPANNG::InsertNode(Index::HnswNode *qnode, Index::VisitedList *visited_list) {
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

    void ComponentInitPANNG::SearchAtLayer(Index::HnswNode *qnode, Index::HnswNode *enterpoint, int level,
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

    void ComponentInitPANNG::Link(Index::HnswNode *source, Index::HnswNode *target, int level) {
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
        std::priority_queue<Index::FurtherFirst>().swap(tempres);
    }

    // Construct the tree from given objects set
    void ComponentInitPANNG::MakeVPTree(const std::vector<unsigned>& objects)
    {
        index->vp_tree.m_root.reset();

#pragma omp parallel
        {
#pragma omp single
#pragma omp task
            index->vp_tree.m_root = MakeVPTree(objects, index->vp_tree.m_root);
        }
    }

    void ComponentInitPANNG::InsertSplitLeafRoot(Index::VPNodePtr& root, const unsigned& new_value)
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
    void ComponentInitPANNG::CollectObjects(const Index::VPNodePtr& node, std::vector<unsigned>& S)
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
    const float ComponentInitPANNG::MedianSumm(const std::vector<unsigned>& SS1, const std::vector<unsigned>& SS2, const unsigned& v) const
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
    float ComponentInitPANNG::Median(const unsigned& value, const std::vector<unsigned>::const_iterator it_begin,
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
    void ComponentInitPANNG::RedistributeAmongLeafNodes(const Index::VPNodePtr& parent_node, const unsigned& new_value)
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
    void ComponentInitPANNG::SplitLeafNode(const Index::VPNodePtr& parent_node, const size_t child_id, const unsigned& new_value)
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

    void ComponentInitPANNG::Remove(const unsigned& query_value, const Index::VPNodePtr& node)
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
    void ComponentInitPANNG::RedistributeAmongNonLeafNodes(const Index::VPNodePtr& parent_node, const size_t k_id,
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

    void ComponentInitPANNG::InsertSplitRoot(Index::VPNodePtr& root, const unsigned& new_value)
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
    const unsigned& ComponentInitPANNG::SelectVP(const std::vector<unsigned>& objects) {
        assert(!objects.empty());

        return *objects.begin();
    }

    // Construct the tree from given objects set
    Index::VPNodePtr ComponentInitPANNG::MakeVPTree(const std::vector<unsigned>& objects, const Index::VPNodePtr& parent){

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
    void ComponentInitPANNG::SplitNonLeafNode(const Index::VPNodePtr& parent_node, const size_t child_id, const unsigned& new_value)
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

    void ComponentInitPANNG::Insert(const unsigned& new_value, Index::VPNodePtr& root) {
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

    void ComponentInitPANNG::Insert(const unsigned int &new_value) {
        //std::cout << "Insert data to vptree.root" << std::endl;

        Insert(new_value, index->vp_tree.m_root);
    }


    // ANNG_new
    void ComponentInitANNG_new::InitInner() {
        SetConfigs();

        Build();

        for(int i = 0; i < 100; i ++) {
            for(int j = 0; j < index->getFinalGraph()[i].size(); j ++) {
                std::cout << index->getFinalGraph()[i][j].id << "|" << index->getFinalGraph()[i][j].distance << " ";
            }
            std::cout << std::endl;
        }
    }

    void ComponentInitANNG_new::SetConfigs() {
        index->edgeSizeForCreation = index->getParam().get<unsigned>("edgeSizeForCreation");

        index->batchSizeForCreation = 200;
        //index->batchSizeForCreation = index->getParam().get<unsigned>("batchSizeForCreation");
    }

    unsigned ComponentInitANNG_new::searchMultipleQueryForCreation(unsigned &id, Index::OutputJobQueue &output) {
        size_t cnt = 0;
        for (; id < index->getBaseLen(); id++) {
            output.pushBack(id);
            cnt++;
            if (cnt >= index->batchSizeForCreation) {
                id++;
                break;
            }
        } // for
        return cnt;
    }

//    bool addEdge(ObjectID target, ObjectID addID, Distance addDistance, bool identityCheck = true)
//    {
//        size_t minsize = 0;
//        GraphNode &node = property.truncationThreshold == 0 ? *getNode(target) : *getNode(target, minsize);
//        addEdge(node, addID, addDistance, identityCheck);
//        if ((size_t)property.truncationThreshold != 0 && node.size() - minsize >
//                                                         (size_t)property.truncationThreshold)
//        {
//            return true;
//        }
//        return false;
//    }
//
    void ComponentInitANNG_new::insertMultipleSearchResults(Index::OutputJobQueue &output, unsigned cnt) {
        // compute distances among all of the resultant objects
        // This processing occupies about 30% of total indexing time when batch size is 200.
        // Only initial batch objects should be connected for each other.
        // The number of nodes in the graph is checked to know whether the batch is initial.
        //size_t size = NeighborhoodGraph::property.edgeSizeForCreation;
        unsigned size = index->edgeSizeForCreation;
        // add distances from a current object to subsequence objects to imitate of sequential insertion.

        sort(output.begin(), output.end()); // sort by batchIdx

        for (unsigned idxi = 0; idxi < cnt; idxi++) {
            // add distances
            std::vector<Index::SimpleNeighbor> tmp;
            for (unsigned idxj = 0; idxj < idxi; idxj++) {
                float dist = index->getDist()->compare(index->getBaseData() + index->getBaseDim() * output[idxi],
                                                       index->getBaseData() + index->getBaseDim() * output[idxj],
                                                       index->getBaseDim());
                index->getFinalGraph()[output[idxi]].push_back(Index::SimpleNeighbor(output[idxj], dist));
            }
            // sort and cut excess edges
            std::sort(index->getFinalGraph()[output[idxi]].begin(), index->getFinalGraph()[output[idxi]].end());
            if (index->getFinalGraph()[output[idxi]].size() > size) {
                index->getFinalGraph()[output[idxi]].resize(size);
            }
        } // for (size_t idxi ....
        // insert resultant objects into the graph as edges
        for(size_t i = 0; i < cnt; i ++) {
            std::queue<unsigned> truncateQueue;
            for (auto ri = index->getFinalGraph()[output[i]].begin(); ri != index->getFinalGraph()[output[i]].end(); ri++)
            {
                assert(output[i] != (*ri).id);
                //if (addEdge((*ri).id, id, (*ri).distance))
                {
                    truncateQueue.push((*ri).id);
                }
            }
            while (!truncateQueue.empty())
            {
                //ObjectID tid = truncateQueue.front();
                //truncateEdges(tid);
                truncateQueue.pop();
            }
        }
//        for (size_t i = 0; i < cnt; i++) {
//            unsigned gr = output[i];
//            std::cout << "gr " << gr << " " << index->getFinalGraph()[gr].size() << std::endl;
//            if (gr > index->edgeSizeForCreation &&
//                index->getFinalGraph()[gr].size() < size) {
//                std::cerr
//                        << "createIndex: Warning. The specified number of edges could not be acquired, because the pruned parameter [-S] might be set."
//                        << std::endl;
//                std::cerr << "  The node id=" << gr << std::endl;
//                std::cerr << "  The number of edges for the node=" << index->getFinalGraph()[gr].size() << std::endl;
//                std::cerr << "  The pruned parameter (edgeSizeForSearch [-S])=" << index->edgeSizeForSearch
//                          << std::endl;
//            }
//        }
    }

    void ComponentInitANNG_new::Build() {

        index->getFinalGraph().resize(index->getBaseLen());

        if (index->edgeSizeForCreation == 0) {
            return;
        }

        unsigned count = 0;

        unsigned pathAdjustCount = 0;

        Index::OutputJobQueue output;

        unsigned id = 0;
        for (;;) {
            size_t cnt = searchMultipleQueryForCreation(id, output);

//            std::cout << id << std::endl;

            if (cnt == 0) {
                break;
            }

            if (output.size() != cnt) {
                std::cerr << "NNTGIndex::insertGraphIndexByThread: Warning!! Thread response size is wrong."
                          << std::endl;
                cnt = output.size();
            }

            insertMultipleSearchResults(output, cnt);

            if(id == index->batchSizeForCreation) {
                std::vector<unsigned> obj;
                for(int i = 0; i < cnt; i ++) {
                    obj.push_back(output[i]);
                }
                MakeVPTree(obj);
            } else {
                for (size_t i = 0; i < cnt; i++) {
                    unsigned job = output[i];
                    if (((index->getFinalGraph()[job].size() > 0) && (index->getFinalGraph()[job][0].distance != 0.0)) ||
                        (index->getFinalGraph()[job].size() == 0)) {
                        Insert(job);
//                        Index::DVPTree::InsertContainer tiobj(index->getBaseData() + index->getBaseDim() * job, job);
//                        index->dvp.insert(tiobj);
                    }
                } // for
            }

            while (!output.empty()) {
                output.pop_front();
            }

            count += cnt;
//            if (pathAdjustCount > 0 && pathAdjustCount <= count)
//            {
//                adjustPathsEffectively();
//                pathAdjustCount += property.pathAdjustmentInterval;
//            }
        }
    }

    // Construct the tree from given objects set
    void ComponentInitANNG_new::MakeVPTree(const std::vector<unsigned>& objects)
    {
        index->vp_tree.m_root.reset();

#pragma omp parallel
        {
#pragma omp single
#pragma omp task
            index->vp_tree.m_root = MakeVPTree(objects, index->vp_tree.m_root);
        }
    }

    void ComponentInitANNG_new::InsertSplitLeafRoot(Index::VPNodePtr& root, const unsigned& new_value)
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
    void ComponentInitANNG_new::CollectObjects(const Index::VPNodePtr& node, std::vector<unsigned>& S)
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
    const float ComponentInitANNG_new::MedianSumm(const std::vector<unsigned>& SS1, const std::vector<unsigned>& SS2, const unsigned& v) const
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
    float ComponentInitANNG_new::Median(const unsigned& value, const std::vector<unsigned>::const_iterator it_begin,
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
    void ComponentInitANNG_new::RedistributeAmongLeafNodes(const Index::VPNodePtr& parent_node, const unsigned& new_value)
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
    void ComponentInitANNG_new::SplitLeafNode(const Index::VPNodePtr& parent_node, const size_t child_id, const unsigned& new_value)
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

    void ComponentInitANNG_new::Remove(const unsigned& query_value, const Index::VPNodePtr& node)
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
    void ComponentInitANNG_new::RedistributeAmongNonLeafNodes(const Index::VPNodePtr& parent_node, const size_t k_id,
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

    void ComponentInitANNG_new::InsertSplitRoot(Index::VPNodePtr& root, const unsigned& new_value)
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
    const unsigned& ComponentInitANNG_new::SelectVP(const std::vector<unsigned>& objects) {
        assert(!objects.empty());

        return *objects.begin();
    }

    // Construct the tree from given objects set
    Index::VPNodePtr ComponentInitANNG_new::MakeVPTree(const std::vector<unsigned>& objects, const Index::VPNodePtr& parent){

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
    void ComponentInitANNG_new::SplitNonLeafNode(const Index::VPNodePtr& parent_node, const size_t child_id, const unsigned& new_value)
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

    void ComponentInitANNG_new::Insert(const unsigned& new_value, Index::VPNodePtr& root) {
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

    void ComponentInitANNG_new::Insert(const unsigned int &new_value) {
        //std::cout << "Insert data to vptree.root" << std::endl;

        Insert(new_value, index->vp_tree.m_root);
    }


    // SPTAG KDT new
    void ComponentInitSPTAG_KDT_new::InitInner() {
        SetConfigs();

        BuildTrees();

        BuildGraph();

//        for(int i = 0; i < 10; i ++) {
//            std::cout << "len : " << index->getFinalGraph()[i].size() << std::endl;
//            for(int j = 0; j < index->getFinalGraph()[i].size(); j ++){
//                std::cout << index->getFinalGraph()[i][j].id << "|" << index->getFinalGraph()[i][j].distance << " ";
//            }
//            std::cout << std::endl;
//        }
    }

    void ComponentInitSPTAG_KDT_new::SetConfigs() {
        index->numOfThreads = index->getParam().get<unsigned>("numOfThreads");

        index->m_iTreeNumber = index->getParam().get<unsigned>("KDTNumber");

        index->m_iTPTNumber = index->getParam().get<unsigned>("TPTNumber");

        index->m_iTPTLeafSize = index->getParam().get<unsigned>("TPTLeafSize");

        index->m_iNeighborhoodSize = index->getParam().get<unsigned>("NeighborhoodSize");

        index->m_iNeighborhoodScale = index->getParam().get<unsigned>("GraphNeighborhoodScale");

        index->m_iCEF = index->getParam().get<unsigned>("CEF");
    }

    void ComponentInitSPTAG_KDT_new::BuildTrees() {
        std::vector<int> localindices;

        localindices.resize(index->getBaseLen());
        for (int i = 0; i < localindices.size(); i++) localindices[i] = i;

        // 保存所有 KDT 结构
        index->m_pKDTreeRoots.resize(index->m_iTreeNumber * localindices.size());
        // 保存 KDT 树根
        index->m_pTreeStart.resize(index->m_iTreeNumber, 0);
#pragma omp parallel for num_threads(index->numOfThreads)
        for (int i = 0; i < index->m_iTreeNumber; i++) {
            Sleep(i * 100);
            std::srand(clock());

            std::vector<int> pindices(localindices.begin(), localindices.end());
            std::random_shuffle(pindices.begin(), pindices.end());

            index->m_pTreeStart[i] = i * pindices.size();
            std::cout << "Start to build KDTree " << i + 1 << std::endl;
            int iTreeSize = index->m_pTreeStart[i];

            // 分治生成 KDT
            DivideTree(pindices, 0, pindices.size() - 1, index->m_pTreeStart[i], iTreeSize);
            std::cout << i + 1 << " KDTree built, " << iTreeSize - index->m_pTreeStart[i] << " " << pindices.size();
        }
    }

    int ComponentInitSPTAG_KDT_new::rand(int high, int low) {
        return low + (int) (float(high - low) * (std::rand() / (RAND_MAX + 1.0)));
    }

    int ComponentInitSPTAG_KDT_new::SelectDivisionDimension(const std::vector<float> &varianceValues) {
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
        return topind[ComponentInitSPTAG_KDT_new::rand(num)];
    }

    void ComponentInitSPTAG_KDT_new::ChooseDivision(Index::KDTNode &node, const std::vector<int> &indices,
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

    int ComponentInitSPTAG_KDT_new::Subdivide(const Index::KDTNode &node, std::vector<int> &indices,
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

    void ComponentInitSPTAG_KDT_new::DivideTree(std::vector<int> &indices, int first,
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

    bool ComponentInitSPTAG_KDT_new::Compare(const Index::SimpleNeighbor &lhs, const Index::SimpleNeighbor &rhs) {
        return ((lhs.distance < rhs.distance) || ((lhs.distance == rhs.distance) && (lhs.id < rhs.id)));
    }

    void ComponentInitSPTAG_KDT_new::PartitionByTptree(std::vector<int> &indices, const int first,
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
            std::sort(Variance.begin(), Variance.end(), ComponentInitSPTAG_KDT_new::Compare);
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

    void ComponentInitSPTAG_KDT_new::AddNeighbor(int idx, float dist, int origin) {
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

    void ComponentInitSPTAG_KDT_new::BuildGraph() {
        // 初始化索引尺寸
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

        // 构建 TPTree 和 初始图
        std::vector<std::vector<int>> TptreeDataIndices(index->m_iTPTNumber, std::vector<int>(index->getBaseLen()));
        std::vector<std::vector<std::pair<int, int>>> TptreeLeafNodes(index->m_iTPTNumber,
                                                                      std::vector<std::pair<int, int>>());

        auto t1 = std::chrono::high_resolution_clock::now();
        std::cout << "Parallel TpTree Partition begin\n";
#pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < index->m_iTPTNumber; i++) {
            Sleep(i * 100);
            std::srand(clock());
            for (int j = 0; j < index->getBaseLen(); j++) TptreeDataIndices[i][j] = j;
            std::random_shuffle(TptreeDataIndices[i].begin(), TptreeDataIndices[i].end());
            PartitionByTptree(TptreeDataIndices[i], 0, index->getBaseLen() - 1, TptreeLeafNodes[i]);
            std::cout << "Finish Getting Leaves for Tree " << i << std::endl;
        }
        std::cout << "Parallel TpTree Partition done\n";
        auto t2 = std::chrono::high_resolution_clock::now();
        std::cout << "Build TPTree time (s): " << std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count()
                  << std::endl;

        for (int i = 0; i < index->m_iTPTNumber; i++) {
#pragma omp parallel for schedule(dynamic)
            for (int j = 0; j < (int) TptreeLeafNodes[i].size(); j++) {
                int start_index = TptreeLeafNodes[i][j].first;
                int end_index = TptreeLeafNodes[i][j].second;
                if ((j * 5) % TptreeLeafNodes[i].size() == 0)
                    std::cout << "Processing Tree " << i << " "
                              << static_cast<int>(j * 1.0 / TptreeLeafNodes[i].size() * 100) << std::endl;
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
        std::cout << "Process TPTree time (s): " << std::chrono::duration_cast<std::chrono::seconds>(t3 - t2).count()
                  << std::endl;
    }


    // SPTAG KDT
    void ComponentInitSPTAG_KDT::InitInner() {
        SetConfigs();

        BuildTrees();

        BuildGraph();
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
        std::vector<unsigned> localindices;
        localindices.resize(index->getBaseLen());
        for (unsigned i = 0; i < localindices.size(); i++) localindices[i] = i;

        // 记录 KDT 结构
        index->m_pKDTreeRoots.resize(index->m_iTreeNumber * localindices.size());
        // 记录树根
        index->m_pTreeStart.resize(index->m_iTreeNumber, 0);

#pragma omp parallel for num_threads(index->numOfThreads)
        for (int i = 0; i < index->m_iTreeNumber; i++) {
            // 非多线程 -> 删除 ！！！
            //Sleep(i * 100);
            std::srand(clock());

            std::vector<unsigned> pindices(localindices.begin(), localindices.end());
            std::random_shuffle(pindices.begin(), pindices.end());

            index->m_pTreeStart[i] = i * pindices.size();
            std::cout << "Start to build KDTree " << i + 1 << std::endl;
            unsigned iTreeSize = index->m_pTreeStart[i];
            DivideTree(pindices, 0, pindices.size() - 1, index->m_pTreeStart[i], iTreeSize);
            std::cout << i + 1 << " KDTree built, " << iTreeSize - index->m_pTreeStart[i] << " " << pindices.size()
                      << std::endl;
        }
    }

    void ComponentInitSPTAG_KDT::BuildGraph() {
        index->m_iNeighborhoodSize = index->m_iNeighborhoodSize * index->m_iNeighborhoodScale;       // L

        index->getFinalGraph().resize(index->getBaseLen());

        float MaxDist = (std::numeric_limits<float>::max)();
        unsigned MaxId = (std::numeric_limits<unsigned>::max)();

        for (unsigned i = 0; i < index->getBaseLen(); i++) {
            index->getFinalGraph()[i].resize(index->m_iNeighborhoodSize);
            for (unsigned j = 0; j < index->m_iNeighborhoodSize; j++) {
                Index::SimpleNeighbor neighbor(MaxId, MaxDist);
                index->getFinalGraph()[i][j] = neighbor;
            }
        }

        BuildInitKNNGraph();
    }

    void ComponentInitSPTAG_KDT::BuildInitKNNGraph() {
        std::vector<std::vector<unsigned>> TptreeDataIndices(index->m_iTPTNumber,
                                                             std::vector<unsigned>(index->getBaseLen()));
        std::vector<std::vector<std::pair<unsigned, unsigned>>> TptreeLeafNodes(index->m_iTPTNumber,
                                                                                std::vector<std::pair<unsigned, unsigned>>());

#pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < index->m_iTPTNumber; i++) {
            // 非多线程注意注释
            //Sleep(i * 100);
            std::srand(clock());
            for (unsigned j = 0; j < index->getBaseLen(); j++) TptreeDataIndices[i][j] = j;
            std::random_shuffle(TptreeDataIndices[i].begin(), TptreeDataIndices[i].end());
            PartitionByTptree(TptreeDataIndices[i], 0, index->getBaseLen() - 1, TptreeLeafNodes[i]);
            std::cout << "Finish Getting Leaves for Tree : " << i << std::endl;
        }
        std::cout << "Parallel TpTree Partition done" << std::endl;

        for (int i = 0; i < index->m_iTPTNumber; i++) {
#pragma omp parallel for schedule(dynamic)
            for (unsigned j = 0; j < (unsigned) TptreeLeafNodes[i].size(); j++) {
                unsigned start_index = TptreeLeafNodes[i][j].first;
                unsigned end_index = TptreeLeafNodes[i][j].second;
                if ((j * 5) % TptreeLeafNodes[i].size() == 0)
                    std::cout << "Processing Tree : " << i
                              << static_cast<int>(j * 1.0 / TptreeLeafNodes[i].size() * 100) << std::endl;

                for (unsigned x = start_index; x < end_index; x++) {
                    for (unsigned y = x + 1; y <= end_index; y++) {
                        unsigned p1 = TptreeDataIndices[i][x];
                        unsigned p2 = TptreeDataIndices[i][y];
                        float dist = index->getDist()->compare(index->getBaseData() + index->getBaseDim() * p1,
                                                               index->getBaseData() + index->getBaseDim() * p2,
                                                               index->getBaseDim());

                        AddNeighbor(p2, dist, p1, index->m_iNeighborhoodSize);
                        AddNeighbor(p1, dist, p2, index->m_iNeighborhoodSize);
                    }
                }
            }
            TptreeDataIndices[i].clear();
            TptreeLeafNodes[i].clear();
        }
        TptreeDataIndices.clear();
        TptreeLeafNodes.clear();
    }

    void
    ComponentInitSPTAG_KDT::PartitionByTptree(std::vector<unsigned> &indices, const unsigned first, const unsigned last,
                                              std::vector<std::pair<unsigned, unsigned>> &leaves) {
        unsigned m_numTopDimensionTPTSplit = 5;

        if (last - first <= index->m_iTPTLeafSize) {
            leaves.emplace_back(first, last);
        } else {
            std::vector<float> Mean(index->getBaseDim(), 0);

            int iIteration = 100;
            unsigned end = std::min(first + index->m_iSamples, last);
            unsigned count = end - first + 1;
            // calculate the mean of each dimension
            for (unsigned j = first; j <= end; j++) {
                const float *v = index->getBaseData() + indices[j] * index->getBaseDim();
                for (unsigned k = 0; k < index->getBaseDim(); k++) {
                    Mean[k] += v[k];
                }
            }
            // 计算每个维度平均值
            for (unsigned k = 0; k < index->getBaseDim(); k++) {
                Mean[k] /= count;
            }
            std::vector<Index::SimpleNeighbor> Variance;
            Variance.reserve(index->getBaseDim());
            for (unsigned j = 0; j < index->getBaseDim(); j++) {
                Variance.emplace_back(j, 0.0f);
            }
            // calculate the variance of each dimension
            for (unsigned j = first; j <= end; j++) {
                const float *v = index->getBaseData() + index->getBaseDim() * indices[j];
                for (unsigned k = 0; k < index->getBaseDim(); k++) {
                    float dist = v[k] - Mean[k];
                    Variance[k].distance += dist * dist;
                }
            }
            std::sort(Variance.begin(), Variance.end(), ComponentInitSPTAG_KDT::Compare);
            std::vector<unsigned> indexs(m_numTopDimensionTPTSplit);
            std::vector<float> weight(m_numTopDimensionTPTSplit), bestweight(m_numTopDimensionTPTSplit);
            float bestvariance = Variance[index->getBaseDim() - 1].distance;
            // 选出离散程度更大的 m_numTopDimensionTPTSplit 个维度
            for (int i = 0; i < m_numTopDimensionTPTSplit; i++) {
                indexs[i] = Variance[index->getBaseDim() - 1 - i].id;
                bestweight[i] = 0;
            }
            bestweight[0] = 1;
            float bestmean = Mean[indexs[0]];

            std::vector<float> Val(count);
            for (int i = 0; i < iIteration; i++) {
                float sumweight = 0;
                for (int j = 0; j < m_numTopDimensionTPTSplit; j++) {
                    weight[j] = float(std::rand() % 10000) / 5000.0f - 1.0f;
                    sumweight += weight[j] * weight[j];
                }
                sumweight = sqrt(sumweight);
                for (int j = 0; j < m_numTopDimensionTPTSplit; j++) {
                    weight[j] /= sumweight;
                }
                float mean = 0;
                for (unsigned j = 0; j < count; j++) {
                    Val[j] = 0;
                    const float *v = index->getBaseData() + index->getBaseDim() * indices[first + j];
                    for (int k = 0; k < m_numTopDimensionTPTSplit; k++) {
                        Val[j] += weight[k] * v[indexs[k]];
                    }
                    mean += Val[j];
                }
                mean /= count;
                float var = 0;
                for (unsigned j = 0; j < count; j++) {
                    float dist = Val[j] - mean;
                    var += dist * dist;
                }
                if (var > bestvariance) {
                    bestvariance = var;
                    bestmean = mean;
                    for (int j = 0; j < m_numTopDimensionTPTSplit; j++) {
                        bestweight[j] = weight[j];
                    }
                }
            }
            unsigned i = first;
            unsigned j = last;
            // decide which child one point belongs
            while (i <= j) {
                float val = 0;
                const float *v = index->getBaseData() + index->getBaseDim() * indices[i];
                for (int k = 0; k < m_numTopDimensionTPTSplit; k++) {
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

    void ComponentInitSPTAG_KDT::AddNeighbor(unsigned idx, float dist, unsigned origin, unsigned size) {
        size--;
        if (dist < index->getFinalGraph()[origin][size].distance ||
            (dist == index->getFinalGraph()[origin][size].distance && idx < index->getFinalGraph()[origin][size].id)) {
            unsigned nb;

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

    inline bool ComponentInitSPTAG_KDT::Compare(const Index::SimpleNeighbor &lhs, const Index::SimpleNeighbor &rhs) {
        return ((lhs.distance < rhs.distance) || ((lhs.distance == rhs.distance) && (lhs.id < rhs.id)));
    }

    void ComponentInitSPTAG_KDT::DivideTree(std::vector<unsigned> &indices, unsigned first, unsigned last,
                                            unsigned tree_index, unsigned &iTreeSize) {
        // 选择分离维度
        ChooseDivision(index->m_pKDTreeRoots[tree_index], indices, first, last);
        unsigned i = Subdivide(index->m_pKDTreeRoots[tree_index], indices, first, last);
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

    void ComponentInitSPTAG_KDT::ChooseDivision(Index::KDTNode &node, const std::vector<unsigned> &indices,
                                                const unsigned first, const unsigned last) {

        std::vector<float> meanValues(index->getBaseDim(), 0);
        std::vector<float> varianceValues(index->getBaseDim(), 0);
        unsigned end = std::min(first + index->m_iSamples, last);
        unsigned count = end - first + 1;
        // calculate the mean of each dimension
        for (unsigned j = first; j <= end; j++) {
            float *v = index->getBaseData() + index->getBaseDim() * indices[j];
            for (unsigned k = 0; k < index->getBaseDim(); k++) {
                meanValues[k] += v[k];
            }
        }
        for (unsigned k = 0; k < index->getBaseDim(); k++) {
            meanValues[k] /= count;
        }
        // calculate the variance of each dimension
        for (unsigned j = first; j <= end; j++) {
            const float *v = index->getBaseData() + index->getBaseDim() * indices[j];
            for (unsigned k = 0; k < index->getBaseDim(); k++) {
                float dist = v[k] - meanValues[k];
                varianceValues[k] += dist * dist;
            }
        }
        // choose the split dimension as one of the dimension inside TOP_DIM maximum variance
        node.split_dim = SelectDivisionDimension(varianceValues);
        // determine the threshold
        node.split_value = meanValues[node.split_dim];
    }

    unsigned ComponentInitSPTAG_KDT::rand(unsigned high, unsigned low) {
        return low + (unsigned) (float(high - low) * (std::rand() / (RAND_MAX + 1.0)));
    }

    unsigned ComponentInitSPTAG_KDT::SelectDivisionDimension(const std::vector<float> &varianceValues) {
        unsigned m_numTopDimensionKDTSplit = 5;

        // Record the top maximum variances
        std::vector<unsigned> topind(m_numTopDimensionKDTSplit);
        int num = 0;
        // order the variances
        for (unsigned i = 0; i < varianceValues.size(); i++) {
            if (num < m_numTopDimensionKDTSplit || varianceValues[i] > varianceValues[topind[num - 1]]) {
                if (num < m_numTopDimensionKDTSplit) {
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
        return topind[rand(num)];
    }

    unsigned
    ComponentInitSPTAG_KDT::Subdivide(const Index::KDTNode &node, std::vector<unsigned> &indices, const unsigned first,
                                      const unsigned last) {
        unsigned i = first;
        unsigned j = last;
        // decide which child one point belongs
        while (i <= j) {
            unsigned ind = indices[i];
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

    /**
     * SPTAG_BKT_new
     */
    void ComponentInitSPTAG_BKT_new::InitInner() {
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

    void ComponentInitSPTAG_BKT_new::SetConfigs() {
        index->numOfThreads = index->getParam().get<unsigned>("numOfThreads");

        index->m_iTreeNumber = index->getParam().get<unsigned>("BKTNumber");

        index->m_iBKTKmeansK = index->getParam().get<unsigned>("BKTKMeansK");

        index->m_iNeighborhoodSize = index->getParam().get<unsigned>("NeighborhoodSize");

        index->m_iNeighborhoodScale = index->getParam().get<unsigned>("GraphNeighborhoodScale");

        index->m_iCEF = index->getParam().get<unsigned>("CEF");
    }

    int ComponentInitSPTAG_BKT_new::rand(int high, int low) {
        return low + (int) (float(high - low) * (std::rand() / (RAND_MAX + 1.0)));
    }

    float ComponentInitSPTAG_BKT_new::KmeansAssign(std::vector<int> &indices, const int first,
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

    void ComponentInitSPTAG_BKT_new::InitCenters(std::vector<int> &indices, const int first,
                                                 const int last, Index::KmeansArgs<float> &args, int samples,
                                                 int tryIters) {
        const float MaxDist = (std::numeric_limits<float>::max)();

        int batchEnd = std::min(first + samples, last);
        float currDist, minClusterDist = MaxDist;
        for (int numKmeans = 0; numKmeans < tryIters; numKmeans++) {
            for (int k = 0; k < args._DK; k++) {
                int randid = ComponentInitSPTAG_BKT_new::rand(last, first);
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

    float ComponentInitSPTAG_BKT_new::RefineCenters(Index::KmeansArgs<float> &args) {
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

    int ComponentInitSPTAG_BKT_new::KmeansClustering(std::vector<int> &indices, const int first,
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

    void ComponentInitSPTAG_BKT_new::BuildTrees() {
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

            std::cout << "Start to build BKTree " << i + 1 << std::endl;

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

    bool ComponentInitSPTAG_BKT_new::Compare(const Index::SimpleNeighbor &lhs, const Index::SimpleNeighbor &rhs) {
        return ((lhs.distance < rhs.distance) || ((lhs.distance == rhs.distance) && (lhs.id < rhs.id)));
    }

    void ComponentInitSPTAG_BKT_new::PartitionByTptree(std::vector<int> &indices, const int first,
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
            std::sort(Variance.begin(), Variance.end(), ComponentInitSPTAG_BKT_new::Compare);
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

    void ComponentInitSPTAG_BKT_new::AddNeighbor(int idx, float dist, int origin) {
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

    void ComponentInitSPTAG_BKT_new::BuildGraph() {
        // 初始化索引尺寸
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

        // 构建 TPTree 和 初始图
        std::vector<std::vector<int>> TptreeDataIndices(index->m_iTPTNumber, std::vector<int>(index->getBaseLen()));
        std::vector<std::vector<std::pair<int, int>>> TptreeLeafNodes(index->m_iTPTNumber,
                                                                      std::vector<std::pair<int, int>>());

        auto t1 = std::chrono::high_resolution_clock::now();
        std::cout << "Parallel TpTree Partition begin\n";
#pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < index->m_iTPTNumber; i++) {
            Sleep(i * 100);
            std::srand(clock());
            for (int j = 0; j < index->getBaseLen(); j++) TptreeDataIndices[i][j] = j;
            std::random_shuffle(TptreeDataIndices[i].begin(), TptreeDataIndices[i].end());
            PartitionByTptree(TptreeDataIndices[i], 0, index->getBaseLen() - 1, TptreeLeafNodes[i]);
            std::cout << "Finish Getting Leaves for Tree " << i << std::endl;
        }
        std::cout << "Parallel TpTree Partition done\n";
        auto t2 = std::chrono::high_resolution_clock::now();
        std::cout << "Build TPTree time (s): " << std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count()
                  << std::endl;

        for (int i = 0; i < index->m_iTPTNumber; i++) {
#pragma omp parallel for schedule(dynamic)
            for (int j = 0; j < (int) TptreeLeafNodes[i].size(); j++) {
                int start_index = TptreeLeafNodes[i][j].first;
                int end_index = TptreeLeafNodes[i][j].second;
                if ((j * 5) % TptreeLeafNodes[i].size() == 0)
                    std::cout << "Processing Tree " << i << " "
                              << static_cast<int>(j * 1.0 / TptreeLeafNodes[i].size() * 100) << std::endl;
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
        std::cout << "Process TPTree time (s): " << std::chrono::duration_cast<std::chrono::seconds>(t3 - t2).count()
                  << std::endl;
    }


    // SPTAG_BKT
    void ComponentInitSPTAG_BKT::InitInner() {
        SetConfigs();

        BuildTrees();

        BuildGraph();

        for (int i = 0; i < 10; i++) {
            std::cout << "len : " << index->getFinalGraph()[i].size() << std::endl;
            for (int j = 0; j < index->getFinalGraph()[i].size(); j++) {
                std::cout << index->getFinalGraph()[i][j].id << "|" << index->getFinalGraph()[i][j].distance << " ";
            }
            std::cout << std::endl;
        }
    }

    void ComponentInitSPTAG_BKT::SetConfigs() {
        index->numOfThreads = index->getParam().get<unsigned>("numOfThreads");

        index->m_iTreeNumber = index->getParam().get<unsigned>("BKTNumber");

        index->m_iNeighborhoodSize = index->getParam().get<unsigned>("NeighborhoodSize");

        index->m_iNeighborhoodScale = index->getParam().get<unsigned>("GraphNeighborhoodScale");

        index->m_iCEF = index->getParam().get<unsigned>("CEF");
    }

    void ComponentInitSPTAG_BKT::BuildTrees() {
        struct BKTStackItem {
            unsigned index, first, last;

            BKTStackItem(unsigned index_, unsigned first_, unsigned last_) : index(index_), first(first_),
                                                                             last(last_) {}
        };
        std::stack<BKTStackItem> ss;

        std::vector<unsigned> localindices;
        localindices.resize(index->getBaseLen());
        for (unsigned i = 0; i < localindices.size(); i++) localindices[i] = i;

        Index::KmeansArgs<float> args(index->m_iBKTKmeansK, index->getBaseDim(), localindices.size(),
                                      index->numOfThreads);

        index->m_pSampleCenterMap.clear();

        unsigned m_iBKTLeafSize = 8;
        // 构建 m_iTreeNumber 棵 BKT
        for (int i = 0; i < index->m_iTreeNumber; i++) {
            std::random_shuffle(localindices.begin(), localindices.end());

            index->m_pTreeStart.push_back(index->m_pBKTreeRoots.size());
            index->m_pBKTreeRoots.emplace_back(localindices.size());
            std::cout << "Start to build BKTree : " << i + 1 << std::endl;

            // 分治构建 BKT
            ss.push(BKTStackItem(index->m_pTreeStart[i], 0, localindices.size()));
            while (!ss.empty()) {
                BKTStackItem item = ss.top();
                ss.pop();
                unsigned newBKTid = (unsigned) index->m_pBKTreeRoots.size();
                index->m_pBKTreeRoots[item.index].childStart = newBKTid;

                // 小于叶子结点个数
                if (item.last - item.first <= m_iBKTLeafSize) {
                    for (unsigned j = item.first; j < item.last; j++) {
                        unsigned cid = localindices[j];
                        index->m_pBKTreeRoots.emplace_back(cid);
                    }
                } else { // clustering the data into BKTKmeansK clusters
                    int numClusters = KmeansClustering(localindices, item.first, item.last, args, index->m_iSamples);
                    //std::cout << "wtf" << std::endl;
                    if (numClusters <= 1) {
                        //std::cout << "wtf" << std::endl;
                        unsigned end = std::min(item.last + 1, (unsigned) localindices.size());
                        std::sort(localindices.begin() + item.first, localindices.begin() + end);
                        index->m_pBKTreeRoots[item.index].centerid = localindices[item.first];
                        index->m_pBKTreeRoots[item.index].childStart = -index->m_pBKTreeRoots[item.index].childStart;
                        for (unsigned j = item.first + 1; j < end; j++) {
                            unsigned cid = localindices[j];
                            index->m_pBKTreeRoots.emplace_back(cid);
                            index->m_pSampleCenterMap[cid] = index->m_pBKTreeRoots[item.index].centerid;
                        }
                        index->m_pSampleCenterMap[-1 - index->m_pBKTreeRoots[item.index].centerid] = item.index;
                    } else {
                        //std::cout << "wtf" << std::endl;
                        for (int k = 0; k < index->m_iBKTKmeansK; k++) {
                            //std::cout << k << std::endl;
                            if (args.counts[k] == 0) continue;
                            unsigned cid = localindices[item.first + args.counts[k] - 1];
                            index->m_pBKTreeRoots.emplace_back(cid);
                            if (args.counts[k] > 1)
                                ss.push(BKTStackItem(newBKTid++, item.first, item.first + args.counts[k] - 1));
                            item.first += args.counts[k];
                        }
                    }
                }
                //std::cout << "wtf" << std::endl;
                index->m_pBKTreeRoots[item.index].childEnd = (unsigned) index->m_pBKTreeRoots.size();
                //std::cout << "wtf" << std::endl;
            }
            index->m_pBKTreeRoots.emplace_back(-1);
            std::cout << i + 1 << " BKTree built, " << index->m_pBKTreeRoots.size() - index->m_pTreeStart[i] << " "
                      << localindices.size() << std::endl;
        }
    }

    void ComponentInitSPTAG_BKT::BuildGraph() {
        index->m_iNeighborhoodSize = index->m_iNeighborhoodSize * index->m_iNeighborhoodScale;       // L

        index->getFinalGraph().resize(index->getBaseLen());

        float MaxDist = (std::numeric_limits<float>::max)();
        unsigned MaxId = (std::numeric_limits<unsigned>::max)();

        for (unsigned i = 0; i < index->getBaseLen(); i++) {
            index->getFinalGraph()[i].resize(index->m_iNeighborhoodSize);
            for (unsigned j = 0; j < index->m_iNeighborhoodSize; j++) {
                Index::SimpleNeighbor neighbor(MaxId, MaxDist);
                index->getFinalGraph()[i][j] = neighbor;
            }
        }

        BuildInitKNNGraph();
    }

    void ComponentInitSPTAG_BKT::BuildInitKNNGraph() {
        std::vector<std::vector<unsigned>> TptreeDataIndices(index->m_iTPTNumber,
                                                             std::vector<unsigned>(index->getBaseLen()));
        std::vector<std::vector<std::pair<unsigned, unsigned>>> TptreeLeafNodes(index->m_iTPTNumber,
                                                                                std::vector<std::pair<unsigned, unsigned>>());

#pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < index->m_iTPTNumber; i++) {
            // 非多线程注意注释
            //Sleep(i * 100);
            std::srand(clock());
            for (unsigned j = 0; j < index->getBaseLen(); j++) TptreeDataIndices[i][j] = j;
            std::random_shuffle(TptreeDataIndices[i].begin(), TptreeDataIndices[i].end());
            PartitionByTptree(TptreeDataIndices[i], 0, index->getBaseLen() - 1, TptreeLeafNodes[i]);
            std::cout << "Finish Getting Leaves for Tree : " << i << std::endl;
        }
        std::cout << "Parallel TpTree Partition done" << std::endl;

        for (int i = 0; i < index->m_iTPTNumber; i++) {
#pragma omp parallel for schedule(dynamic)
            for (unsigned j = 0; j < (unsigned) TptreeLeafNodes[i].size(); j++) {
                unsigned start_index = TptreeLeafNodes[i][j].first;
                unsigned end_index = TptreeLeafNodes[i][j].second;
                if ((j * 5) % TptreeLeafNodes[i].size() == 0)
                    std::cout << "Processing Tree : " << i
                              << static_cast<int>(j * 1.0 / TptreeLeafNodes[i].size() * 100) << std::endl;

                for (unsigned x = start_index; x < end_index; x++) {
                    for (unsigned y = x + 1; y <= end_index; y++) {
                        unsigned p1 = TptreeDataIndices[i][x];
                        unsigned p2 = TptreeDataIndices[i][y];
                        float dist = index->getDist()->compare(index->getBaseData() + index->getBaseDim() * p1,
                                                               index->getBaseData() + index->getBaseDim() * p2,
                                                               index->getBaseDim());

                        AddNeighbor(p2, dist, p1, index->m_iNeighborhoodSize);
                        AddNeighbor(p1, dist, p2, index->m_iNeighborhoodSize);
                    }
                }
            }
            TptreeDataIndices[i].clear();
            TptreeLeafNodes[i].clear();
        }
        TptreeDataIndices.clear();
        TptreeLeafNodes.clear();
    }

    int
    ComponentInitSPTAG_BKT::KmeansClustering(std::vector<unsigned> &indices, const unsigned first, const unsigned last,
                                             Index::KmeansArgs<float> &args, int samples) {

//        const float MaxDist = (std::numeric_limits<float>::max)();
//
//        // 随机选取 _DK 个向量作为簇中心
//        InitCenters(indices, first, last, args, samples, 3);
//
//        unsigned batchEnd = std::min(first + samples, last);
//        float currDiff, currDist, minClusterDist = MaxDist;
//        int noImprovement = 0;
//        for (int iter = 0; iter < 100; iter++) {
//            std::memcpy(args.centers, args.newTCenters, sizeof(float) * args._K * args._D);
//            std::random_shuffle(indices.begin() + first, indices.begin() + last);
//
//            args.ClearCenters();
//            args.ClearCounts();
//            args.ClearDists(-MaxDist);
//            currDist = KmeansAssign(indices, first, batchEnd, args, true, 1 / (100.0f * (batchEnd - first)));
////            std::cout << "2" << std::endl;
////            for(int i = 0; i < args._DK; i ++) {
////                std::cout << args.clusterIdx[i] << " ";
////            }
////            std::cout << std::endl;
//            std::memcpy(args.counts, args.newCounts, sizeof(unsigned) * args._K);
//
//            if (currDist < minClusterDist) {
//                noImprovement = 0;
//                minClusterDist = currDist;
//            } else {
//                noImprovement++;
//            }
//            currDiff = RefineCenters(args);
//            if (currDiff < 1e-3 || noImprovement >= 5) break;
//        }
//
//        //std::cout << "wt2f" << std::endl;
//
//        args.ClearCounts();
//        args.ClearDists(MaxDist);
//        currDist = KmeansAssign(indices, first, last, args, false, 0);
//        std::memcpy(args.counts, args.newCounts, sizeof(unsigned) * args._K);
//
//        //std::cout << "wt2f" << std::endl;
//
//        int numClusters = 0;
//        for (int i = 0; i < args._K; i++) if (args.counts[i] > 0) numClusters++;
//
//        if (numClusters <= 1) {
//            return numClusters;
//        }
//        args.Shuffle(indices, first, last);
//        return numClusters;
        return 0;
    }

    void
    ComponentInitSPTAG_BKT::PartitionByTptree(std::vector<unsigned> &indices, const unsigned first, const unsigned last,
                                              std::vector<std::pair<unsigned, unsigned>> &leaves) {
        unsigned m_numTopDimensionTPTSplit = 5;

        if (last - first <= index->m_iTPTLeafSize) {
            leaves.emplace_back(first, last);
        } else {
            std::vector<float> Mean(index->getBaseDim(), 0);

            int iIteration = 100;
            unsigned end = std::min(first + index->m_iSamples, last);
            unsigned count = end - first + 1;
            // calculate the mean of each dimension
            for (unsigned j = first; j <= end; j++) {
                const float *v = index->getBaseData() + indices[j] * index->getBaseDim();
                for (unsigned k = 0; k < index->getBaseDim(); k++) {
                    Mean[k] += v[k];
                }
            }
            // 计算每个维度平均值
            for (unsigned k = 0; k < index->getBaseDim(); k++) {
                Mean[k] /= count;
            }
            std::vector<Index::SimpleNeighbor> Variance;
            Variance.reserve(index->getBaseDim());
            for (unsigned j = 0; j < index->getBaseDim(); j++) {
                Variance.emplace_back(j, 0.0f);
            }
            // calculate the variance of each dimension
            for (unsigned j = first; j <= end; j++) {
                const float *v = index->getBaseData() + index->getBaseDim() * indices[j];
                for (unsigned k = 0; k < index->getBaseDim(); k++) {
                    float dist = v[k] - Mean[k];
                    Variance[k].distance += dist * dist;
                }
            }
            std::sort(Variance.begin(), Variance.end(), ComponentInitSPTAG_BKT::Compare);
            std::vector<unsigned> indexs(m_numTopDimensionTPTSplit);
            std::vector<float> weight(m_numTopDimensionTPTSplit), bestweight(m_numTopDimensionTPTSplit);
            float bestvariance = Variance[index->getBaseDim() - 1].distance;
            // 选出离散程度更大的 m_numTopDimensionTPTSplit 个维度
            for (int i = 0; i < m_numTopDimensionTPTSplit; i++) {
                indexs[i] = Variance[index->getBaseDim() - 1 - i].id;
                bestweight[i] = 0;
            }
            bestweight[0] = 1;
            float bestmean = Mean[indexs[0]];

            std::vector<float> Val(count);
            for (int i = 0; i < iIteration; i++) {
                float sumweight = 0;
                for (int j = 0; j < m_numTopDimensionTPTSplit; j++) {
                    weight[j] = float(std::rand() % 10000) / 5000.0f - 1.0f;
                    sumweight += weight[j] * weight[j];
                }
                sumweight = sqrt(sumweight);
                for (int j = 0; j < m_numTopDimensionTPTSplit; j++) {
                    weight[j] /= sumweight;
                }
                float mean = 0;
                for (unsigned j = 0; j < count; j++) {
                    Val[j] = 0;
                    const float *v = index->getBaseData() + index->getBaseDim() * indices[first + j];
                    for (int k = 0; k < m_numTopDimensionTPTSplit; k++) {
                        Val[j] += weight[k] * v[indexs[k]];
                    }
                    mean += Val[j];
                }
                mean /= count;
                float var = 0;
                for (unsigned j = 0; j < count; j++) {
                    float dist = Val[j] - mean;
                    var += dist * dist;
                }
                if (var > bestvariance) {
                    bestvariance = var;
                    bestmean = mean;
                    for (int j = 0; j < m_numTopDimensionTPTSplit; j++) {
                        bestweight[j] = weight[j];
                    }
                }
            }
            unsigned i = first;
            unsigned j = last;
            // decide which child one point belongs
            while (i <= j) {
                float val = 0;
                const float *v = index->getBaseData() + index->getBaseDim() * indices[i];
                for (int k = 0; k < m_numTopDimensionTPTSplit; k++) {
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

    inline bool ComponentInitSPTAG_BKT::Compare(const Index::SimpleNeighbor &lhs, const Index::SimpleNeighbor &rhs) {
        return ((lhs.distance < rhs.distance) || ((lhs.distance == rhs.distance) && (lhs.id < rhs.id)));
    }

    void ComponentInitSPTAG_BKT::AddNeighbor(unsigned idx, float dist, unsigned origin, unsigned size) {
        size--;
        if (dist < index->getFinalGraph()[origin][size].distance ||
            (dist == index->getFinalGraph()[origin][size].distance && idx < index->getFinalGraph()[origin][size].id)) {
            unsigned nb;

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

    float ComponentInitSPTAG_BKT::RefineCenters(Index::KmeansArgs<float> &args) {
        int maxcluster = -1;
        unsigned maxCount = 0;
        for (int k = 0; k < args._DK; k++) {
            unsigned test = -1;
            //std::cout << "k : " << k << " " << args._DK << " " << args.clusterIdx[k] << " " << test << std::endl;

            if (args.clusterIdx[k] < 0 || args.clusterIdx[k] > index->getBaseLen()) continue;
            float dist = index->getDist()->compare(index->getBaseData() + index->getBaseDim() * args.clusterIdx[k],
                                                   args.centers + k * args._D,
                                                   args._D);
            if (args.counts[k] > maxCount && args.newCounts[k] > 0 && dist > 1e-6) {
                maxcluster = k;
                maxCount = args.counts[k];
            }
        }

        if (maxcluster != -1 && (args.clusterIdx[maxcluster] < 0 || args.clusterIdx[maxcluster] >= index->getBaseLen()))
            std::cout << "maxcluster:" << maxcluster << "(" << args.newCounts[maxcluster] << ")" << " Error dist:"
                      << args.clusterDist[maxcluster] << std::endl;

        //std::cout << "11" << std::endl;
        float diff = 0;
        for (int k = 0; k < args._DK; k++) {
            float *TCenter = args.newTCenters + k * args._D;
            if (args.counts[k] == 0) {
                //std::cout << "wtf7" << std::endl;
                if (maxcluster != -1) {
                    //int nextid = Utils::rand_int(last, first);
                    //while (args.label[nextid] != maxcluster) nextid = Utils::rand_int(last, first);
                    int nextid = args.clusterIdx[maxcluster];
                    std::memcpy(TCenter, index->getBaseData() + index->getBaseDim() * nextid, sizeof(float) * args._D);
                } else {
                    std::memcpy(TCenter, args.centers + k * args._D, sizeof(float) * args._D);
                }
                //std::cout << "wtf7" << std::endl;
            } else {
                //std::cout << "wtf8" << std::endl;
                float *currCenters = args.newCenters + k * args._D;
                for (unsigned j = 0; j < args._D; j++) currCenters[j] /= args.counts[k];
                //std::cout << "wtf8" << std::endl;
                for (unsigned j = 0; j < args._D; j++) TCenter[j] = (float) (currCenters[j]);
                //std::cout << "wtf8" << std::endl;
            }
            diff += index->getDist()->compare(args.centers + k * args._D, TCenter, args._D);
        }
        return diff;
    }

    // 返回所有结点与当前所选簇中心的距离之和
    inline float ComponentInitSPTAG_BKT::KmeansAssign(std::vector<unsigned> &indices,
                                                      const unsigned first, const unsigned last,
                                                      Index::KmeansArgs<float> &args,
                                                      const bool updateCenters, float lambda) {
//        const float MaxDist = (std::numeric_limits<float>::max)();
//        float currDist = 0;
//        unsigned subsize = (last - first - 1) / args._T + 1;
//
//        //并行已删除
//        for (int tid = 0; tid < args._T; tid++) {
//            unsigned istart = first + tid * subsize;
//            unsigned iend = std::min(first + (tid + 1) * subsize, last);
//            unsigned *inewCounts = args.newCounts + tid * args._K;
//            float *inewCenters = args.newCenters + tid * args._K * args._D;
//            unsigned *iclusterIdx = args.clusterIdx + tid * args._K;
//            float *iclusterDist = args.clusterDist + tid * args._K;
//            float idist = 0;
//            for (unsigned i = istart; i < iend; i++) {
//                int clusterid = 0;
//                float smallestDist = MaxDist;
//                // 寻找最小距离簇中心
//                for (int k = 0; k < args._DK; k++) {
//                    float dist = index->getDist()->compare(index->getBaseData() + index->getBaseDim() * indices[i],
//                                                           args.centers + k * args._D,
//                                                           args._D) + lambda * args.counts[k];
//                    if (dist > -MaxDist && dist < smallestDist) {
//                        clusterid = k;
//                        smallestDist = dist;
//                    }
//                }
//                // 标记当前结点所属簇
//                args.label[i] = clusterid;
//                inewCounts[clusterid]++;
//                idist += smallestDist;
//                if (updateCenters) {
//                    // 待分类结点
//                    const float *v = index->getBaseData() + index->getBaseDim() * indices[i];
//                    // 结点所属簇中心
//                    float *center = inewCenters + clusterid * args._D;
//                    for (unsigned j = 0; j < args._D; j++) center[j] += v[j];
//                    if (smallestDist > iclusterDist[clusterid]) {
//                        iclusterDist[clusterid] = smallestDist;
//                        iclusterIdx[clusterid] = indices[i];
//                    }
//                } else {
//                    if (smallestDist <= iclusterDist[clusterid]) {
//                        iclusterDist[clusterid] = smallestDist;
//                        iclusterIdx[clusterid] = indices[i];
//                    }
//                }
//            }
//            currDist += idist;
//        }

//        std::cout << "4" << std::endl;
//        for(int i = 0; i < args._DK; i ++) {
//            std::cout << args.clusterIdx[i] << " ";
//        }
//        std::cout << std::endl;

        // 基本没用
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
        //return currDist;
        return 0;
    }

    unsigned ComponentInitSPTAG_BKT::rand(unsigned high, unsigned low) {
        return low + (unsigned) (float(high - low) * (std::rand() / (RAND_MAX + 1.0)));
    }

    inline void
    ComponentInitSPTAG_BKT::InitCenters(std::vector<unsigned> &indices, const unsigned first, const unsigned last,
                                        Index::KmeansArgs<float> &args, int samples, int tryIters) {
        const float MaxDist = (std::numeric_limits<float>::max)();
        unsigned batchEnd = std::min(first + samples, last);
        float currDist, minClusterDist = MaxDist;
        for (int numKmeans = 0; numKmeans < tryIters; numKmeans++) {
            for (int k = 0; k < args._DK; k++) {
                unsigned randid = rand(last, first);
                std::memcpy(args.centers + k * args._D, index->getBaseData() + index->getBaseDim() * indices[randid],
                            sizeof(float) * args._D);
            }
            args.ClearCounts();
            args.ClearDists(MaxDist);
            currDist = KmeansAssign(indices, first, batchEnd, args, false, 0);
            if (currDist < minClusterDist) {
                minClusterDist = currDist;
                memcpy(args.newTCenters, args.centers, sizeof(float) * args._K * args._D);
                memcpy(args.counts, args.newCounts, sizeof(unsigned) * args._K);
            }
        }
    }


    // HCNNG
    void ComponentInitHCNNG::InitInner() {

        // -- Hierarchical clustering --

        SetConfigs();

        int max_mst_degree = 3;

        std::vector<std::vector<Index::Edge> > G(index->getBaseLen());
        std::vector<omp_lock_t> locks(index->getBaseLen());

        // 初始化数据结构
        for (int i = 0; i < index->getBaseLen(); i++) {
            omp_init_lock(&locks[i]);
            G[i].reserve(max_mst_degree * index->num_cl);
        }

        printf("creating clusters...\n");
#pragma omp parallel for
        for (int i = 0; i < index->num_cl; i++) {
            int *idx_points = new int[index->getBaseLen()];
            for (int j = 0; j < index->getBaseLen(); j++)
                idx_points[j] = j;
            create_clusters(idx_points, 0, index->getBaseLen() - 1, G, index->minsize_cl, locks, max_mst_degree);
            printf("end cluster %d\n", i);
            delete[] idx_points;
        }

        printf("sorting...\n");
        sort_edges(G);
        print_stats_graph(G);

        // G - > final_graph
        index->getFinalGraph().resize(index->getBaseLen());

        for (int i = 0; i < index->getBaseLen(); i++) {
            std::vector<Index::SimpleNeighbor> tmp;

            int degree = G[i].size();

            for (int j = 0; j < degree; j++) {
                tmp.emplace_back(G[i][j].v2, G[i][j].weight);  // 已排序
            }
            index->getFinalGraph()[i] = tmp;
        }

        // -- KD-tree --
        unsigned seed = 1998;

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
        index->minsize_cl = index->getParam().get<unsigned>("minsize_cl");
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
        //std::cout << "www" << edges.size() << std::endl;
        for (Index::Edge &e : edges) {
            //std::cout << "111 " << (disjset->find(e.v1) != disjset->find(e.v2)) << std::endl;
            //std::cout << "111 " << (MST[e.v1].size() < max_mst_degree) << std::endl;
            //std::cout << "111 " << (MST[e.v2].size() < max_mst_degree) << std::endl;
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
                //std::cout << "w1" << " " << mst.size() << std::endl;
                //std::cout << "w2" << " " << mst[2].size() <<std::endl;
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
            // 随机抽取两点进行分簇
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

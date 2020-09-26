//
// Created by MurphySL on 2020/9/14.
//

#include "weavess/component.h"

#define _CONTROL_NUM 100
#define not_in_set(_elto, _set) (_set.find(_elto)==_set.end())

namespace weavess {


    // NN-Descent
    void ComponentInitNNDescent::InitInner() {
        SetConfigs();

        // 添加随机点作为近邻
        init();

        NNDescent();

        index->getFinalGraph().resize(index->getBaseLen());

        for (unsigned i = 0; i < index->getBaseLen(); i++) {
            std::vector<std::vector<unsigned>> level_tmp;
            std::vector<unsigned> tmp;

            std::sort(index->graph_[i].pool.begin(), index->graph_[i].pool.end());

            for (unsigned j = 0; j < index->graph_[i].pool.size(); j++) {
                tmp.push_back(index->graph_[i].pool[j].id);
            }

            tmp.resize(index->K);
            level_tmp.push_back(tmp);
            level_tmp.resize(1);
            index->getFinalGraph()[i] = level_tmp;
            // 内存释放
            std::vector<Index::Neighbor>().swap(index->graph_[i].pool);
            std::vector<unsigned>().swap(index->graph_[i].nn_new);
            std::vector<unsigned>().swap(index->graph_[i].nn_old);
            std::vector<unsigned>().swap(index->graph_[i].rnn_new);
            std::vector<unsigned>().swap(index->graph_[i].rnn_new);
        }
        std::vector<Index::nhood>().swap(index->graph_);
    }

    void ComponentInitNNDescent::SetConfigs() {
        index->K = index->getParam().get<unsigned>("K");
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

#pragma omp parallel for
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

        std::mt19937 rng(rand());

        // 采样用于评估每次迭代效果，与算法无关
        std::vector<unsigned> control_points(_CONTROL_NUM);
        std::vector<std::vector<unsigned> > acc_eval_set(_CONTROL_NUM);
        GenRandom(rng, &control_points[0], control_points.size(), index->getBaseLen());
        generate_control_set(control_points, acc_eval_set, index->getBaseLen());

        for (unsigned it = 0; it < index->ITER; it++) {
            std::cout << "NN-Descent iter: " << it << std::endl;

            join();

            float total = 0;
            unsigned num = 0;
            for(auto const &nhood : index->graph_) {
                total += eval_delta(nhood.pool);
            }

            update();

            eval_recall(control_points, acc_eval_set);

            if(num != 0 && total / num  <= index->delta) {
                std::cout << "stop condition : delta" << std::endl;
                break;
            }
        }
    }

    void ComponentInitNNDescent::join() {
#pragma omp parallel for default(shared) schedule(dynamic, 100)
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
#pragma omp parallel for
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
#pragma omp parallel for
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
#pragma omp parallel for
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
#pragma omp parallel for
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

    void ComponentInitNNDescent::generate_control_set(std::vector<unsigned> &c, std::vector<std::vector<unsigned> > &v,
                                                      unsigned N) {
#pragma omp parallel for
        for (unsigned i = 0; i < c.size(); i++) {
            std::vector<Index::Neighbor> tmp;
            for (unsigned j = 0; j < N; j++) {
                float dist = index->getDist()->compare(index->getBaseData() + c[i] * index->getBaseDim(),
                                                       index->getBaseData() + j * index->getBaseDim(),
                                                       index->getBaseDim());
                tmp.push_back(Index::Neighbor(j, dist, true));
            }
            std::partial_sort(tmp.begin(), tmp.begin() + _CONTROL_NUM, tmp.end());
            for (unsigned j = 0; j < _CONTROL_NUM; j++) {
                v[i].push_back(tmp[j].id);
            }
        }
    }

    void ComponentInitNNDescent::eval_recall(std::vector<unsigned> &ctrl_points,
                                             std::vector<std::vector<unsigned> > &acc_eval_set) {
        float mean_acc = 0;
        for (unsigned i = 0; i < ctrl_points.size(); i++) {
            float acc = 0;
            auto &g = index->graph_[ctrl_points[i]].pool;
            auto &v = acc_eval_set[i];
            for (unsigned j = 0; j < g.size(); j++) {
                for (unsigned k = 0; k < v.size(); k++) {
                    if (g[j].id == v[k]) {
                        acc++;
                        break;
                    }
                }
            }
            mean_acc += acc / v.size();
        }
        std::cout << "recall : " << mean_acc / ctrl_points.size() << std::endl;
    }

    float ComponentInitNNDescent::eval_delta (std::vector<Index::Neighbor> const &pool) {
        unsigned c = 0;
        unsigned N = index->K;

        if (pool.size() < N) N = pool.size();
        for (unsigned i = 0; i < N; ++i) {
            if (pool[i].flag) ++c;
        }
        return float(c) / index->K;
    }


    // RAND
    void ComponentInitRandom::InitInner() {
        SetConfigs();

        index->graph_.resize(index->getBaseLen());
        std::mt19937 rng(rand());

#pragma omp parallel for
        for (unsigned i = 0; i < index->getBaseLen(); i++) {
            std::vector<unsigned> tmp(index->K);

            weavess::GenRandom(rng, tmp.data(), index->K, index->getBaseLen());

            for (unsigned j = 0; j < index->K; j++) {
                unsigned id = tmp[j];

                if (id == i)continue;
                float dist = index->getDist()->compare(index->getBaseData() + i * index->getBaseDim(),
                                                       index->getBaseData() + id * index->getBaseDim(),
                                                       (unsigned) index->getBaseDim());

                index->graph_[i].pool.emplace_back(id, dist, true);
            }
            std::make_heap(index->graph_[i].pool.begin(), index->graph_[i].pool.end());
            index->graph_[i].pool.reserve(index->K);
        }

        index->getFinalGraph().resize(index->getBaseLen());

        for (unsigned i = 0; i < index->getBaseLen(); i++) {
            std::vector<std::vector<unsigned>> level_tmp;
            std::vector<unsigned> tmp;

            std::sort(index->graph_[i].pool.begin(), index->graph_[i].pool.end());

            for (unsigned j = 0; j < index->graph_[i].pool.size(); j++) {
                tmp.push_back(index->graph_[i].pool[j].id);
            }

            //tmp.resize(index->K);
            level_tmp.push_back(tmp);
            level_tmp.resize(1);
            index->getFinalGraph()[i] = level_tmp;
            // 内存释放
            std::vector<Index::Neighbor>().swap(index->graph_[i].pool);
            std::vector<unsigned>().swap(index->graph_[i].nn_new);
            std::vector<unsigned>().swap(index->graph_[i].nn_old);
            std::vector<unsigned>().swap(index->graph_[i].rnn_new);
            std::vector<unsigned>().swap(index->graph_[i].rnn_new);
        }

        std::vector<Index::nhood>().swap(index->graph_);
    }

    void ComponentInitRandom::SetConfigs() {
        index->K = index->getParam().get<unsigned>("K");
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
        std::set<unsigned> result;
        for (unsigned i = 0; i < index->getBaseLen(); i++) {
            std::vector<std::vector<unsigned>> level_tmp;
            std::vector<unsigned> tmp;
            typename Index::CandidateHeap::reverse_iterator it = index->knn_graph[i].rbegin();
            for (; it != index->knn_graph[i].rend(); it++) {
                tmp.push_back(it->row_id);
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
                    result.insert(id);
                }
                tmp.clear();
                std::set<unsigned>::iterator it;
                for (it = result.begin(); it != result.end(); it++) {
                    tmp.push_back(*it);
                }
                //std::copy(result.begin(),result.end(),tmp.begin());
            }
            tmp.reserve(K);
            level_tmp.push_back(tmp);
            level_tmp.resize(1);
            index->getFinalGraph()[i] = level_tmp;
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

    // HCNNG
    void ComponentInitHCNNG::InitInner() {

        SetConfigs();

        auto minsize_cl = index->getParam().get<unsigned>("S");
        auto num_cl = index->getParam().get<unsigned>("N");
        int max_mst_degree = 3;

        std::vector<std::vector<Index::Edge> > G(index->getBaseLen());
        std::vector<omp_lock_t> locks(index->getBaseLen());

        // MST 图
        for (int i = 0; i < index->getBaseLen(); i++) {
            omp_init_lock(&locks[i]);
            G[i].reserve(max_mst_degree * num_cl);
        }

        printf("creating clusters...\n");
#pragma omp parallel for
        for (int i = 0; i < num_cl; i++) {
            int *idx_points = new int[index->getBaseLen()];
            for (int j = 0; j < index->getBaseLen(); j++)
                idx_points[j] = j;
            create_clusters(idx_points, 0, index->getBaseLen() - 1, G, minsize_cl, locks, max_mst_degree);
            printf("end cluster %d\n", i);
            delete[] idx_points;
        }

        printf("sorting...\n");
        sort_edges(G);
        print_stats_graph(G);

        for (size_t i = 0; i < index->getBaseLen(); i++) {
            auto size = G[i].size();
            long long min_diff = 1e6, min_diff_dim = -1;
            for (size_t j = 0; j < index->getBaseDim(); j++) {
                int lnum = 0, rnum = 0;
                for (size_t k = 0; k < size; k++) {
                    if (*(index->getBaseData() + G[i][k].v2 + j) < *(index->getBaseData() + i + j)) {
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
                if (*(index->getBaseData() + G[i][k].v2 + min_diff_dim) < *(index->getBaseData() + i + min_diff_dim)) {
                    index->Tn[i].left.push_back(G[i][k].v2);
                } else {
                    index->Tn[i].right.push_back(G[i][k].v2);
                }
            }
        }
    }

    void ComponentInitHCNNG::SetConfigs() {
        index->S_hcnng = index->getParam().get<float>("S");
        index->N = index->getParam().get<unsigned>("N");
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
                }

        }
        tie(mst, cost) = kruskal(full, N, max_mst_degree);
        return mst;
    }

    bool check_in_neighbors(int u, std::vector<Index::Edge> &neigh) {
        for (int i = 0; i < neigh.size(); i++)
            if (neigh[i].v2 == u)
                return true;
        return false;
    }

    void ComponentInitHCNNG::create_clusters(int *idx_points, int left, int right,
                                             std::vector<std::vector<Index::Edge> > &graph, int minsize_cl,
                                             std::vector<omp_lock_t> &locks, int max_mst_degree) {
        int num_points = right - left + 1;
        std::cout << "points num : " << num_points << std::endl;
        if (num_points < minsize_cl) {
            std::vector<std::vector<Index::Edge> > mst = create_exact_mst(idx_points, left, right, max_mst_degree);
            for (int i = 0; i < num_points; i++) {
                for (int j = 0; j < mst[i].size(); j++) {
                    omp_set_lock(&locks[idx_points[left + i]]);
                    if (!check_in_neighbors(idx_points[left + mst[i][j].v2], graph[idx_points[left + i]]))
                        graph[idx_points[left + i]].push_back(
                                Index::Edge(idx_points[left + i], idx_points[left + mst[i][j].v2], mst[i][j].weight));
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
                float dist = index->getDist()->compare(index->getBaseData() + idx_points[x] * index->getBaseDim(),
                                                       index->getBaseData() +
                                                       idx_points[left + i] * index->getBaseDim(),
                                                       (unsigned) index->getBaseDim());
                dx[i] = std::make_pair(dist, idx_points[left + i]);

                dist = index->getDist()->compare(index->getBaseData() + idx_points[y] * index->getBaseDim(),
                                                 index->getBaseData() + idx_points[left + i] * index->getBaseDim(),
                                                 (unsigned) index->getBaseDim());
                dy[i] = std::make_pair(dist, idx_points[left + i]);
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

}
//
// Created by MurphySL on 2020/10/23.
//

#include "weavess/component.h"

namespace weavess {

    /**
     * 全局近似质心入口点
     */
    void ComponentPreEntryCentroid::PreEntryInner() {
        index->getSeeds().resize(1);

        auto *center = new float[index->getBaseDim()];
        for (unsigned j = 0; j < index->getBaseDim(); j++) center[j] = 0;
        for (unsigned i = 0; i < index->getBaseLen(); i++) {
            for (unsigned j = 0; j < index->getBaseDim(); j++) {
                center[j] += index->getBaseData()[i * index->getBaseDim() + j];
            }
        }

        for (unsigned j = 0; j < index->getBaseDim(); j++) {
            center[j] /= index->getBaseLen();
        }

        std::vector<Index::Neighbor> tmp, pool;
        index->getSeeds()[0] = rand() % index->getBaseLen();  // random initialize navigating point
        get_neighbors(center, index->getSeeds()[0], tmp, pool);
        index->getSeeds()[0] = tmp[0].id;
    }

    void ComponentPreEntryCentroid::get_neighbors(const float *query, const unsigned entry, std::vector<Index::Neighbor> &retset, std::vector<Index::Neighbor> &fullset) {
        unsigned L = index->L_refine;
        retset.resize(L + 1);
        std::vector<unsigned> init_ids(L);
        // initializer_->Search(query, nullptr, L, parameter, init_ids.data());
        boost::dynamic_bitset<> flags{index->getBaseLen(), 0};
        L = 0;
        for (unsigned i = 0; i < init_ids.size() && i < index->getFinalGraph()[entry].size(); i++) {
            init_ids[i] = index->getFinalGraph()[entry][i].id;
            flags[init_ids[i]] = true;
            L++;
        }
        while (L < init_ids.size()) {
            unsigned id = rand() % index->getBaseLen();
            if (flags[id]) continue;
            init_ids[L] = id;
            L++;
            flags[id] = true;
        }
        L = 0;
        for (unsigned i = 0; i < init_ids.size(); i++) {
            unsigned id = init_ids[i];
            if (id >= index->getBaseLen()) continue;
            float dist = index->getDist()->compare(index->getBaseData() + index->getBaseDim() * (size_t) id, query,
                                                   (unsigned) index->getBaseDim());
            retset[i] = Index::Neighbor(id, dist, true);
            //retset[i] = new Index::Node(id, dist, true, 0);
            // flags[id] = 1;
            L++;
        }
        std::sort(retset.begin(), retset.begin() + L);

        int k = 0;
        while (k < (int) L) {
            int nk = L;

            if (retset[k].flag) {
                retset[k].flag = false;
                unsigned n = retset[k].id;
                for (unsigned m = 0; m < index->getFinalGraph()[n].size(); ++m) {
                    unsigned id = index->getFinalGraph()[n][m].id;
                    if (flags[id]) continue;
                    flags[id] = true;

                    float dist = index->getDist()->compare(query,
                                                           index->getBaseData() + index->getBaseDim() * (size_t) id,
                                                           (unsigned) index->getBaseDim());
                    Index::Neighbor nn(id, dist, true);
                    fullset.push_back(nn);
                    if (dist >= retset[L - 1].distance) continue;
                    int r = Index::InsertIntoPool(retset.data(), L, nn);

                    if (L + 1 < retset.size()) ++L;
                    if (r < nk) nk = r;
                }
            }
            if (nk <= k)
                k = nk;
            else
                ++k;
        }
    }

    /**
    * 全局质心入口点
    * @param query 查询点
    * @param pool 候选池
    */
    void ComponentSearchEntryCentroid::SearchEntryInner(unsigned int query, std::vector<Index::Neighbor> &pool) {
        const auto L = index->getParam().get<unsigned>("L_search");
        pool.reserve(L + 1);
        std::vector<unsigned> init_ids(L);
        boost::dynamic_bitset<> flags{index->getBaseLen(), 0};
        // std::mt19937 rng(rand());
        // GenRandom(rng, init_ids.data(), L, (unsigned) index_->n_);

        if(index->getSeeds().size() != 1) {
            std::cerr << "pre entry wrong type" << std::endl;
            if(index->getSeeds().size() < 1) exit(-1);
        }

        unsigned tmp_l = 0;
        for (; tmp_l < L && tmp_l < index->getLoadGraph()[index->getSeeds()[0]].size(); tmp_l++) {
            init_ids[tmp_l] = index->getLoadGraph()[index->getSeeds()[0]][tmp_l];
            flags[init_ids[tmp_l]] = true;
        }

        while (tmp_l < L) {
            unsigned id = rand() % index->getBaseLen();
            if (flags[id]) continue;
            flags[id] = true;
            init_ids[tmp_l] = id;
            tmp_l++;
        }

        for (unsigned i = 0; i < init_ids.size(); i++) {
            unsigned id = init_ids[i];
            float dist =
                    index->getDist()->compare(index->getBaseData() + index->getBaseDim() * id,
                                              index->getQueryData() + index->getQueryDim() * query,
                                              (unsigned) index->getBaseDim());
            index->addDistCount();
            pool[i] = Index::Neighbor(id, dist, true);
            // flags[id] = true;
        }

        std::sort(pool.begin(), pool.begin() + L);
    }

    /**
     * 静态随机入口点
     */
    void ComponentPreEntryRandom::PreEntryInner() {
        std::mt19937 rng(rand());
        index->getSeeds().resize(10);
        GenRandom(rng, index->getSeeds().data(), 10, (unsigned) index->getBaseLen());
    }

    /**
     * 随机入口点
     * @param query 查询点
     * @param pool 候选池
     */
    void ComponentSearchEntryRandom::SearchEntryInner(unsigned query, std::vector<Index::Neighbor> &pool) {
        const auto L = index->getParam().get<unsigned>("L_search");

        pool.resize(L + 1);

        std::vector<unsigned> init_ids(L);
        for(int i = 0; i < index->getSeeds().size(); i ++){
            init_ids.push_back(index->getSeeds()[i]);
        }
        std::mt19937 rng(rand());

        std::vector<unsigned> tmp(L - index->getSeeds().size());
        GenRandom(rng, tmp.data(), L - index->getSeeds().size(), (unsigned) index->getBaseLen());

        for(int i = 0; i < tmp.size(); i ++) {
            init_ids.push_back(tmp[i]);
        }

        for (unsigned i = 0; i < L; i++) {
            unsigned id = init_ids[i];
            float dist = index->getDist()->compare(index->getQueryData() + query * index->getQueryDim(),
                                                   index->getBaseData() + id * index->getBaseDim(),
                                                   (unsigned) index->getBaseDim());
            index->addDistCount();
            pool[i] = Index::Neighbor(id, dist, true);
        }

        std::sort(pool.begin(), pool.begin() + L);
    }




    /**
     * 子图近似质心入口点
     */
    void ComponentPreEntrySubCentroid::PreEntryInner() {
        unsigned root = 0;
        boost::dynamic_bitset<> flags{index->getBaseLen(), 0};

        unsigned unlinked_cnt = 0;

        while (unlinked_cnt < index->getBaseLen()) {
            DFS(flags, root, unlinked_cnt);
            if (unlinked_cnt >= index->getBaseLen()) break;
            findRoot(flags, root);
        }
    }

    void ComponentPreEntrySubCentroid::DFS(boost::dynamic_bitset<> &flag, unsigned root, unsigned &cnt) {
        auto *center = new float[index->getBaseDim()];
        for (unsigned j = 0; j < index->getBaseDim(); j++) center[j] = 0;

        unsigned tmp = root;
        std::stack<unsigned> s;
        s.push(root);
        if (!flag[root]) cnt++;
        flag[root] = true;
        for (unsigned j = 0; j < index->getBaseDim(); j++) {
            center[j] += index->getBaseData()[root * index->getBaseDim() + j];
        }

        while (!s.empty()) {
            unsigned next = index->getBaseLen() + 1;
            for (unsigned i = 0; i < index->getFinalGraph()[tmp].size(); i++) {
                if (!flag[index->getFinalGraph()[tmp][i].id]) {
                    next = index->getFinalGraph()[tmp][i].id;
                    break;
                }
            }
            if (next == (index->getBaseLen() + 1)) {
                s.pop();
                if (s.empty()) break;
                tmp = s.top();
                continue;
            }
            tmp = next;
            flag[tmp] = true;
            s.push(tmp);
            cnt++;
            for (unsigned j = 0; j < index->getBaseDim(); j++) {
                center[j] += index->getBaseData()[root * index->getBaseDim() + j];
            }
        }

        for (unsigned j = 0; j < index->getBaseDim(); j++) {
            center[j] /= cnt;
        }

        std::vector<Index::Neighbor> tmp_pool, pool;
        index->getSeeds().push_back(rand() % index->getBaseLen());
        index->getSeeds()[0] = rand() % index->getBaseLen();  // random initialize navigating point
        get_neighbors(center, index->getSeeds()[0], tmp_pool, pool);
        index->getSeeds()[0] = tmp_pool[0].id;
    }

    void ComponentPreEntrySubCentroid::findRoot(boost::dynamic_bitset<> &flag, unsigned &root) {
        unsigned id = index->getBaseLen();
        for (unsigned i = 0; i < index->getBaseLen(); i++) {
            if (!flag[i]) {
                id = i;
                break;
            }
        }
        root = id;
    }

    void ComponentPreEntrySubCentroid::get_neighbors(const float *query, const unsigned entry,
                                                          std::vector<Index::Neighbor> &retset,
                                                          std::vector<Index::Neighbor> &fullset) {
        unsigned L = index->L_refine;
        retset.resize(L + 1);
        std::vector<unsigned> init_ids(L);
        // initializer_->Search(query, nullptr, L, parameter, init_ids.data());
        boost::dynamic_bitset<> flags{index->getBaseLen(), 0};
        L = 0;
        for (unsigned i = 0; i < init_ids.size() && i < index->getFinalGraph()[entry].size(); i++) {
            init_ids[i] = index->getFinalGraph()[entry][i].id;
            flags[init_ids[i]] = true;
            L++;
        }
        while (L < init_ids.size()) {
            unsigned id = rand() % index->getBaseLen();
            if (flags[id]) continue;
            init_ids[L] = id;
            L++;
            flags[id] = true;
        }
        L = 0;
        for (unsigned i = 0; i < init_ids.size(); i++) {
            unsigned id = init_ids[i];
            if (id >= index->getBaseLen()) continue;
            float dist = index->getDist()->compare(index->getBaseData() + index->getBaseDim() * (size_t) id, query,
                                                   (unsigned) index->getBaseDim());
            retset[i] = Index::Neighbor(id, dist, true);
            //retset[i] = new Index::Node(id, dist, true, 0);
            // flags[id] = 1;
            L++;
        }
        std::sort(retset.begin(), retset.begin() + L);

        int k = 0;
        while (k < (int) L) {
            int nk = L;

            if (retset[k].flag) {
                retset[k].flag = false;
                unsigned n = retset[k].id;
                for (unsigned m = 0; m < index->getFinalGraph()[n].size(); ++m) {
                    unsigned id = index->getFinalGraph()[n][m].id;
                    if (flags[id]) continue;
                    flags[id] = true;

                    float dist = index->getDist()->compare(query,
                                                           index->getBaseData() + index->getBaseDim() * (size_t) id,
                                                           (unsigned) index->getBaseDim());
                    Index::Neighbor nn(id, dist, true);
                    fullset.push_back(nn);
                    if (dist >= retset[L - 1].distance) continue;
                    int r = Index::InsertIntoPool(retset.data(), L, nn);

                    if (L + 1 < retset.size()) ++L;
                    if (r < nk) nk = r;
                }
            }
            if (nk <= k)
                k = nk;
            else
                ++k;
        }
    }

    /**
     * 全局质心入口点
     * @param query 查询点
     * @param pool 候选池
     */
    void ComponentSearchEntrySubCentroid::SearchEntryInner(unsigned int query, std::vector<Index::Neighbor> &pool) {
        const auto L = index->getParam().get<unsigned>("L_search");
        pool.reserve(L + 1);
        std::vector<unsigned> init_ids(L);
        boost::dynamic_bitset<> flags{index->getBaseLen(), 0};
        // std::mt19937 rng(rand());
        // GenRandom(rng, init_ids.data(), L, (unsigned) index_->n_);

        assert(!index->getSeeds().empty());

        unsigned tmp_l = 0;
        for(int i = 0; i < index->getSeeds().size() && tmp_l < L; i ++, tmp_l ++) {
            init_ids[tmp_l] = index->getSeeds()[i];
            flags[init_ids[tmp_l]] = true;
        }

        while (tmp_l < L) {
            unsigned id = rand() % index->getBaseLen();
            if (flags[id]) continue;
            flags[id] = true;
            init_ids[tmp_l] = id;
            tmp_l++;
        }

        for (unsigned i = 0; i < init_ids.size(); i++) {
            unsigned id = init_ids[i];
            float dist =
                    index->getDist()->compare(index->getBaseData() + index->getBaseDim() * id,
                                              index->getQueryData() + index->getQueryDim() * query,
                                              (unsigned) index->getBaseDim());
            index->addDistCount();
            pool[i] = Index::Neighbor(id, dist, true);
            // flags[id] = true;
        }

        std::sort(pool.begin(), pool.begin() + L);
    }




    /**
     * 构建 KDT 用于寻找入口点
     *
     * 需要指定构建新树阈值 num
     */
    void ComponentPreEntryKDTree::PreEntryInner() {
        SetConfigs();

        unsigned seed = 1998;

        const auto TreeNum = index->getParam().get<unsigned>("nTrees");
        const auto TreeNumBuild = index->getParam().get<unsigned>("nTrees");
        const auto K = index->getParam().get<unsigned>("num");

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
    }

    void ComponentPreEntryKDTree::SetConfigs() {
        index->nTrees = index->getParam().get<unsigned>("nTrees");
        index->mLevel = index->getParam().get<unsigned>("mLevel");
    }

    void ComponentPreEntryKDTree::meanSplit(std::mt19937 &rng, unsigned *indices, unsigned count, unsigned &index1,
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
    ComponentPreEntryKDTree::planeSplit(unsigned *indices, unsigned count, unsigned cutdim, float cutval,
                                        unsigned &lim1,
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

    int ComponentPreEntryKDTree::selectDivision(std::mt19937 &rng, float *v) {
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

    void
    ComponentPreEntryKDTree::DFSbuild(Index::EFANNA::Node *node, std::mt19937 &rng, unsigned *indices, unsigned count,
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

    void ComponentPreEntryKDTree::getMergeLevelNodeList(Index::EFANNA::Node *node, size_t treeid, unsigned deepth) {
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

    /**
    * KDT获取入口点
     *
    * @param query 查询点
    * @param pool 候选池
    */
    void ComponentSearchEntryKDT::SearchEntryInner(unsigned int query, std::vector<Index::Neighbor> &pool) {
        unsigned TreeNum = index->nTrees;
        const auto L = index->getParam().get<unsigned>("L_search");

        pool.clear();
        pool.resize(L+1);

        std::vector<char> flags(index->getBaseLen());
        std::memset(flags.data(), 0, index->getBaseLen() * sizeof(char));
        flags[query] = true;

        std::vector<unsigned> init_ids(L);

        unsigned lsize = L / (TreeNum * index->TNS) + 1;
        std::vector<std::vector<Index::Node*> > Vnl;
        Vnl.resize(TreeNum);
        for(unsigned i =0; i < TreeNum; i ++)
            getSearchNodeList(index->tree_roots_[i], index->getQueryData() + index->getQueryDim() * query, lsize, Vnl[i]);

        unsigned p = 0;
        for(unsigned ni = 0; ni < lsize; ni ++) {
            for(unsigned i = 0; i < Vnl.size(); i ++) {
                Index::Node *leafn = Vnl[i][ni];
                for(size_t j = leafn->StartIdx; j < leafn->EndIdx && p < L; j ++) {
                    size_t nn = index->LeafLists[i][j];
                    if(flags[nn])continue;
                    flags[nn] = 1;
                    init_ids[p++]=(nn);
                }
                if(p >= L) break;
            }
            if(p >= L) break;
        }

        while(p < L){
            unsigned int nn = rand() % index->getBaseLen();
            if(flags[nn])continue;
            flags[nn] = 1;
            init_ids[p++]=(nn);
        }

        memset(flags.data(), 0, index->getBaseLen() * sizeof(char));
        for(unsigned i=0; i<L; i++){
            unsigned id = init_ids[i];
            float dist = index->getDist()->compare(index->getBaseData() + index->getBaseDim() * id,
                                                   index->getQueryData() + index->getQueryDim() * query,
                                                   index->getBaseDim());
            index->addDistCount();
            pool[i]=Index::Neighbor(id, dist, true);
        }

        std::sort(pool.begin(), pool.begin()+L);
    }

    void ComponentSearchEntryKDT::getSearchNodeList(Index::Node* node, const float *q, unsigned int lsize, std::vector<Index::Node*>& vn){
        if(vn.size() >= lsize)
            return;

        if(node->Lchild != nullptr && node->Rchild != nullptr){
            if(q[node->DivDim] < node->DivVal){
                getSearchNodeList(node->Lchild, q, lsize,  vn );
                getSearchNodeList(node->Rchild, q, lsize, vn);
            }else{
                getSearchNodeList(node->Rchild, q, lsize, vn);
                getSearchNodeList(node->Lchild, q, lsize, vn);
            }
        }else
            vn.push_back(node);
    }




    /**
     * 构建 VPT 用于寻找入口点
     */
    void ComponentPreEntryVPTree::PreEntryInner() {
        std::vector<unsigned> obj(index->getBaseLen());

        for (int i = 0; i < index->getBaseLen(); i++) {
            obj[i] = i;
        }
        MakeVPTree(obj);
    }

    // Construct the tree from given objects set
    void ComponentPreEntryVPTree::MakeVPTree(const std::vector<unsigned> &objects) {
        index->vp_tree.m_root.reset();

#pragma omp parallel
        {
#pragma omp single
#pragma omp task
            index->vp_tree.m_root = MakeVPTree(objects, index->vp_tree.m_root);
        }
    }

    void ComponentPreEntryVPTree::InsertSplitLeafRoot(Index::VPNodePtr &root, const unsigned &new_value) {
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

        for (size_t c_pos = 0; c_pos < root->get_objects_count(); ++c_pos)
            Insert(root->m_objects_list[c_pos], root);

        root->m_objects_list.clear();

        Insert(new_value, root);

        //RedistributeAmongLeafNodes(root, new_value);

        //m_root->m_mu_list[0] = 0;
        //m_root->set_value(new_value); // Set Vantage Point
    }

    // Recursively collect data from subtree, and push them into S
    void ComponentPreEntryVPTree::CollectObjects(const Index::VPNodePtr &node, std::vector<unsigned> &S) {
        if (node->get_leaf_node())
            S.insert(S.end(), node->m_objects_list.begin(), node->m_objects_list.end());
        else {
            for (size_t c_pos = 0; c_pos < node->get_branches_count(); ++c_pos)
                CollectObjects(node->m_child_list[c_pos], S);
        }
    }

    // (max{d(v, sj) | sj c SS1} + min{d(v, sj) | sj c SS2}) / 2
    const float ComponentPreEntryVPTree::MedianSumm(const std::vector<unsigned> &SS1, const std::vector<unsigned> &SS2,
                                                    const unsigned &v) const {
        float c_max_distance = 0;
        float c_min_distance = 0;
        float c_current_distance = 0;

        if (!SS1.empty()) {
            // max{d(v, sj) | sj c SSj}
            typename std::vector<unsigned>::const_iterator it_obj = SS1.begin();
            c_max_distance = index->getDist()->compare(index->getBaseData() + index->getBaseDim() * (*it_obj),
                                                       index->getBaseData() + index->getBaseDim() * v,
                                                       index->getBaseDim());
            ++it_obj;
            c_current_distance = c_max_distance;
            while (it_obj != SS1.end()) {
                c_current_distance = index->getDist()->compare(index->getBaseData() + index->getBaseDim() * (*it_obj),
                                                               index->getBaseData() + index->getBaseDim() * v,
                                                               index->getBaseDim());
                if (c_current_distance > c_max_distance)
                    c_max_distance = c_current_distance;

                ++it_obj;
            }
        }

        if (!SS2.empty()) {
            // min{d(v, sj) | sj c SSj}
            typename std::vector<unsigned>::const_iterator it_obj = SS2.begin();
            c_min_distance = index->getDist()->compare(index->getBaseData() + index->getBaseDim() * (*it_obj),
                                                       index->getBaseData() + index->getBaseDim() * v,
                                                       index->getBaseDim());
            ++it_obj;
            c_current_distance = c_min_distance;
            while (it_obj != SS2.end()) {
                c_current_distance = index->getDist()->compare(index->getBaseData() + index->getBaseDim() * (*it_obj),
                                                               index->getBaseData() + index->getBaseDim() * v,
                                                               index->getBaseDim());
                if (c_current_distance < c_min_distance)
                    c_min_distance = c_current_distance;

                ++it_obj;
            }
        }

        return (c_max_distance + c_min_distance) / static_cast<float>(2);
    }

    // Calc the median value for object set
    float ComponentPreEntryVPTree::Median(const unsigned &value, const std::vector<unsigned>::const_iterator it_begin,
                                          const std::vector<unsigned>::const_iterator it_end) {
        std::vector<unsigned>::const_iterator it_obj = it_begin;
        float current_distance = 0;
        size_t count = 0;
        while (it_obj != it_end) {
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
    void ComponentPreEntryVPTree::RedistributeAmongLeafNodes(const Index::VPNodePtr &parent_node,
                                                             const unsigned &new_value) {
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
        for (size_t c_pos = 0; c_pos < F; ++c_pos) {
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
        for (size_t c_pos = 0; c_pos < F - 1; ++c_pos) {
            const std::vector<unsigned> &SS1 = parent_node->m_child_list[c_pos]->m_objects_list;
            const std::vector<unsigned> &SS2 = parent_node->m_child_list[c_pos + 1]->m_objects_list;

            parent_node->m_mu_list[c_pos] = MedianSumm(SS1, SS2, parent_node->get_value());
        }

    }

    // If L(eaf) node has a P(arent) node and P has room for one more child,
    // split the leaf node L
    void ComponentPreEntryVPTree::SplitLeafNode(const Index::VPNodePtr &parent_node, const size_t child_id,
                                                const unsigned &new_value) {
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
        for (size_t c_pos = 0; c_pos < S.size(); ++c_pos) {
            if (c_pos < c_half_count)
                ss1_node->AddObject(S[c_pos]);
            else
                ss2_node->AddObject(S[c_pos]);
        }

        // insertion/shift process
        for (size_t c_pos = F - 2; c_pos >= k; --c_pos) {
            parent_node->m_mu_list[c_pos + 1] = parent_node->m_mu_list[c_pos];
            if (!c_pos) // !!! hack :(
                break;
        }

        const std::vector<unsigned> &SS1 = ss1_node->m_objects_list;
        const std::vector<unsigned> &SS2 = ss2_node->m_objects_list;
        parent_node->m_mu_list[k] = MedianSumm(SS1, SS2, parent_node->get_value());

        // !! --c_pos
        for (size_t c_pos = F - 1; c_pos >= k + 1; --c_pos)
            parent_node->m_child_list[c_pos + 1] = parent_node->m_child_list[c_pos];

        parent_node->m_child_list[k] = ss1_node;
        parent_node->m_child_list[k + 1] = ss2_node;
    }

    void ComponentPreEntryVPTree::Remove(const unsigned &query_value, const Index::VPNodePtr &node) {
        if (node->get_leaf_node())
            node->DeleteObject(query_value);
        else {
            for (size_t c_pos = 0; c_pos < node->get_branches_count(); ++c_pos)
                Remove(query_value, node->m_child_list[c_pos]);
        }
    }

    // 3.a. Redistribute, among the sibling subtrees
    void ComponentPreEntryVPTree::RedistributeAmongNonLeafNodes(const Index::VPNodePtr &parent_node, const size_t k_id,
                                                                const size_t k1_id, const unsigned &new_value) {
        //std::cout << "	redistribute among nodes(subtrees)" << std::endl;
        assert(k_id != k1_id);

        size_t num_k = index->vp_tree.get_object_count(parent_node->m_child_list[k_id]);
        size_t num_k1 = index->vp_tree.get_object_count(parent_node->m_child_list[k1_id]);

        size_t average = (num_k + num_k1) / 2;

        if (num_k > num_k1) {
            // Create Set of objects from leaf nodes K-th subtree
            std::vector<unsigned> S; // Set of leaf objects + new one;
            CollectObjects(parent_node->m_child_list[k_id], S);
            //S.push_back(new_value);

            unsigned m_main_val = parent_node->get_value();
            Index *tmp_index = index;

            //Index::ValueSorterType val_sorter(parent_node->get_value(), m_get_distance);
            std::sort(S.begin(), S.end(), [m_main_val, tmp_index](unsigned val1, unsigned val2) {
                float dist1 = tmp_index->getDist()->compare(
                        tmp_index->getBaseData() + tmp_index->getBaseDim() * m_main_val,
                        tmp_index->getBaseData() + tmp_index->getBaseDim() * val1,
                        tmp_index->getBaseDim());
                float dist2 = tmp_index->getDist()->compare(
                        tmp_index->getBaseData() + tmp_index->getBaseDim() * m_main_val,
                        tmp_index->getBaseData() + tmp_index->getBaseDim() * val2,
                        tmp_index->getBaseDim());
                return dist1 < dist2;
            });

            size_t w = num_k - average;

            std::vector<unsigned> SS1(S.begin(), S.begin() + num_k - w);
            std::vector<unsigned> SS2(S.begin() + num_k - w, S.end());

            SS1.push_back(new_value);

            typename std::vector<unsigned>::const_iterator it_obj = SS2.begin();
            for (; it_obj != SS2.end(); ++it_obj)
                Remove(*it_obj, parent_node->m_child_list[k_id]);

            parent_node->m_mu_list[k_id] = MedianSumm(SS1, SS2, parent_node->get_value());

            for (it_obj = SS2.begin(); it_obj != SS2.end(); ++it_obj)
                Insert(*it_obj, parent_node->m_child_list[k1_id]);

        } else {
            // Create Set of objects from leaf nodes K-th subtree
            std::vector<unsigned> S; // Set of leaf objects + new one;
            CollectObjects(parent_node->m_child_list[k1_id], S);
            //S.push_back(new_value);
            unsigned m_main_val = parent_node->get_value();
            Index *tmp_index = index;
            //ValueSorterType val_sorter(parent_node->get_value(), m_get_distance);
            std::sort(S.begin(), S.end(), [m_main_val, tmp_index](unsigned val1, unsigned val2) {
                float dist1 = tmp_index->getDist()->compare(
                        tmp_index->getBaseData() + tmp_index->getBaseDim() * m_main_val,
                        tmp_index->getBaseData() + tmp_index->getBaseDim() * val1,
                        tmp_index->getBaseDim());
                float dist2 = tmp_index->getDist()->compare(
                        tmp_index->getBaseData() + tmp_index->getBaseDim() * m_main_val,
                        tmp_index->getBaseData() + tmp_index->getBaseDim() * val2,
                        tmp_index->getBaseDim());
                return dist1 < dist2;
            });

            size_t w = num_k1 - average;

            std::vector<unsigned> SS1(S.begin(), S.begin() + w);
            std::vector<unsigned> SS2(S.begin() + w, S.end());
            SS2.push_back(new_value);

            typename std::vector<unsigned>::const_iterator it_obj = SS1.begin();
            for (; it_obj != SS1.end(); ++it_obj)
                Remove(*it_obj, parent_node->m_child_list[k1_id]);

            parent_node->m_mu_list[k_id] = MedianSumm(SS1, SS2, parent_node->get_value());

            for (it_obj = SS1.begin(); it_obj != SS1.end(); ++it_obj)
                Insert(*it_obj, parent_node->m_child_list[k_id]);
        }

        num_k = index->vp_tree.get_object_count(parent_node->m_child_list[k_id]);
        num_k1 = index->vp_tree.get_object_count(parent_node->m_child_list[k1_id]);

        num_k = 0;
    }

    void ComponentPreEntryVPTree::InsertSplitRoot(Index::VPNodePtr &root, const unsigned &new_value) {
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
    const unsigned &ComponentPreEntryVPTree::SelectVP(const std::vector<unsigned> &objects) {
        assert(!objects.empty());

        return *objects.begin();
    }

    // Construct the tree from given objects set
    Index::VPNodePtr
    ComponentPreEntryVPTree::MakeVPTree(const std::vector<unsigned> &objects, const Index::VPNodePtr &parent) {

        if (objects.empty())
            return Index::VPNodePtr(new Index::VPNodeType);

        Index::VPNodePtr new_node(new Index::VPNodeType);

        new_node->set_parent(parent);

        // Set the VP
        new_node->set_value(SelectVP(objects));

        if (objects.size() <= index->vp_tree.m_non_leaf_branching_factor * index->vp_tree.m_leaf_branching_factor) {
            for (size_t c_pos = 0; c_pos < index->vp_tree.m_leaf_branching_factor; ++c_pos) {
                new_node->AddChild(0, Index::VPNodePtr(new Index::VPNodeType));
                new_node->m_child_list[c_pos]->set_leaf_node(true);
                new_node->m_child_list[c_pos]->set_parent(new_node);
            }

            new_node->m_child_list[0]->m_objects_list.insert(new_node->m_child_list[0]->m_objects_list.begin(),
                                                             objects.begin() + 1, objects.end());

            RedistributeAmongLeafNodes(new_node, *objects.begin());

            return new_node;
        }

        // Init children
        new_node->AddChild(0, Index::VPNodePtr(new Index::VPNodeType));
        new_node->AddChild(0, Index::VPNodePtr(new Index::VPNodeType));

        float median = Median(new_node->get_value(), objects.begin(), objects.end());
        new_node->m_mu_list[0] = median;

        size_t objects_count = objects.size();
        if (median == 0)
            objects_count = 0;

        bool c_left = false;

        // 60% of size
        size_t reserved_memory = static_cast<size_t>(static_cast<double>(objects_count) * 0.6);
        std::vector<unsigned> s_left, s_right;
        s_left.reserve(reserved_memory);
        s_right.reserve(reserved_memory);

        typename std::vector<unsigned>::const_iterator it_obj = objects.begin();
        while (it_obj != objects.end()) {
            float dist = index->getDist()->compare(index->getBaseData() + index->getBaseDim() * new_node->get_value(),
                                                   index->getBaseData() + index->getBaseDim() * (*it_obj),
                                                   index->getBaseDim());
            if (dist < new_node->m_mu_list[0] || (dist == 0 && !c_left)) {
                s_left.push_back(*it_obj);
                c_left = true;
            } else {
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
    void ComponentPreEntryVPTree::SplitNonLeafNode(const Index::VPNodePtr &parent_node, const size_t child_id,
                                                   const unsigned &new_value) {
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

        if (F > 1)
            for (size_t c_pos = F - 2; c_pos >= k; --c_pos) {
                parent_node->m_mu_list[c_pos + 1] = parent_node->m_mu_list[c_pos];
                if (!c_pos) // !!! hack :(
                    break;
            }

        parent_node->m_mu_list[k] = MedianSumm(SS1, SS2, parent_node->get_value());
        for (size_t c_pos = F - 1; c_pos >= k + 1; --c_pos)
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
        parent_node->m_child_list[k + 1] = ss2_node;
    }

    void ComponentPreEntryVPTree::Insert(const unsigned& new_value, Index::VPNodePtr& root) {
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

    void ComponentPreEntryVPTree::Insert(const unsigned int &new_value) {
        //std::cout << "Insert data to vptree.root" << std::endl;

        Insert(new_value, index->vp_tree.m_root);
    }

    /**
    * VP-tree 获取近邻
    * @param query 查询点
    * @param pool 候选池
    */
    void ComponentSearchEntryVPT::SearchEntryInner(unsigned int query, std::vector<Index::Neighbor> &pool) {
        float cq = static_cast<float>(FLT_MAX);
        //float cq = 100;
        std::multimap<float, unsigned> result_found;
        Search(query, NGT_SEED_SIZE, result_found, index->vp_tree.get_root(), cq);

        for(auto it : result_found) {
            pool.emplace_back(it.second, it.first, true);
        }
    }


    void ComponentSearchEntryVPT::Search(const unsigned& query_value, const size_t count, std::multimap<float, unsigned> &pool,
                                         const Index::VPNodePtr& node, float& q)
    {
        assert(node.get());

        if(node->get_leaf_node())
        {
            for(size_t c_pos = 0; c_pos < node->m_objects_list.size(); ++c_pos)
            {
                //m_stat.distance_threshold_count++;
                float c_distance = index->getDist()->compare(index->getQueryData() + index->getQueryDim() * query_value,
                                                             index->getBaseData() + index->getBaseDim() * node->m_objects_list[c_pos],
                                                             index->getBaseDim());
                index->addDistCount();
                if( c_distance <= q)
                {
                    pool.insert(std::pair<float, unsigned>(c_distance, node->m_objects_list[c_pos]));

                    while(pool.size() > count)
                    {
                        typename std::multimap<float, unsigned>::iterator it_last = pool.end();

                        pool.erase(--it_last);
                    }

                    if(pool.size() == count)
                        q = (*pool.rbegin()).first;
                }
            }

        }else{
            float dist = 0; //m_get_distance(node->get_value(), query_value);

            // Search flag
            size_t c_mu_pos = index->vp_tree.m_non_leaf_branching_factor;
            if(node->m_mu_list.size() == 1)
            {
                c_mu_pos = 0;
                //m_stat.distance_threshold_count++;
                //dist = m_get_distance(node->get_value(), query_value, node->m_mu_list[c_mu_pos] + q);
                dist = index->getDist()->compare(index->getBaseData() + index->getBaseDim() * node->get_value(),
                                                 index->getQueryData() + index->getQueryDim() * query_value,
                                                 index->getBaseDim());
                index->addDistCount();
            }else{
                //m_stat.distance_count++;
                dist = index->getDist()->compare(index->getBaseData() + index->getBaseDim() * node->get_value(),
                                                 index->getQueryData() + index->getQueryDim() * query_value,
                                                 index->getBaseDim());
                index->addDistCount();
                //dist = m_get_distance(node->get_value(), query_value);
                for(size_t c_pos = 0; c_pos < node->m_mu_list.size() -1 ; ++c_pos)
                {
                    if(dist > node->m_mu_list[c_pos] && dist < node->m_mu_list[c_pos + 1] )
                    {
                        c_mu_pos = c_pos;
                        break;
                    }
                }
            }

            if(c_mu_pos != index->vp_tree.m_non_leaf_branching_factor)
            {
                float c_mu = node->m_mu_list[c_mu_pos];
                if(dist < c_mu)
                {
                    if(dist < c_mu + q)
                    {
                        //m_stat.search_jump++;
                        Search(query_value, count, pool, node->m_child_list[c_mu_pos], q);
                    }
                    if(dist >= c_mu - q)
                    {
                        //m_stat.search_jump++;
                        Search(query_value, count, pool, node->m_child_list[c_mu_pos + 1], q);
                    }
                }else
                {
                    if(dist >= c_mu - q)
                    {
                        //m_stat.search_jump++;
                        Search(query_value, count, pool, node->m_child_list[c_mu_pos + 1], q);
                    }
                    if(dist < c_mu + q)
                    {
                        //m_stat.search_jump++;
                        Search(query_value, count, pool, node->m_child_list[c_mu_pos], q);
                    }
                }
            }
        }

    }




    /**
     * K-Means
     *
     * 需要指定 m_iBKTKmeansK，numOfThreads，m_iTreeNumber
     */
    void ComponentPreEntryBKTree::PreEntryInner() {
        SetConfigs();

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
        for (int i = 0; i < index->m_iTreeNumber; i++) {
            std::random_shuffle(localindices.begin(), localindices.end());

            index->m_pTreeStart.push_back((int) index->m_pBKTreeRoots.size());
            index->m_pBKTreeRoots.emplace_back((int) localindices.size());

            std::cout << "Start to build BKTree " << i + 1 << std::endl;

            ss.push(BKTStackItem(index->m_pTreeStart[i], 0, (int) localindices.size()));
            while (!ss.empty()) {
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
                    int numClusters = KmeansClustering(localindices, item.first, item.last, args, index->m_iSamples);
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
                }
                index->m_pBKTreeRoots[item.bkt_index].childEnd = (int) index->m_pBKTreeRoots.size();
            }
            index->m_pBKTreeRoots.emplace_back(-1);
            //std::cout << i + 1 << " BKTree built, " << index->m_pBKTreeRoots.size() - index->m_pTreeStart[i] << localindices.size() << std::endl;
        }
    }

    void ComponentPreEntryBKTree::SetConfigs() {
        index->m_iBKTKmeansK = index->getParam().get<unsigned>("BKTKMeansK");
        index->m_iTreeNumber = index->getParam().get<unsigned>("BKTNumber");
        index->numOfThreads = index->getParam().get<unsigned>("numOfThreads");
    }

    int ComponentPreEntryBKTree::KmeansClustering(std::vector<int> &indices, const int first,
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

    float ComponentPreEntryBKTree::KmeansAssign(std::vector<int> &indices, const int first,
                                               const int last, Index::KmeansArgs<float> &args,
                                               const bool updateCenters, float lambda) {
        float currDist = 0;
        int subsize = (last - first - 1) / args._T + 1;
        const float MaxDist = (std::numeric_limits<float>::max)();

//#pragma omp parallel for num_threads(args._T) shared(test, indices) reduction(+:currDist)
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

    int  ComponentPreEntryBKTree::rand(int high, int low) {
        return low + (int) (float(high - low) * (std::rand() / (RAND_MAX + 1.0)));
    }

    void ComponentPreEntryBKTree::InitCenters(std::vector<int> &indices, const int first,
                                             const int last, Index::KmeansArgs<float> &args, int samples,
                                             int tryIters) {
        const float MaxDist = (std::numeric_limits<float>::max)();

        int batchEnd = std::min(first + samples, last);
        float currDist, minClusterDist = MaxDist;
        for (int numKmeans = 0; numKmeans < tryIters; numKmeans++) {
            for (int k = 0; k < args._DK; k++) {
                int randid = rand(last, first);
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

    float ComponentPreEntryBKTree::RefineCenters(Index::KmeansArgs<float> &args) {
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

    /**
     * K-MEANS
     *
     * @param query 查询点
     * @param pool 入口点
     */
    void ComponentSearchEntryBKT::SearchEntryInner(unsigned int query, std::vector<Index::Neighbor> &pool) {
        const auto L = index->getParam().get<unsigned>("L_search");

        unsigned m_iNumberOfCheckedLeaves = 0;

        // Prioriy queue used for neighborhood graph
        Index::Heap m_NGQueue;
        m_NGQueue.Resize(L + 1);

        // Priority queue Used for Tree
        Index::Heap m_SPTQueue;
        m_SPTQueue.Resize(100);

        Index::OptHashPosVector nodeCheckStatus;
        nodeCheckStatus.Init(index->getBaseLen(), index->m_iHashTableExp);
        nodeCheckStatus.CheckAndSet(query);

        for (unsigned i = 0; i < index->m_iTreeNumber; i++) {
            const Index::BKTNode& node = index->m_pBKTreeRoots[index->m_pTreeStart[i]];
            if (node.childStart < 0) {
                float dist = index->getDist()->compare(index->getQueryData() + index->getQueryDim() * query,
                                                       index->getBaseData() + index->getBaseDim() * node.centerid,
                                                       index->getBaseDim());
                index->addDistCount();
                m_SPTQueue.insert(Index::HeapCell(index->m_pTreeStart[i], dist));
            } else {
                for (int begin = node.childStart; begin < node.childEnd; begin++) {
                    int tmp = index->m_pBKTreeRoots[begin].centerid;
                    float dist = index->getDist()->compare(index->getQueryData() + index->getQueryDim() * query,
                                                           index->getBaseData() + index->getBaseDim() * tmp,
                                                           index->getBaseDim());
                    index->addDistCount();
                    m_SPTQueue.insert(Index::HeapCell(begin, dist));
                }
            }
        }

        BKTSearch(query, m_NGQueue, m_SPTQueue, nodeCheckStatus, m_iNumberOfCheckedLeaves, L);

        while(!m_NGQueue.empty()) {
            Index::HeapCell cell = m_NGQueue.pop();
            pool.push_back(Index::Neighbor(cell.node, cell.distance, true));
        }

        std::sort(pool.begin(), pool.end());
    }

    void ComponentSearchEntryBKT::BKTSearch(unsigned int query, Index::Heap &m_NGQueue,
                                                  Index::Heap &m_SPTQueue, Index::OptHashPosVector &nodeCheckStatus,
                                                  unsigned int &m_iNumberOfCheckedLeaves,
                                                  int p_limits) {
        while (!m_SPTQueue.empty())
        {
            Index::HeapCell bcell = m_SPTQueue.pop();
            const Index::BKTNode& tnode = index->m_pBKTreeRoots[bcell.node];
            if (tnode.childStart < 0) {
                if (!nodeCheckStatus.CheckAndSet(tnode.centerid)) {
                    m_iNumberOfCheckedLeaves++;
                    m_NGQueue.insert(Index::HeapCell(tnode.centerid, bcell.distance));
                }
                if (m_iNumberOfCheckedLeaves >= p_limits) break;
            }
            else {
                if (!nodeCheckStatus.CheckAndSet(tnode.centerid)) {
                    m_NGQueue.insert(Index::HeapCell(tnode.centerid, bcell.distance));
                    m_iNumberOfCheckedLeaves++;
                }
                if (m_iNumberOfCheckedLeaves >= p_limits) break;
                for (int begin = tnode.childStart; begin < tnode.childEnd; begin++) {
                    int tmp = index->m_pBKTreeRoots[begin].centerid;
                    float dist = index->getDist()->compare(index->getQueryData() + index->getQueryDim() * query,
                                                           index->getBaseData() + index->getBaseDim() * tmp,
                                                           index->getBaseDim());
                    index->addDistCount();
                    m_SPTQueue.insert(Index::HeapCell(begin, dist));
                }
            }
        }
    }




    /**
     * 导向搜索
     */
    void ComponentPreEntryGuided::PreEntryInner() {
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











    /**
     * SSG 随机入口点
     * @param query 查询点
     * @param pool 候选池
     */
    void ComponentSearchEntrySSG::SearchEntryInner(unsigned int query, std::vector<Index::Neighbor> &pool) {
        auto L = index->getParam().get<unsigned>("L_search");
        pool.resize(L + 1);
        std::vector<unsigned> init_ids(L);
        std::mt19937 rng(rand());
        GenRandom(rng, init_ids.data(), L, (unsigned) index->getBaseLen());

        assert(index->getSeeds().size() <= L);
        for (unsigned i = 0; i < index->getSeeds().size(); i++) {
            init_ids[i] = index->getSeeds()[i];
        }

        boost::dynamic_bitset<> flags{index->getBaseLen(), 0};
        L = 0;
        for (unsigned i = 0; i < init_ids.size(); i++) {
            unsigned id = init_ids[i];
            if (id >= index->getBaseLen()) continue;
            float *x = (float *) (index->getBaseData() + index->getBaseDim() * id);
            float norm_x = *x;
            x++;
            float dist = index->getDist()->compare(x, index->getQueryData() + query * index->getQueryDim(),
                                                   (unsigned) index->getBaseDim());
            index->addDistCount();
            pool[i] = Index::Neighbor(id, dist, true);
            flags[id] = true;
            L++;
        }

        std::sort(pool.begin(), pool.begin() + L);
    }
}
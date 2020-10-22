//
// Created by Murph on 2020/9/14.
//

#include "weavess/component.h"

namespace weavess {

    void ComponentRefineEntryCentroid::EntryInner() {
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
        index->ep_ = rand() % index->getBaseLen();  // random initialize navigating point
        get_neighbors(center, tmp, pool);
        index->ep_ = tmp[0].id;

        std::cout << "ep_ " << index->ep_ << std::endl;
    }

    void ComponentRefineEntryCentroid::get_neighbors(const float *query, std::vector<Index::Neighbor> &retset,
                                                     std::vector<Index::Neighbor> &fullset) {
        unsigned L = index->L_refine;
        retset.resize(L + 1);
        std::vector<unsigned> init_ids(L);
        // initializer_->Search(query, nullptr, L, parameter, init_ids.data());
        boost::dynamic_bitset<> flags{index->getBaseLen(), 0};
        L = 0;
        for (unsigned i = 0; i < init_ids.size() && i < index->getFinalGraph()[index->ep_][0].size(); i++) {
            init_ids[i] = index->getFinalGraph()[index->ep_][0][i];
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
            // std::cout<<id<<std::endl;
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
                for (unsigned m = 0; m < index->getFinalGraph()[n][0].size(); ++m) {
                    unsigned id = index->getFinalGraph()[n][0][m];
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
     * 随机入口点
     * @param query 查询点
     * @param pool 候选池
     */
    void ComponentSearchEntryRand::SearchEntryInner(unsigned query, std::vector<Index::Neighbor> &pool) {
        const auto L = index->getParam().get<unsigned>("L_search");

        pool.clear();
        pool.resize(L + 1);

        std::vector<unsigned> init_ids(L);
        std::mt19937 rng(rand());

        GenRandom(rng, init_ids.data(), L, (unsigned) index->getBaseLen());
        std::vector<char> flags(index->getBaseLen());
        memset(flags.data(), 0, index->getBaseLen() * sizeof(char));
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
     * 质心入口点
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

        unsigned tmp_l = 0;
        for (; tmp_l < L && tmp_l < index->getFinalGraph()[index->ep_][0].size(); tmp_l++) {
            init_ids[tmp_l] = index->getFinalGraph()[index->ep_][0][tmp_l];
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
            pool[i] = Index::Neighbor(id, dist, true);
            // flags[id] = true;
        }

        std::sort(pool.begin(), pool.begin() + L);
    }

    void ComponentSearchEntrySubCentroid::SearchEntryInner(unsigned int query, std::vector<Index::Neighbor> &pool) {
        auto L = index->getParam().get<unsigned>("L_search");
        //std::vector<Neighbor> retset(L + 1);
        pool.resize(L + 1);
        std::vector<unsigned> init_ids(L);
        std::mt19937 rng(rand());
        GenRandom(rng, init_ids.data(), L, (unsigned) index->getBaseLen());

        assert(index->eps_.size() <= L);
        for (unsigned i = 0; i < index->eps_.size(); i++) {
            init_ids[i] = index->eps_[i];
        }

        boost::dynamic_bitset<> flags{index->getBaseLen(), 0};
        L = 0;
        for (unsigned i = 0; i < init_ids.size(); i++) {
            unsigned id = init_ids[i];
            if (id >= index->getBaseLen()) continue;
            float *x = (float *) (index->getBaseData() + index->getBaseDim() * id);
            float norm_x = *x;
            x++;
            float dist = index->getDist()->compare(x, index->getBaseData() + query * index->getBaseDim(),
                                                   (unsigned) index->getBaseDim());
            pool[i] = Index::Neighbor(id, dist, true);
            flags[id] = true;
            L++;
        }
        // std::cout<<L<<std::endl;

        std::sort(pool.begin(), pool.begin() + L);
    }

//    void ComponentSearchEntryRandom::SearchEntryInner(unsigned int query, std::vector<Index::Neighbor> &pool) {
//        const auto L = index->getParam().get<unsigned>("L_search");
//        pool.reserve(L + 1);
//        std::vector<unsigned> init_ids(L);
//        boost::dynamic_bitset<> flags{index->getBaseLen(), 0};
//        // std::mt19937 rng(rand());
//        // GenRandom(rng, init_ids.data(), L, (unsigned) index_->n_);
//
//        unsigned tmp_l = 0;
//        for (; tmp_l < L && tmp_l < index->getFinalGraph()[index->ep_][0].size(); tmp_l++) {
//            init_ids[tmp_l] = index->getFinalGraph()[index->ep_][0][tmp_l];
//            flags[init_ids[tmp_l]] = true;
//        }
//
//        while (tmp_l < L) {
//            unsigned id = rand() % index->getBaseLen();
//            if (flags[id]) continue;
//            flags[id] = true;
//            init_ids[tmp_l] = id;
//            tmp_l++;
//        }
//
//        for (unsigned i = 0; i < init_ids.size(); i++) {
//            unsigned id = init_ids[i];
//            float dist =
//                    index->getDist()->compare(index->getBaseData() + index->getBaseDim() * id,
//                                              index->getQueryData() + index->getQueryDim() * query,
//                                              (unsigned)index->getBaseDim());
//            pool[i] = Index::Neighbor(id, dist, true);
//            // flags[id] = true;
//        }
//
//        std::sort(pool.begin(), pool.begin() + L);
//    }
}
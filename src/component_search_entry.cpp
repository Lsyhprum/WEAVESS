//
// Created by MurphySL on 2020/10/23.
//

#include "weavess/component.h"

namespace weavess {
    /**
     * L 个随机入口点
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
        for (; tmp_l < L && tmp_l < index->getFinalGraph()[index->ep_].size(); tmp_l++) {
            init_ids[tmp_l] = index->getFinalGraph()[index->ep_][tmp_l].id;
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

    /**
     * 子图质心入口点
     * @param query 查询点
     * @param pool 候选池
     */
    void ComponentSearchEntrySubCentroid::SearchEntryInner(unsigned int query, std::vector<Index::Neighbor> &pool) {
        auto L = index->getParam().get<unsigned>("L_search");
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

        std::sort(pool.begin(), pool.begin() + L);
    }

    /**
     * KDT获取入口点
     * @param query 查询点
     * @param pool 候选池
     */
    void ComponentSearchEntryKDT::SearchEntryInner(unsigned int query, std::vector<Index::Neighbor> &pool) {

    }
}

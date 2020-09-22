//
// Created by Murph on 2020/9/14.
//

#include "weavess/component.h"

namespace weavess {

    void ComponentEntryCentroidNSG::EntryInner() {
        std::cout << "__REFINE : Single Centroid__" << std::endl;
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
    }

    void ComponentEntryCentroidNSG::get_neighbors(const float *query, std::vector<Index::Neighbor> &retset,
                                           std::vector<Index::Neighbor> &fullset) {
        unsigned L = index->L;
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
            float dist = index->getDist()->compare(index->getBaseData() + index->getBaseDim() * (size_t)id, query,
                                                   (unsigned)index->getBaseDim());
            retset[i] = Index::Neighbor(id, dist, true);
            //retset[i] = new Index::Node(id, dist, true, 0);
            // flags[id] = 1;
            L++;
        }
        std::sort(retset.begin(), retset.begin() + L);
        int k = 0;
        while (k < (int)L) {
            int nk = L;

            if (retset[k].flag) {
                retset[k].flag = false;
                unsigned n = retset[k].id;
                for (unsigned m = 0; m < index->getFinalGraph()[n][0].size(); ++m) {
                    unsigned id = index->getFinalGraph()[n][0][m];
                    if (flags[id]) continue;
                    flags[id] = 1;

                    float dist = index->getDist()->compare(query, index->getBaseData() + index->getBaseDim() * (size_t)id,
                                                           (unsigned)index->getBaseDim());
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

    void ComponentEntryCentroidNSSG::EntryInner() {
        std::cout << "__ENTRY : NSSG__ " << std::endl;
        float *center = new float[index->getBaseDim()];
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
        // ep_ = rand() % nd_;  // random initialize navigating point
        get_neighbors(center, tmp, pool);
        //std::cout << "pool size : " << pool.size() << std::endl;
        index->ep_ = tmp[0].id;  // For Compatibility
        std::cout << "__ENTRY : FINISH__ " << std::endl;
    }

    void ComponentEntryCentroidNSSG::get_neighbors(const float *query, std::vector<Index::Neighbor> &retset,
                                 std::vector<Index::Neighbor> &fullset) {
        unsigned L = index->L_nsg;

        retset.resize(L + 1);
        std::vector<unsigned> init_ids(L);
        // initializer_->Search(query, nullptr, L, parameter, init_ids.data());
        std::mt19937 rng(rand());
        GenRandom(rng, init_ids.data(), L, (unsigned)index->getBaseLen());

        boost::dynamic_bitset<> flags{index->getBaseLen(), 0};
        L = 0;
        for (unsigned i = 0; i < init_ids.size(); i++) {
            unsigned id = init_ids[i];
            if (id >= index->getBaseLen()) continue;
            // std::cout<<id<<std::endl;
            float dist = index->getDist()->compare(index->getBaseData() + index->getBaseDim() * (size_t)id, query,
                                            (unsigned)index->getBaseDim());
            retset[i] = Index::Neighbor(id, dist, true);
            flags[id] = 1;
            L++;
        }

        std::sort(retset.begin(), retset.begin() + L);
        int k = 0;
        while (k < (int)L) {
            int nk = L;

            if (retset[k].flag) {
                retset[k].flag = false;
                unsigned n = retset[k].id;

                for (unsigned m = 0; m < index->getFinalGraph()[n][0].size(); ++m) {
                    unsigned id = index->getFinalGraph()[n][0][m];
                    if (flags[id]) continue;
                    flags[id] = 1;

                    float dist = index->getDist()->compare(query, index->getBaseData() + index->getBaseDim() * (size_t)id,
                                                    (unsigned)index->getBaseDim());

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


}
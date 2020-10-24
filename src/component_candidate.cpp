//
// Created by MurphySL on 2020/10/23.
//

#include "weavess/component.h"

namespace weavess {

    // NO LIMIT GREEDY
    void
    ComponentCandidateNSG::CandidateInner(const unsigned query, const unsigned enter, boost::dynamic_bitset<> flags,
                                          std::vector<Index::SimpleNeighbor> &result) {
        auto L = index->getParam().get<unsigned>("L_refine");

        std::vector<unsigned> init_ids(L);
        std::vector<Index::Neighbor> retset;
        retset.resize(L + 1);

        L = 0;
        // 选取质点近邻作为初始候选点
        for (unsigned i = 0; i < init_ids.size() && i < index->getFinalGraph()[enter].size(); i++) {
            init_ids[i] = index->getFinalGraph()[enter][i].id;
            flags[init_ids[i]] = true;
            L++;
        }
        // 候选点不足填入随机点
        while (L < init_ids.size()) {
            unsigned id = rand() % index->getBaseLen();
            if (flags[id]) continue;
            init_ids[L] = id;
            L++;
            flags[id] = true;
        }
        L = 0;
        // unsinged -> SimpleNeighbor
        for (unsigned i = 0; i < init_ids.size(); i++) {
            unsigned id = init_ids[i];
            if (id >= index->getBaseLen()) continue;
            float dist = index->getDist()->compare(index->getBaseData() + index->getBaseDim() * id,
                                                   index->getBaseData() + index->getBaseDim() * query,
                                                   (unsigned) index->getBaseDim());

            retset[i] = Index::Neighbor(id, dist, true);
            result.emplace_back(Index::SimpleNeighbor(id, dist));

            L++;
        }
        std::sort(retset.begin(), retset.begin() + L);
        index->i++;

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

                    float dist = index->getDist()->compare(index->getBaseData() + index->getBaseDim() * query,
                                                           index->getBaseData() + index->getBaseDim() * (size_t) id,
                                                           (unsigned) index->getBaseDim());

                    Index::Neighbor nn(id, dist, true);
                    result.push_back(Index::SimpleNeighbor(id, dist));

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

        std::vector<Index::Neighbor>().swap(retset);
        std::vector<unsigned>().swap(init_ids);
    }

    // PROPAGATION 2
    void ComponentCandidatePropagation2::CandidateInner(const unsigned query, const unsigned enter,
                                                        boost::dynamic_bitset<> flags,
                                                        std::vector<Index::SimpleNeighbor> &pool) {
        flags[enter] = true;

        for (unsigned i = 0; i < index->getFinalGraph()[enter].size(); i++) {
            unsigned nid = index->getFinalGraph()[enter][i].id;
            for (unsigned nn = 0; nn < index->getFinalGraph()[nid].size(); nn++) {
                unsigned nnid = index->getFinalGraph()[nid][nn].id;
                if (flags[nnid]) continue;
                flags[nnid] = true;
                float dist = index->getDist()->compare(index->getBaseData() + index->getBaseDim() * query,
                                                       index->getBaseData() + index->getBaseDim() * nnid,
                                                       index->getBaseDim());
                pool.emplace_back(nnid, dist);
                if (pool.size() >= index->L_refine) break;
            }
            if (pool.size() >= index->L_refine) break;
        }
    }

}
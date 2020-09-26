//
// Created by Murph on 2020/9/14.
//

#include "weavess/component.h"

namespace weavess {

    // LIMIT GREEDY
    void ComponentCandidateGreedy::CandidateInner(const unsigned int query, const unsigned int enter,
                                                  boost::dynamic_bitset<> flags, std::vector<Index::Neighbor> &result,
                                                  int level) {
        auto L = index->getParam().get<unsigned>("L_nsg");

        std::vector<unsigned> init_ids(L);
        std::vector<Index::Neighbor> retset;
        retset.resize(L + 1);

        L = 0;
        // 选取质点近邻作为初始候选点
        for(unsigned i = 0; i < init_ids.size() && i < index->getFinalGraph()[enter][level].size(); i ++){
            init_ids[i] = index->getFinalGraph()[enter][level][i];
            flags[init_ids[i]] = true;
            L++;
        }
        // 候选点不足填入随机点
        while(L < init_ids.size()) {
            unsigned id = rand() % index->getBaseLen();
            if(flags[id]) continue;
            init_ids[L] = id;
            L++;
            flags[id] = true;
        }
        L = 0;
        // unsinged -> Neighbor
        for(unsigned i = 0; i < init_ids.size(); i ++) {
            unsigned id = init_ids[i];
            if (id >= index->getBaseLen()) continue;
            float dist = index->getDist()->compare(index->getBaseData() + index->getBaseDim() * id,
                                                   index->getBaseData() + index->getBaseDim() * query,
                                                   (unsigned)index->getBaseDim());

            retset[i] = Index::Neighbor(id, dist, true);

            L++;
        }
        std::sort(retset.begin(), retset.begin() + L);
        index->i ++;

        int k = 0;
        while (k < (int)L) {
            int nk = L;
            if (retset[k].flag) {
                retset[k].flag = false;
                unsigned n = retset[k].id;
                for (unsigned m = 0; m < index->getFinalGraph()[n][level].size(); ++m) {

                    unsigned id = index->getFinalGraph()[n][level][m];

                    if(flags[id]) continue;
                    flags[id] = true;

                    float dist = index->getDist()->compare(index->getBaseData() + index->getBaseDim() * query,
                                                           index->getBaseData() + index->getBaseDim() * (size_t)id,
                                                           (unsigned)index->getBaseDim());

                    Index::Neighbor nn(id, dist, true);

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

        for(int i = 0; i < retset.size(); i ++ ){
            result.push_back(retset[i]);
        }

        std::vector<Index::Neighbor>().swap(retset);
        std::vector<unsigned>().swap(init_ids);
    }

    // NO LIMIT GREEDY
    void ComponentCandidateNSG::CandidateInner(const unsigned query, const unsigned enter, boost::dynamic_bitset<> flags,
                                               std::vector<Index::Neighbor> &result, int level) {
        auto L = index->getParam().get<unsigned>("L_nsg");

        std::vector<unsigned> init_ids(L);
        std::vector<Index::Neighbor> retset;
        retset.resize(L + 1);

        L = 0;
        // 选取质点近邻作为初始候选点
        for(unsigned i = 0; i < init_ids.size() && i < index->getFinalGraph()[enter][level].size(); i ++){
            init_ids[i] = index->getFinalGraph()[enter][level][i];
            flags[init_ids[i]] = true;
            L++;
        }
        // 候选点不足填入随机点
        while(L < init_ids.size()) {
            unsigned id = rand() % index->getBaseLen();
            if(flags[id]) continue;
            init_ids[L] = id;
            L++;
            flags[id] = true;
        }
        L = 0;
        // unsinged -> Neighbor
        for(unsigned i = 0; i < init_ids.size(); i ++) {
            unsigned id = init_ids[i];
            if (id >= index->getBaseLen()) continue;
            float dist = index->getDist()->compare(index->getBaseData() + index->getBaseDim() * id,
                                                   index->getBaseData() + index->getBaseDim() * query,
                                            (unsigned)index->getBaseDim());

            retset[i] = Index::Neighbor(id, dist, true);
            result.emplace_back(retset[i]);

            L++;
        }
        std::sort(retset.begin(), retset.begin() + L);
        index->i ++;

        int k = 0;
        while (k < (int)L) {
            int nk = L;
            if (retset[k].flag) {
                retset[k].flag = false;
                unsigned n = retset[k].id;
                for (unsigned m = 0; m < index->getFinalGraph()[n][level].size(); ++m) {

                    unsigned id = index->getFinalGraph()[n][level][m];

                    if(flags[id]) continue;
                    flags[id] = true;

                    float dist = index->getDist()->compare(index->getBaseData() + index->getBaseDim() * query,
                                                           index->getBaseData() + index->getBaseDim() * (size_t)id,
                                                    (unsigned)index->getBaseDim());

                    Index::Neighbor nn(id, dist, true);
                    result.push_back(nn);

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
    void ComponentCandidateNSSG::CandidateInner(const unsigned query, const unsigned enter,
                                                boost::dynamic_bitset<> flags, std::vector<Index::Neighbor> &result,
                                                int level) {
        flags[query] = true;

        for (unsigned i = 0; i < index->getFinalGraph()[query][0].size(); i++) {
            unsigned nid = index->getFinalGraph()[query][0][i];
            for (unsigned nn = 0; nn < index->getFinalGraph()[nid][0].size(); nn++) {
                unsigned nnid = index->getFinalGraph()[nid][0][nn];
                if (flags[nnid]) continue;
                flags[nnid] = true;
                float dist = index->getDist()->compare(index->getBaseData() + index->getBaseDim() * query,
                                                       index->getBaseData() + index->getBaseDim() * nnid, index->getBaseDim());
                result.emplace_back(nnid, dist, true);
                if (result.size() >= index->L_nsg) break;
            }
            if (result.size() >= index->L_nsg) break;
        }
    }

    // PROPAGATION 1
    void ComponentCandidatePropagation1::CandidateInner(const unsigned int query, const unsigned int enter,
                                                        boost::dynamic_bitset<> flags, std::vector<Index::Neighbor> &result,
                                                        int level) {
        for (unsigned i = 0; i < index->getFinalGraph()[query][0].size(); i++) {
            unsigned nid = index->getFinalGraph()[query][0][i];

            float dist = index->getDist()->compare(index->getBaseData() + index->getBaseDim() * query,
                                                   index->getBaseData() + index->getBaseDim() * nid, index->getBaseDim());
            result.emplace_back(nid, dist, true);
        }

        std::sort(result.begin(), result.begin() + index->getFinalGraph()[query][0].size());
    }
}
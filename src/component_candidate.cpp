//
// Created by Murph on 2020/9/14.
//

#include "weavess/component.h"

namespace weavess {

    void ComponentCandidateNSG::CandidateInner(const unsigned query, const unsigned enter, Index::VisitedList *visited_list,
                                               std::vector<Index::Neighbor> &result, int level) {
        unsigned L = index->L_nsg;
        std::vector<unsigned> init_ids(L);

        std::vector<Index::Neighbor> retset;
        retset.resize(L + 1);

        result.clear();
        visited_list->Reset();

        L = 0;
        for(unsigned i = 0; i < init_ids.size() && i < index->getFinalGraph()[enter][level].size(); i ++){
            init_ids[i] = index->getFinalGraph()[enter][level][i];
            visited_list->MarkAsVisited(init_ids[i]);
            L++;
        }
        while(L < init_ids.size()) {
            unsigned id = rand() % index->getBaseLen();
            if(visited_list->Visited(id)) continue;
            init_ids[L] = id;
            L++;
            visited_list->MarkAsVisited(id);
        }
        L = 0;
        for(unsigned i = 0; i < init_ids.size(); i ++) {
            unsigned id = init_ids[i];
            if (id >= index->getBaseLen()) continue;
            // std::cout<<id<<std::endl;
            float dist = index->getDist()->compare(index->getBaseData() + index->getBaseDim() * id,
                                                   index->getBaseData() + index->getBaseDim() * query,
                                            (unsigned)index->getBaseDim());

            retset[i] = Index::Neighbor(id, dist, true);
            result.emplace_back(retset[i]);

            L++;
        }
        std::sort(result.begin(), result.begin() + L);

        int k = 0;
        while (k < (int)L) {
            int nk = L;

            if (retset[k].flag) {
                retset[k].flag = false;
                unsigned n = retset[k].id;

                for (unsigned m = 0; m < index->getFinalGraph()[n][level].size(); ++m) {
                    unsigned id = index->getFinalGraph()[n][level][m];
                    if(visited_list->Visited(id)) continue;
                    visited_list->MarkAsVisited(id);

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
        //std::cout << "candidate finish" << std::endl;
    }

    void ComponentCandidateNSSG::CandidateInner(const unsigned query, const unsigned enter,
                                                Index::VisitedList *visited_list, std::vector<Index::Neighbor> &result,
                                                int level) {
        visited_list->MarkAsVisited(query);

        for (unsigned i = 0; i < index->getFinalGraph()[query][0].size(); i++) {
            unsigned nid = index->getFinalGraph()[query][0][i];
            for (unsigned nn = 0; nn < index->getFinalGraph()[nid][0].size(); nn++) {
                unsigned nnid = index->getFinalGraph()[nid][0][nn];
                if(visited_list->Visited(nnid)) continue;
                visited_list->MarkAsVisited(nnid);
                float dist = index->getDist()->compare(index->getBaseData() + index->getBaseDim() * query,
                                                       index->getBaseData() + index->getBaseDim() * nnid, index->getBaseDim());
                result.emplace_back(nnid, dist, true);
                if (result.size() >= index->L) break;
            }
            if (result.size() >= index->L) break;
        }
    }

    void ComponentCandidateNone::CandidateInner(const unsigned int query, const unsigned int enter,
                                                Index::VisitedList *visited_list, std::vector<Index::Neighbor> &result,
                                                int level) {
        for (unsigned i = 0; i < index->getFinalGraph()[query][0].size(); i++) {
            unsigned nid = index->getFinalGraph()[query][0][i];

            float dist = index->getDist()->compare(index->getBaseData() + index->getBaseDim() * query,
                                                   index->getBaseData() + index->getBaseDim() * nid, index->getBaseDim());
            result.emplace_back(nid, dist, true);
        }
    }
}
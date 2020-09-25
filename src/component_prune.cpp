//
// Created by Murph on 2020/9/14.
//

#include "weavess/component.h"

namespace weavess {
    void ComponentPruneNaive::PruneInner(unsigned int q, unsigned int range, boost::dynamic_bitset<> flags,
                                         std::vector<Index::Neighbor> &result, Index::SimpleNeighbor *cut_graph_, unsigned int level) {
        while(result.size() > range)
            result.pop_back();
    }

    void ComponentPruneNSG::PruneInner(unsigned q, unsigned range, boost::dynamic_bitset<> flags,
                                    std::vector<Index::Neighbor> &pool, Index::SimpleNeighbor *cut_graph_, unsigned level) {
        unsigned maxc = index->C_nsg;

        unsigned start = 0;

        for (unsigned nn = 0; nn < index->getFinalGraph()[q][level].size(); nn++) {
            unsigned id = index->getFinalGraph()[q][level][nn];
            if (flags[id]) continue;
            float dist =
                    index->getDist()->compare(index->getBaseData() + index->getBaseDim() * (size_t)q,
                                       index->getBaseData() + index->getBaseDim() * (size_t)id, (unsigned)index->getBaseDim());
            pool.push_back(Index::Neighbor(id, dist, true));
        }

        std::sort(pool.begin(), pool.end());
        std::vector<Index::Neighbor> result;
        if (pool[start].id == q) start++;
        result.push_back(pool[start]);

        while (result.size() < range && (++start) < pool.size() && start < maxc) {
            auto &p = pool[start];
            bool occlude = false;
            for (unsigned t = 0; t < result.size(); t++) {
                if (p.id == result[t].id) {
                    occlude = true;
                    break;
                }
                float djk = index->getDist()->compare(index->getBaseData() + index->getBaseDim() * (size_t)result[t].id,
                                               index->getBaseData() + index->getBaseDim() * (size_t)p.id,
                                               (unsigned)index->getBaseDim());
                if (djk < p.distance /* dik */) {
                    occlude = true;
                    break;
                }
            }
            if (!occlude) result.push_back(p);
        }

        Index::SimpleNeighbor *des_pool = cut_graph_ + (size_t)q * (size_t)range;
        for (size_t t = 0; t < result.size(); t++) {
            des_pool[t].id = result[t].id;
            des_pool[t].distance = result[t].distance;
        }
        if (result.size() < range) {
            des_pool[result.size()].distance = -1;
        }
    }

    void ComponentPruneNSSG::PruneInner(unsigned q, unsigned range, boost::dynamic_bitset<> flags,
                                        std::vector<Index::Neighbor> &pool, Index::SimpleNeighbor *cut_graph_, unsigned level) {
        index->width = range;
        unsigned start = 0;

        for (unsigned i = 0; i < pool.size(); ++i) {
            flags[pool[i].id] = 1;
        }
        for (unsigned nn = 0; nn < index->getFinalGraph()[q][0].size(); nn++) {
            unsigned id = index->getFinalGraph()[q][0][nn];
            if (flags[id]) continue;
            float dist = index->getDist()->compare(index->getBaseData() + index->getBaseDim() * (size_t)q,
                                            index->getBaseData() + index->getBaseDim() * (size_t)id,
                                            (unsigned)index->getBaseDim());
            pool.push_back(Index::Neighbor(id, dist, true));
        }

        std::sort(pool.begin(), pool.end());
        std::vector<Index::Neighbor> result;
        if (pool[start].id == q) start++;
        result.push_back(pool[start]);

        double kPi = std::acos(-1);
        float threshold = std::cos((float)index->A / 180 * kPi);

        while (result.size() < range && (++start) < pool.size()) {
            auto &p = pool[start];
            bool occlude = false;
            for (unsigned t = 0; t < result.size(); t++) {
                if (p.id == result[t].id) {
                    occlude = true;
                    break;
                }
                float djk = index->getDist()->compare(index->getBaseData() + index->getBaseDim() * (size_t)result[t].id,
                                               index->getBaseData() + index->getBaseDim() * (size_t)p.id,
                                               (unsigned)index->getBaseDim());
                float cos_ij = (p.distance + result[t].distance - djk) / 2 /
                               sqrt(p.distance * result[t].distance);

                if (cos_ij > threshold) {
                    occlude = true;
                    break;
                }
            }
            if (!occlude) result.push_back(p);
        }

        Index::SimpleNeighbor *des_pool = cut_graph_ + (size_t)q * (size_t)range;
        for (size_t t = 0; t < result.size(); t++) {
            des_pool[t].id = result[t].id;
            des_pool[t].distance = result[t].distance;
        }
        if (result.size() < range) {
            des_pool[result.size()].distance = -1;
        }
    }

    void ComponentPruneDPG::PruneInner(unsigned int q, unsigned int range, boost::dynamic_bitset<> flags,
                                       std::vector<Index::Neighbor> &pool, Index::SimpleNeighbor *cut_graph_, unsigned int level) {

        int len = pool.size();
        if(len > 2 * range) {
            len = 2 * range;
        }

        std::vector<int> hit(len, 0);

//        std::sort(pool.begin(), pool.end());
//        unsigned start = 0;
//        if (pool[start].id == q) start++;
//        result.push_back(pool[start]);

        for(int i = 0; i < len - 1; i ++){
            unsigned aid = pool[i].id;
            for(int j = i + 1; j < len; j ++){
                if (i == j)
                    continue;

                unsigned bid = pool[j].id;

                float dist =
                        index->getDist()->compare(index->getBaseData() + index->getBaseDim() * (size_t)aid,
                                                   index->getBaseData() + index->getBaseDim() * (size_t)bid, (unsigned)index->getBaseDim());
                if(dist < pool[j].distance){
                    hit[j] ++;
                }
            }
        }

        std::vector<int> tmp_hit;

        for(const auto &i : hit)
            tmp_hit.push_back(i);

        std::sort(tmp_hit.begin(), tmp_hit.end());

        int cut = tmp_hit[range];

        std::vector<Index::Neighbor> result;

        for(int i = 0; i < len; i ++){
            if(hit[i] <= cut)
                result.push_back(pool[i]);
        }

        result.resize(range);

        Index::SimpleNeighbor *des_pool = cut_graph_ + (size_t)q * (size_t)range;
        //std::cout << "prune : " << result.size() << "len : " << len << std::endl;
        for (size_t t = 0; t < result.size(); t++) {
            des_pool[t].id = result[t].id;
            des_pool[t].distance = result[t].distance;
        }
        if (result.size() < range) {
            des_pool[result.size()].distance = -1;
        }
    }

    void ComponentPruneHeuristic::PruneInner(unsigned int q, unsigned int range, boost::dynamic_bitset<> flags,
                                             std::vector<Index::Neighbor> &pool, Index::SimpleNeighbor *cut_graph_, unsigned int level) {

        std::vector<Index::Neighbor> picked;
        if(pool.size() > range){
            //std::cout << "wtf" << std::endl;
            std::sort(pool.begin(), pool.end());
            Index::MinHeap<float, Index::Neighbor> skipped;

            for(int i = pool.size() - 1; i >= 0; --i){
                //std::cout << "wtf" << i << std::endl;
                bool skip = false;
                float cur_dist = pool[i].distance;
                for(size_t j = 0; j < picked.size(); j ++){
                    float dist = index->getDist()->compare(index->getBaseData() + index->getBaseDim() * (size_t)picked[j].id,
                                                           index->getBaseData() + index->getBaseDim() * (size_t)pool[i].id, (unsigned)index->getBaseDim());
                    if(dist < cur_dist) {
                        skip = true;
                        break;
                    }
                }

                if(!skip){
                    picked.push_back(pool[i]);
                }else {
                    // save_remains  ??
                    skipped.push(cur_dist, pool[i]);
                }

                if(picked.size() == range)
                    break;
            }

            while (picked.size() < range && skipped.size()) {
                picked.push_back(skipped.top().data);
                skipped.pop();
            }
            //std::cout << "range picked : " << picked.size() << std::endl;
        }else{
            picked = pool;
        }
        //std::cout << "pool " << pool.size() << std::endl;
        Index::SimpleNeighbor *des_pool = cut_graph_ + (size_t)q * (size_t)range;
        //std::cout << "pool " << pool.size() << std::endl;
        for (size_t t = 0; t < picked.size(); t++) {
            des_pool[t].id = picked[t].id;
            des_pool[t].distance = picked[t].distance;
        }
        //std::cout << "pool " << pool.size() << std::endl;
        if (picked.size() < range) {
            des_pool[picked.size()].distance = -1;
        }
        //std::cout << "pool " << pool.size() << std::endl;
    }

    void ComponentPruneVAMANA::PruneInner(unsigned int q, unsigned int range, boost::dynamic_bitset<> flags,
                                          std::vector<Index::Neighbor> &pool, Index::SimpleNeighbor *cut_graph_,
                                          unsigned int level) {
        if (pool.size() <= range) return;

        std::sort(pool.begin(), pool.end());

        std::vector<Index::Neighbor> picked;

        if(pool.size() > range){
            Index::MinHeap<float, Index::Neighbor> skipped;

            for(size_t i = pool.size() - 1; i >= 0; --i){
                bool skip = false;
                float cur_dist = pool[i].distance;
                for(size_t j = 0; j < picked.size(); j ++){
                    float dist = index->getDist()->compare(index->getBaseData() + index->getBaseDim() * (size_t)picked[j].id,
                                                            index->getBaseData() + index->getBaseDim() * (size_t) pool[i].id, (unsigned)index->getBaseDim());
                    if(index->alpha * dist < cur_dist) {
                        skip = true;
                        break;
                    }
                }

                if(!skip){
                    picked.push_back(pool[i]);
                }else {
                    // save_remains  ??
                    skipped.push(cur_dist, pool[i]);
                }

                if(picked.size() == range)
                    break;
            }

            while (picked.size() < range && skipped.size()) {
                picked.push_back(skipped.top().data);
                skipped.pop();
            }
        }else{
            picked = pool;
        }

        Index::SimpleNeighbor *des_pool = cut_graph_ + (size_t)q * (size_t)range;
        for (size_t t = 0; t < picked.size(); t++) {
            des_pool[t].id = picked[t].id;
            des_pool[t].distance = picked[t].distance;
        }
        if (picked.size() < range) {
            des_pool[picked.size()].distance = -1;
        }
    }
}
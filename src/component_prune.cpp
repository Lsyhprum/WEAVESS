//
// Created by MurphySL on 2020/10/23.
//

#include "weavess/component.h"

namespace weavess {
    void ComponentPruneNaive::PruneInner(unsigned query, unsigned range, boost::dynamic_bitset<> flags,
                                         std::vector<Index::SimpleNeighbor> &pool, Index::SimpleNeighbor *cut_graph_) {
        Index::SimpleNeighbor *des_pool = cut_graph_ + (size_t) query * (size_t) range;
        for (size_t t = 0; t < (pool.size() > range ? range : pool.size()); t++) {
            des_pool[t].id = pool[t].id;
            des_pool[t].distance = pool[t].distance;
        }
        if (pool.size() < range) {
            des_pool[pool.size()].distance = -1;
        }
    }

    void ComponentPruneNSG::PruneInner(unsigned query, unsigned range, boost::dynamic_bitset<> flags,
                                       std::vector<Index::SimpleNeighbor> &pool, Index::SimpleNeighbor *cut_graph_) {
        unsigned maxc = index->C_refine;

        unsigned start = 0;

        for (unsigned nn = 0; nn < index->getFinalGraph()[query].size(); nn++) {
            unsigned id = index->getFinalGraph()[query][nn].id;
            if (flags[id]) continue;
            float dist =
                    index->getDist()->compare(index->getBaseData() + index->getBaseDim() * (size_t) query,
                                              index->getBaseData() + index->getBaseDim() * (size_t) id,
                                              (unsigned) index->getBaseDim());
            pool.push_back(Index::SimpleNeighbor(id, dist));
        }

        std::sort(pool.begin(), pool.end());
        std::vector<Index::SimpleNeighbor> result;
        if (pool[start].id == query) start++;
        result.push_back(pool[start]);

        while (result.size() < range && (++start) < pool.size() && start < maxc) {
            auto &p = pool[start];
            bool occlude = false;
            for (unsigned t = 0; t < result.size(); t++) {
                if (p.id == result[t].id) {
                    occlude = true;
                    break;
                }
                float djk = index->getDist()->compare(
                        index->getBaseData() + index->getBaseDim() * (size_t) result[t].id,
                        index->getBaseData() + index->getBaseDim() * (size_t) p.id,
                        (unsigned) index->getBaseDim());
                if (djk < p.distance /* dik */) {
                    occlude = true;
                    break;
                }
            }
            if (!occlude) result.push_back(p);
        }

        Index::SimpleNeighbor *des_pool = cut_graph_ + (size_t) query * (size_t) range;
        for (size_t t = 0; t < result.size(); t++) {
            des_pool[t].id = result[t].id;
            des_pool[t].distance = result[t].distance;
        }
        if (result.size() < range) {
            des_pool[result.size()].distance = -1;
        }
    }

    void ComponentPruneSSG::PruneInner(unsigned query, unsigned range, boost::dynamic_bitset<> flags,
                                        std::vector<Index::SimpleNeighbor> &pool, Index::SimpleNeighbor *cut_graph_) {
        unsigned start = 0;

        for (unsigned nn = 0; nn < index->getFinalGraph()[query].size(); nn++) {
            unsigned id = index->getFinalGraph()[query][nn].id;
            if (flags[id]) continue;
            float dist = index->getDist()->compare(index->getBaseData() + index->getBaseDim() * (size_t)query,
                                                   index->getBaseData() + index->getBaseDim() * (size_t)id,
                                                   (unsigned)index->getBaseDim());
            pool.push_back(Index::SimpleNeighbor(id, dist));
        }

        std::sort(pool.begin(), pool.end());
        std::vector<Index::SimpleNeighbor> result;
        if (pool[start].id == query) start++;
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

        Index::SimpleNeighbor *des_pool = cut_graph_ + (size_t)query * (size_t)range;
        for (size_t t = 0; t < result.size(); t++) {
            des_pool[t].id = result[t].id;
            des_pool[t].distance = result[t].distance;
        }
        if (result.size() < range) {
            des_pool[result.size()].distance = -1;
        }
    }

    void ComponentPruneDPG::PruneInner(unsigned query, unsigned int range, boost::dynamic_bitset<> flags,
                                       std::vector<Index::SimpleNeighbor> &pool, Index::SimpleNeighbor *cut_graph_) {

        int len = pool.size();
        if(len > 2 * range)
            len = 2 * range;

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
                                                  index->getBaseData() + index->getBaseDim() * (size_t)bid,
                                                  index->getBaseDim());
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

        std::vector<Index::SimpleNeighbor> result;

        for(int i = 0; i < len; i ++){
            if(hit[i] <= cut)
                result.push_back(pool[i]);
        }

        result.resize(range);

        Index::SimpleNeighbor *des_pool = cut_graph_ + (size_t)query * (size_t)range;
        //std::cout << "prune : " << result.size() << "len : " << len << std::endl;
        for (size_t t = 0; t < result.size(); t++) {
            des_pool[t].id = result[t].id;
            des_pool[t].distance = result[t].distance;
        }
        if (result.size() < range) {
            des_pool[result.size()].distance = -1;
        }

        std::vector<int>().swap(tmp_hit);
        std::vector<int>().swap(hit);
    }

    void ComponentPruneHeuristic::PruneInner(unsigned query, unsigned int range, boost::dynamic_bitset<> flags,
                                             std::vector<Index::SimpleNeighbor> &pool, Index::SimpleNeighbor *cut_graph_) {

        std::vector<Index::SimpleNeighbor> picked;
        if(pool.size() > range){
//            std::sort(pool.begin(), pool.end());
            Index::MinHeap<float, Index::SimpleNeighbor> skipped;

            for(int i = 0; i < pool.size(); i++){
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
            for(int i = 0; i < pool.size(); i++) {
                picked.push_back(pool[i]);
            }
        }
        Index::SimpleNeighbor *des_pool = cut_graph_ + (size_t)query * (size_t)range;
        //std::cout << "pick : " << picked.size() << std::endl;
        for (size_t t = 0; t < picked.size(); t++) {
            des_pool[t].id = picked[t].id;
            des_pool[t].distance = picked[t].distance;
            //std::cout << picked[t].id << "|" << picked[t].distance << " ";
        }
        //std::cout << std::endl;

        if (picked.size() < range) {
            des_pool[picked.size()].distance = -1;
        }

        std::vector<Index::SimpleNeighbor>().swap(picked);
    }

    void ComponentPruneVAMANA::PruneInner(unsigned query, unsigned int range, boost::dynamic_bitset<> flags,
                                          std::vector<Index::SimpleNeighbor> &pool, Index::SimpleNeighbor *cut_graph_) {
        std::vector<Index::SimpleNeighbor> picked;
        if(pool.size() > range){
            std::sort(pool.begin(), pool.end());

            Index::MinHeap<float, Index::SimpleNeighbor> skipped;

            for(int i = 0; i < pool.size(); i ++){
                bool skip = false;
                float cur_dist = pool[i].distance;
                for(size_t j = 0; j < picked.size(); j ++){
                    float dist = index->getDist()->compare(index->getBaseData() + index->getBaseDim() * (size_t)picked[j].id,
                                                           index->getBaseData() + index->getBaseDim() * (size_t)pool[i].id, (unsigned)index->getBaseDim());
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
            //std::cout << "range picked : " << picked.size() << std::endl;
        }else{
            for(int i = 0; i < pool.size(); i++) {
                picked.push_back(pool[i]);
            }
        }
        Index::SimpleNeighbor *des_pool = cut_graph_ + (size_t)query * (size_t)range;
        //std::cout << "pick : " << picked.size() << std::endl;
        for (size_t t = 0; t < picked.size(); t++) {
            des_pool[t].id = picked[t].id;
            des_pool[t].distance = picked[t].distance;
            //std::cout << picked[t].id << "|" << picked[t].distance << " ";
        }
        //std::cout << std::endl;

        if (picked.size() < range) {
            des_pool[picked.size()].distance = -1;
        }

        std::vector<Index::SimpleNeighbor>().swap(picked);
    }

    void ComponentPruneRNG::PruneInner(unsigned int query, unsigned int range, boost::dynamic_bitset<> flags,
                                       std::vector<Index::SimpleNeighbor> &pool, Index::SimpleNeighbor *cut_graph_) {
        unsigned count = 0;

        Index::SimpleNeighbor *des_pool = cut_graph_ + (size_t) query * (size_t) range;

        for(int j = 0; j < index->getFinalGraph()[query].size() && j < range; j ++) {
            des_pool[j].id = index->getFinalGraph()[query][j].id;
            des_pool[j].distance = index->getFinalGraph()[query][j].distance;
        }

        for(int j = 0; j < pool.size() && count < range; j ++) {
            const Index::SimpleNeighbor item = pool[j];
            if(item.id < 0) break;
            if(item.id == query) continue;

            bool good = true;
            for(unsigned k = 0; k < count; k ++) {
                float dist = index->getDist()->compare(index->getBaseData() + index->getBaseDim() * (index->getFinalGraph()[query][k]).id,
                                                       index->getBaseData() + index->getBaseDim() * item.id,
                                                       index->getBaseDim());
                if(dist <= item.distance) {
                    good = false;
                    break;
                }
            }
            if(good) {
                des_pool[count].id = item.id;
                des_pool[count].distance = item.distance;
                count ++;
            }
        }
        for(unsigned j = count; j < range; j ++) {
            des_pool[j].distance = -1;
        }
    }
}

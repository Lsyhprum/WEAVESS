//
// Created by Murph on 2020/8/25.
//
#include "weavess/index_builder.h"
#include <weavess/min_heap.h>

namespace weavess {
    // NSG & NSSG
    void IndexComponentPruneNSG::PruneInner() {
        auto range = index_->param_.get<unsigned>("R_nsg");

        init_graph();
        auto *cut_graph_ = new SimpleNeighbor[index_->n_ * (size_t)range];
        Link(index_->param_, cut_graph_);
        index_->final_graph_.resize(index_->n_);

        for (size_t i = 0; i < index_->n_; i++) {
            SimpleNeighbor *pool = cut_graph_ + i * (size_t)range;
            unsigned pool_size = 0;
            for (unsigned j = 0; j < range; j++) {
                if (pool[j].distance == -1) break;
                pool_size = j;
            }
            pool_size++;
            index_->final_graph_[i].resize(pool_size);
            for (unsigned j = 0; j < pool_size; j++) {
                index_->final_graph_[i][j] = pool[j].id;
            }
        }
    }

    // NSG
    void IndexComponentPruneNSG::Link(const Parameters &parameters, SimpleNeighbor *cut_graph_) {
        /*
        std::cout << " graph link" << std::endl;
        unsigned progress=0;
        unsigned percent = 100;
        unsigned step_size = nd_/percent;
        std::mutex progress_lock;
        */
        unsigned range = parameters.get<unsigned>("R_nsg");
        std::vector<std::mutex> locks(index_->n_);

#pragma omp parallel
        {
            // unsigned cnt = 0;
            std::vector<Neighbor> pool, tmp;
            boost::dynamic_bitset<> flags{index_->n_, 0};
#pragma omp for schedule(dynamic, 100)
            for (unsigned n = 0; n < index_->n_; ++n) {
                pool.clear();
                tmp.clear();
                flags.reset();
                get_neighbors(index_->data_ + index_->dim_ * n, parameters, flags, tmp, pool);
                sync_prune(n, pool, parameters, flags, cut_graph_);
                /*
              cnt++;
              if(cnt % step_size == 0){
                LockGuard g(progress_lock);
                std::cout<<progress++ <<"/"<< percent << " completed" << std::endl;
                }
                */
            }
        }

#pragma omp for schedule(dynamic, 100)
        for (unsigned n = 0; n < index_->n_; ++n) {
            InterInsert(n, range, locks, cut_graph_);
        }
    }

    void IndexComponentPruneNSG::init_graph() {
        float *center = new float[index_->dim_];
        for (unsigned j = 0; j < index_->dim_; j++) center[j] = 0;
        for (unsigned i = 0; i < index_->n_; i++) {
            for (unsigned j = 0; j < index_->dim_; j++) {
                center[j] += index_->data_[i * index_->dim_ + j];
            }
        }
        for (unsigned j = 0; j < index_->dim_; j++) {
            center[j] /= index_->n_;
        }
        std::vector<Neighbor> tmp, pool;
        index_->ep_ = rand() % index_->n_;  // random initialize navigating point
        get_neighbors(center, index_->param_, tmp, pool);
        index_->ep_ = tmp[0].id;
    }

    void IndexComponentPruneNSG::get_neighbors(const float *query, const Parameters &parameter,
                                 std::vector<Neighbor> &retset,
                                 std::vector<Neighbor> &fullset) {
        unsigned L = parameter.get<unsigned>("L_nsg");

        retset.resize(L + 1);
        std::vector<unsigned> init_ids(L);
        // initializer_->Search(query, nullptr, L, parameter, init_ids.data());

        boost::dynamic_bitset<> flags{index_->n_, 0};
        L = 0;
        for (unsigned i = 0; i < init_ids.size() && i < index_->final_graph_[index_->ep_].size(); i++) {
            init_ids[i] = index_->final_graph_[index_->ep_][i];
            flags[init_ids[i]] = true;
            L++;
        }
        while (L < init_ids.size()) {
            unsigned id = rand() % index_->n_;
            if (flags[id]) continue;
            init_ids[L] = id;
            L++;
            flags[id] = true;
        }

        L = 0;
        for (unsigned i = 0; i < init_ids.size(); i++) {
            unsigned id = init_ids[i];
            if (id >= index_->n_) continue;
            // std::cout<<id<<std::endl;
            float dist = index_->distance_->compare(index_->data_ + index_->dim_ * (size_t)id, query,
                                            (unsigned)index_->dim_);
            retset[i] = Neighbor(id, dist, true);
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

                for (unsigned m = 0; m < index_->final_graph_[n].size(); ++m) {
                    unsigned id = index_->final_graph_[n][m];
                    if (flags[id]) continue;
                    flags[id] = 1;

                    float dist = index_->distance_->compare(query, index_->data_ + index_->dim_ * (size_t)id,
                                                    (unsigned)index_->dim_);
                    Neighbor nn(id, dist, true);
                    fullset.push_back(nn);
                    if (dist >= retset[L - 1].distance) continue;
                    int r = InsertIntoPool(retset.data(), L, nn);

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

    void IndexComponentPruneNSG::get_neighbors(const float *query, const Parameters &parameter,
                                 boost::dynamic_bitset<> &flags,
                                 std::vector<Neighbor> &retset,
                                 std::vector<Neighbor> &fullset) {
        auto L = parameter.get<unsigned>("L_nsg");

        retset.resize(L + 1);
        std::vector<unsigned> init_ids(L);
        // initializer_->Search(query, nullptr, L, parameter, init_ids.data());

        L = 0;
        for (unsigned i = 0; i < init_ids.size() && i < index_->final_graph_[index_->ep_].size(); i++) {
            init_ids[i] = index_->final_graph_[index_->ep_][i];
            flags[init_ids[i]] = true;
            L++;
        }
        while (L < init_ids.size()) {
            unsigned id = rand() % index_->n_;
            if (flags[id]) continue;
            init_ids[L] = id;
            L++;
            flags[id] = true;
        }

        L = 0;
        for (unsigned i = 0; i < init_ids.size(); i++) {
            unsigned id = init_ids[i];
            if (id >= index_->n_) continue;
            // std::cout<<id<<std::endl;
            float dist = index_->distance_->compare(index_->data_ + index_->dim_ * (size_t)id, query,
                                            (unsigned)index_->dim_);
            retset[i] = Neighbor(id, dist, true);
            fullset.push_back(retset[i]);
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

                for (unsigned m = 0; m < index_->final_graph_[n].size(); ++m) {
                    unsigned id = index_->final_graph_[n][m];
                    if (flags[id]) continue;
                    flags[id] = 1;

                    float dist = index_->distance_->compare(query, index_->data_ + index_->dim_ * (size_t)id,
                                                    (unsigned)index_->dim_);
                    Neighbor nn(id, dist, true);
                    fullset.push_back(nn);
                    if (dist >= retset[L - 1].distance) continue;
                    int r = InsertIntoPool(retset.data(), L, nn);

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

    void IndexComponentPruneNSG::sync_prune(unsigned q, std::vector<Neighbor> &pool,
                              const Parameters &parameter,
                              boost::dynamic_bitset<> &flags,
                              SimpleNeighbor *cut_graph_) {
        unsigned range = parameter.get<unsigned>("R_nsg");
        unsigned maxc = parameter.get<unsigned>("C_nsg");
        index_->width = range;
        unsigned start = 0;

        for (unsigned nn = 0; nn < index_->final_graph_[q].size(); nn++) {
            unsigned id = index_->final_graph_[q][nn];
            if (flags[id]) continue;
            float dist =
                    index_->distance_->compare(index_->data_ + index_->dim_ * (size_t)q,
                                       index_->data_ + index_->dim_ * (size_t)id, (unsigned)index_->dim_);
            pool.push_back(Neighbor(id, dist, true));
        }

        std::sort(pool.begin(), pool.end());
        std::vector<Neighbor> result;
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
                float djk = index_->distance_->compare(index_->data_ + index_->dim_ * (size_t)result[t].id,
                                               index_->data_ + index_->dim_ * (size_t)p.id,
                                               (unsigned)index_->dim_);
                if (djk < p.distance /* dik */) {
                    occlude = true;
                    break;
                }
            }
            if (!occlude) result.push_back(p);
        }

        SimpleNeighbor *des_pool = cut_graph_ + (size_t)q * (size_t)range;
        for (size_t t = 0; t < result.size(); t++) {
            des_pool[t].id = result[t].id;
            des_pool[t].distance = result[t].distance;
        }
        if (result.size() < range) {
            des_pool[result.size()].distance = -1;
        }
    }

    void IndexComponentPruneNSG::InterInsert(unsigned n, unsigned range,
                               std::vector<std::mutex> &locks,
                               SimpleNeighbor *cut_graph_) {
        SimpleNeighbor *src_pool = cut_graph_ + (size_t)n * (size_t)range;
        for (size_t i = 0; i < range; i++) {
            if (src_pool[i].distance == -1) break;

            SimpleNeighbor sn(n, src_pool[i].distance);
            size_t des = src_pool[i].id;
            SimpleNeighbor *des_pool = cut_graph_ + des * (size_t)range;

            std::vector<SimpleNeighbor> temp_pool;
            int dup = 0;
            {
                LockGuard guard(locks[des]);
                for (size_t j = 0; j < range; j++) {
                    if (des_pool[j].distance == -1) break;
                    if (n == des_pool[j].id) {
                        dup = 1;
                        break;
                    }
                    temp_pool.push_back(des_pool[j]);
                }
            }
            if (dup) continue;

            temp_pool.push_back(sn);
            if (temp_pool.size() > range) {
                std::vector<SimpleNeighbor> result;
                unsigned start = 0;
                std::sort(temp_pool.begin(), temp_pool.end());
                result.push_back(temp_pool[start]);
                while (result.size() < range && (++start) < temp_pool.size()) {
                    auto &p = temp_pool[start];
                    bool occlude = false;
                    for (unsigned t = 0; t < result.size(); t++) {
                        if (p.id == result[t].id) {
                            occlude = true;
                            break;
                        }
                        float djk = index_->distance_->compare(index_->data_ + index_->dim_ * (size_t)result[t].id,
                                                       index_->data_ + index_->dim_ * (size_t)p.id,
                                                       (unsigned)index_->dim_);
                        if (djk < p.distance /* dik */) {
                            occlude = true;
                            break;
                        }
                    }
                    if (!occlude) result.push_back(p);
                }
                {
                    LockGuard guard(locks[des]);
                    for (unsigned t = 0; t < result.size(); t++) {
                        des_pool[t] = result[t];
                    }
                }
            } else {
                LockGuard guard(locks[des]);
                for (unsigned t = 0; t < range; t++) {
                    if (des_pool[t].distance == -1) {
                        des_pool[t] = sn;
                        if (t + 1 < range) des_pool[t + 1].distance = -1;
                        break;
                    }
                }
            }
        }
    }


    // NSSG
    void IndexComponentPruneNSSG::Link(const Parameters &parameters, SimpleNeighbor *cut_graph_) {
        unsigned range = parameters.get<unsigned>("R_nsg");
        std::vector<std::mutex> locks(index_->n_);

        float angle = parameters.get<float>("A");

        double kPi = std::acos(-1);

        float threshold = std::cos(angle / 180 * kPi);

#pragma omp parallel
        {
            // unsigned cnt = 0;
            std::vector<Neighbor> pool, tmp;
#pragma omp for schedule(dynamic, 100)
            for (unsigned n = 0; n < index_->n_; ++n) {
                pool.clear();
                tmp.clear();
                get_neighbors(n, parameters, pool);
                sync_prune(n, pool, parameters, threshold, cut_graph_);
                /*
                cnt++;
                if (cnt % step_size == 0) {
                  LockGuard g(progress_lock);
                  std::cout << progress++ << "/" << percent << " completed" << std::endl;
                }
                */
            }

#pragma omp for schedule(dynamic, 100)
            for (unsigned n = 0; n < index_->n_; ++n) {
                InterInsert(n, range, threshold, locks, cut_graph_);
            }
        }
    }

    void IndexComponentPruneNSSG::InterInsert(unsigned n, unsigned range, float threshold,
                               std::vector<std::mutex> &locks,
                               SimpleNeighbor *cut_graph_) {
        SimpleNeighbor *src_pool = cut_graph_ + (size_t)n * (size_t)range;
        for (size_t i = 0; i < range; i++) {
            if (src_pool[i].distance == -1) break;

            SimpleNeighbor sn(n, src_pool[i].distance);
            size_t des = src_pool[i].id;
            SimpleNeighbor *des_pool = cut_graph_ + des * (size_t)range;

            std::vector<SimpleNeighbor> temp_pool;
            int dup = 0;
            {
                LockGuard guard(locks[des]);
                for (size_t j = 0; j < range; j++) {
                    if (des_pool[j].distance == -1) break;
                    if (n == des_pool[j].id) {
                        dup = 1;
                        break;
                    }
                    temp_pool.push_back(des_pool[j]);
                }
            }
            if (dup) continue;

            temp_pool.push_back(sn);
            if (temp_pool.size() > range) {
                std::vector<SimpleNeighbor> result;
                unsigned start = 0;
                std::sort(temp_pool.begin(), temp_pool.end());
                result.push_back(temp_pool[start]);
                while (result.size() < range && (++start) < temp_pool.size()) {
                    auto &p = temp_pool[start];
                    bool occlude = false;
                    for (unsigned t = 0; t < result.size(); t++) {
                        if (p.id == result[t].id) {
                            occlude = true;
                            break;
                        }
                        float djk = index_->distance_->compare(
                                index_->data_ + index_->dim_ * (size_t)result[t].id,
                                index_->data_ + index_->dim_ * (size_t)p.id, (unsigned)index_->dim_);
                        float cos_ij = (p.distance + result[t].distance - djk) / 2 /
                                       sqrt(p.distance * result[t].distance);
                        if (cos_ij > threshold) {
                            occlude = true;
                            break;
                        }
                    }
                    if (!occlude) result.push_back(p);
                }
                {
                    LockGuard guard(locks[des]);
                    for (unsigned t = 0; t < result.size(); t++) {
                        des_pool[t] = result[t];
                    }
                    if (result.size() < range) {
                        des_pool[result.size()].distance = -1;
                    }
                }
            } else {
                LockGuard guard(locks[des]);
                for (unsigned t = 0; t < range; t++) {
                    if (des_pool[t].distance == -1) {
                        des_pool[t] = sn;
                        if (t + 1 < range) des_pool[t + 1].distance = -1;
                        break;
                    }
                }
            }
        }
    }

    void IndexComponentPruneNSSG::sync_prune(unsigned q, std::vector<Neighbor> &pool,
                              const Parameters &parameters, float threshold,
                              SimpleNeighbor *cut_graph_) {
        unsigned range = parameters.get<unsigned>("R_nsg");
        index_->width = range;
        unsigned start = 0;

        // 重复 ？
        boost::dynamic_bitset<> flags{index_->n_, 0};
        for (unsigned i = 0; i < pool.size(); ++i) {
            flags[pool[i].id] = 1;
        }
        for (unsigned nn = 0; nn < index_->final_graph_[q].size(); nn++) {
            unsigned id = index_->final_graph_[q][nn];
            if (flags[id]) continue;
            float dist = index_->distance_->compare(index_->data_ + index_->dim_ * (size_t)q,
                                            index_->data_ + index_->dim_ * (size_t)id,
                                            (unsigned)index_->dim_);
            pool.push_back(Neighbor(id, dist, true));
        }

        std::sort(pool.begin(), pool.end());
        std::vector<Neighbor> result;
        if (pool[start].id == q) start++;
        result.push_back(pool[start]);

        while (result.size() < range && (++start) < pool.size()) {
            auto &p = pool[start];
            bool occlude = false;
            for (unsigned t = 0; t < result.size(); t++) {
                if (p.id == result[t].id) {
                    occlude = true;
                    break;
                }
                float djk = index_->distance_->compare(index_->data_ + index_->dim_ * (size_t)result[t].id,
                                               index_->data_ + index_->dim_ * (size_t)p.id,
                                               (unsigned)index_->dim_);
                float cos_ij = (p.distance + result[t].distance - djk) / 2 /
                               sqrt(p.distance * result[t].distance);
                if (cos_ij > threshold) {
                    occlude = true;
                    break;
                }
            }
            if (!occlude) result.push_back(p);
        }

        SimpleNeighbor *des_pool = cut_graph_ + (size_t)q * (size_t)range;
        for (size_t t = 0; t < result.size(); t++) {
            des_pool[t].id = result[t].id;
            des_pool[t].distance = result[t].distance;
        }
        if (result.size() < range) {
            des_pool[result.size()].distance = -1;
        }
    }

    void IndexComponentPruneNSSG::get_neighbors(const unsigned q, const Parameters &parameter,
                                 std::vector<Neighbor> &pool) {
        boost::dynamic_bitset<> flags{index_->n_, 0};
        unsigned L = index_->param_.get<unsigned>("L");
        flags[q] = true;
        for (unsigned i = 0; i < index_->final_graph_[q].size(); i++) {
            unsigned nid = index_->final_graph_[q][i];
            for (unsigned nn = 0; nn < index_->final_graph_[nid].size(); nn++) {
                unsigned nnid = index_->final_graph_[nid][nn];
                if (flags[nnid]) continue;
                flags[nnid] = true;
                float dist = index_->distance_->compare(index_->data_ + index_->dim_ * q,
                                                index_->data_ + index_->dim_ * nnid, index_->dim_);
                pool.push_back(Neighbor(nnid, dist, true));
                if (pool.size() >= L) break;
            }
            if (pool.size() >= L) break;
        }
    }


    // DPG
    void IndexComponentPruneDPG::PruneInner() {
        // 通过修改 K 确定 L_dpg
        //auto L = index_->param_.get<unsigned>("L_dpg"); // 待修改 —— 当参数传入
        auto K = index_->param_.get<unsigned>("K");

        unsigned edge_num = K / 2;

        auto *cut_graph_ = new SimpleNeighbor[index_->n_ * (size_t)edge_num];

        Link(index_->param_, cut_graph_);

        index_->final_graph_.resize(index_->n_);

        for (size_t i = 0; i < index_->n_; i++) {
            SimpleNeighbor *pool = cut_graph_ + i * (size_t)edge_num;
            unsigned pool_size = 0;
            for (unsigned j = 0; j < edge_num; j++) {
                if (pool[j].distance == -1) break;
                pool_size = j;
            }
            pool_size++;
            index_->final_graph_[i].resize(pool_size);
            for (unsigned j = 0; j < pool_size; j++) {
                index_->final_graph_[i][j] = pool[j].id;
            }
        }
    }

    void IndexComponentPruneDPG::Link(const Parameters &parameters, SimpleNeighbor *cut_graph_) {
        auto K = index_->param_.get<unsigned>("K");

        unsigned edge_num = K / 2;

        std::vector<std::mutex> locks(index_->n_);

#pragma omp parallel
        {
#pragma omp for schedule(dynamic, 100)
            for (unsigned n = 0; n < index_->n_; ++n) {
                sync_prune(n, cut_graph_);
                /*
              cnt++;
              if(cnt % step_size == 0){
                LockGuard g(progress_lock);
                std::cout<<progress++ <<"/"<< percent << " completed" << std::endl;
                }
                */
            }
        }

#pragma omp for schedule(dynamic, 100)
        for (unsigned n = 0; n < index_->n_; ++n) {
            InterInsert(n, edge_num, locks, cut_graph_);
        }
    }

    void IndexComponentPruneDPG::InterInsert(unsigned n, unsigned range, std::vector<std::mutex> &locks, SimpleNeighbor *cut_graph_) {
        SimpleNeighbor *src_pool = cut_graph_ + (size_t)n * (size_t)range;
        for (size_t i = 0; i < range; i++) {
            if (src_pool[i].distance == -1) break;

            SimpleNeighbor sn(n, src_pool[i].distance);
            size_t des = src_pool[i].id;
            SimpleNeighbor *des_pool = cut_graph_ + des * (size_t)range;

            std::vector<SimpleNeighbor> temp_pool;
            int dup = 0;
            {
                LockGuard guard(locks[des]);
                for (size_t j = 0; j < range; j++) {
                    if (des_pool[j].distance == -1) break;
                    if (n == des_pool[j].id) {
                        dup = 1;
                        break;
                    }
                    temp_pool.push_back(des_pool[j]);
                }
            }
            if (dup) continue;

            temp_pool.push_back(sn);
            if (temp_pool.size() > range) {
                std::vector<SimpleNeighbor> result;
                unsigned start = 0;
                std::sort(temp_pool.begin(), temp_pool.end());
                result.push_back(temp_pool[start]);
                while (result.size() < range && (++start) < temp_pool.size()) {
                    auto &p = temp_pool[start];
                    bool occlude = false;
                    for (unsigned t = 0; t < result.size(); t++) {
                        if (p.id == result[t].id) {
                            occlude = true;
                            break;
                        }
                        float djk = index_->distance_->compare(index_->data_ + index_->dim_ * (size_t)result[t].id,
                                                               index_->data_ + index_->dim_ * (size_t)p.id,
                                                               (unsigned)index_->dim_);
                        if (djk < p.distance /* dik */) {
                            occlude = true;
                            break;
                        }
                    }
                    if (!occlude) result.push_back(p);
                }
                {
                    LockGuard guard(locks[des]);
                    for (unsigned t = 0; t < result.size(); t++) {
                        des_pool[t] = result[t];
                    }
                }
            } else {
                LockGuard guard(locks[des]);
                for (unsigned t = 0; t < range; t++) {
                    if (des_pool[t].distance == -1) {
                        des_pool[t] = sn;
                        if (t + 1 < range) des_pool[t + 1].distance = -1;
                        break;
                    }
                }
            }
        }
    }

    void IndexComponentPruneDPG::sync_prune(unsigned q, SimpleNeighbor *cut_graph_) {
        const auto K = index_->param_.get<unsigned>("K");

        unsigned edge_num = K / 2;

        std::vector<Neighbor> pool;
        std::vector<int> hit;

        for(unsigned nn = 0; nn < index_->final_graph_[q].size(); nn ++){
            unsigned id = index_->final_graph_[q][nn];
            float dist =
                    index_->distance_->compare(index_->data_ + index_->dim_ * (size_t)q,
                                               index_->data_ + index_->dim_ * (size_t)id, (unsigned)index_->dim_);
            pool.emplace_back(id, dist, true);

            hit.push_back(0);
        }

//        std::sort(pool.begin(), pool.end());
//        unsigned start = 0;
        std::vector<Neighbor> result;
//        if (pool[start].id == q) start++;
//        result.push_back(pool[start]);

        for(int i = 0; i < pool.size() - 1; i ++){
            unsigned aid = pool[i].id;
            for(int j = i + 1; j < pool.size(); j ++){
                if (i == j)
                    continue;

                unsigned bid = pool[j].id;

                float dist =
                        index_->distance_->compare(index_->data_ + index_->dim_ * (size_t)aid,
                                                   index_->data_ + index_->dim_ * (size_t)bid, (unsigned)index_->dim_);
                if(dist < pool[j].distance){
                    hit[j] ++;
                }
            }
        }

        std::vector<int> tmp_hit;

        for(const auto &i : hit)
            tmp_hit.push_back(i);

        std::sort(tmp_hit.begin(), tmp_hit.end());

        int cut = tmp_hit[edge_num];

        for(int i = 0; i < pool.size(); i ++){
            if(hit[i] <= cut)
                result.push_back(pool[i]);
        }

        SimpleNeighbor *des_pool = cut_graph_ + (size_t)q * (size_t)edge_num;
        for (size_t t = 0; t < result.size(); t++) {
            des_pool[t].id = result[t].id;
            des_pool[t].distance = result[t].distance;
        }
        if (result.size() < edge_num) {
            des_pool[result.size()].distance = -1;
        }
    }



    // NN-Descent
    void IndexComponentRefineNNDescent::PruneInner() {
        assert(index_->final_graph_.size() == index_->n_);

        const auto L = index_->param_.get<unsigned>("L");
        const auto S = index_->param_.get<unsigned>("S");

        index_->graph_.reserve(index_->n_);
        std::mt19937 rng(rand());
        for (unsigned i = 0; i < index_->n_; i++) {
            index_->graph_.emplace_back(L, S, rng, (unsigned) index_->n_);
        }
#pragma omp parallel for
        for (unsigned i = 0; i < index_->n_; i++) {
            auto& ids = index_->final_graph_[i];
            std::sort(ids.begin(), ids.end());

            size_t K = ids.size();

            for (unsigned j = 0; j < K; j++) {
                unsigned id = ids[j];
                if (id == i || (j>0 &&id == ids[j-1]))continue;
                float dist = index_->distance_->compare(index_->data_ + i * index_->dim_, index_->data_ + id * index_->dim_, (unsigned) index_->dim_);
                index_->graph_[i].pool.emplace_back(id, dist, true);
            }
            std::make_heap(index_->graph_[i].pool.begin(), index_->graph_[i].pool.end());
            index_->graph_[i].pool.reserve(L);
            std::vector<unsigned>().swap(ids);
        }
        std::vector<std::vector<unsigned>>().swap(index_->final_graph_);

        auto *a = new IndexComponentCoarseNNDescent(index_);
        a->CoarseInner();
    }



    // VAMANA
    void IndexComponentRefineVAMANA::PruneInner() {

        std::vector<Neighbor> pool, retset;

        init_graph(pool);

        auto L = index_->param_.get<unsigned>("L");
        auto range = index_->param_.get<unsigned>("R_nsg");

        auto *cut_graph_ = new SimpleNeighbor[index_->n_ * (size_t)range];

        for(unsigned j = 0; j < 2; j ++)
            for(unsigned i = 0; i < index_->n_; i ++){
                if(i == index_->ep_) continue;

                // insert To pool 待修改 —— 裁边策略
                retset = pool;
                greedySearch(i, pool, retset);

                float alpha = 1;
                if(j == 1)
                    alpha = index_->param_.get<float>("alpha");

                boost::dynamic_bitset<> flags{index_->n_, 0};

                sync_prune(i, alpha, retset, flags, cut_graph_);

                // 添加到 final_graph
                index_->final_graph_.resize(index_->n_);

                SimpleNeighbor *pool = cut_graph_ + i * (size_t)range;
                unsigned pool_size = 0;
                for (unsigned j = 0; j < range; j++) {
                    if (pool[j].distance == -1) break;
                    pool_size = j;
                }
                pool_size++;
                index_->final_graph_[i].resize(pool_size);
                for (unsigned j = 0; j < pool_size; j++) {
                    index_->final_graph_[i][j] = pool[j].id;
                }


                // 添加反向边
                for(int n = 0; n < retset.size(); n ++){
                    std::vector<Neighbor> reverse_pool;
                    unsigned nn_id = retset[n].id;
                    index_->final_graph_[nn_id].push_back(i);
                    boost::dynamic_bitset<> flags_tmp{index_->n_, 0};
                    sync_prune(nn_id, alpha, reverse_pool, flags_tmp, cut_graph_);

                    SimpleNeighbor *pool = cut_graph_ + i * (size_t)range;
                    unsigned pool_size = 0;
                    for (unsigned j = 0; j < range; j++) {
                        if (pool[j].distance == -1) break;
                        pool_size = j;
                    }
                    pool_size++;
                    index_->final_graph_[i].resize(pool_size);
                    for (unsigned j = 0; j < pool_size; j++) {
                        index_->final_graph_[i][j] = pool[j].id;
                    }
                }
            }
    }

    void IndexComponentRefineVAMANA::init_graph(std::vector<Neighbor> &pool) {
        auto *center = new float[index_->dim_];
        for (unsigned j = 0; j < index_->dim_; j++) center[j] = 0;
        for (unsigned i = 0; i < index_->n_; i++) {
            for (unsigned j = 0; j < index_->dim_; j++) {
                center[j] += index_->data_[i * index_->dim_ + j];
            }
        }
        for (unsigned j = 0; j < index_->dim_; j++) {
            center[j] /= index_->n_;
        }

        std::vector<Neighbor> tmp;
        index_->ep_ = rand() % index_->n_;  // random initialize navigating point
        get_neighbors(center, tmp, pool);
        index_->ep_ = tmp[0].id;
    }

    void IndexComponentRefineVAMANA::get_neighbors(const float *query, std::vector<Neighbor> &retset, std::vector<Neighbor> &fullset) {
        auto L = index_->param_.get<unsigned>("L");

        retset.clear();
        retset.resize(L + 1);
        std::vector<unsigned> init_ids(L);
        // initializer_->Search(query, nullptr, L, parameter, init_ids.data());

        boost::dynamic_bitset<> flags{index_->n_, 0};
        L = 0;
        for (unsigned i = 0; i < init_ids.size() && i < index_->final_graph_[index_->ep_].size(); i++) {
            init_ids[i] = index_->final_graph_[index_->ep_][i];
            flags[init_ids[i]] = true;
            L++;
        }
        while (L < init_ids.size()) {
            unsigned id = rand() % index_->n_;
            if (flags[id]) continue;
            init_ids[L] = id;
            L++;
            flags[id] = true;
        }

        L = 0;
        for (unsigned i = 0; i < init_ids.size(); i++) {
            unsigned id = init_ids[i];
            if (id >= index_->n_) continue;
            // std::cout<<id<<std::endl;
            float dist = index_->distance_->compare(index_->data_ + index_->dim_ * (size_t)id, query,
                                                    (unsigned)index_->dim_);
            retset[i] = Neighbor(id, dist, true);
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

                for (unsigned m = 0; m < index_->final_graph_[n].size(); ++m) {
                    unsigned id = index_->final_graph_[n][m];
                    if (flags[id]) continue;
                    flags[id] = 1;

                    float dist = index_->distance_->compare(query, index_->data_ + index_->dim_ * (size_t)id,
                                                            (unsigned)index_->dim_);
                    Neighbor nn(id, dist, true);
                    fullset.push_back(nn);
                    if (dist >= retset[L - 1].distance) continue;
                    int r = InsertIntoPool(retset.data(), L, nn);

                    if (L + 1 < retset.size()) ++L;
                    if (r < nk) nk = r;
                }
            }
            if (nk <= k)
                k = nk;
            else
                ++k;
        }

        std::sort(fullset.begin(), fullset.begin()+L);
    }

    void IndexComponentRefineVAMANA::greedySearch(const unsigned query_id, const std::vector<Neighbor> &pool, std::vector<Neighbor> &retset){
        const auto L = index_->param_.get<unsigned>("L");

        std::vector<char> flags(index_->n_);

        int k=0;
        while(k < (int)L) {
            int nk = L;

            if (retset[k].flag) {
                retset[k].flag = false;
                unsigned n = retset[k].id;

                for (unsigned m = 0; m < index_->final_graph_[n].size(); ++m) {
                    unsigned id = index_->final_graph_[n][m];
                    if (flags[id])continue;
                    flags[id] = 1;
                    float dist = index_->distance_->compare(index_->query_data_ + index_->query_dim_ * query_id,
                                                            index_->data_ + index_->dim_ * id, (unsigned) index_->dim_);
                    if (dist >= retset[L - 1].distance)continue;
                    Neighbor nn(id, dist, true);
                    int r = InsertIntoPool(retset.data(), L, nn);

                    //if(L+1 < retset.size()) ++L;
                    if (r < nk)nk = r;
                }
                //lock to here
            }
            if (nk <= k)k = nk;
            else ++k;
        }

    }

    void IndexComponentRefineVAMANA::sync_prune(unsigned q, float alpha, std::vector<Neighbor> &pool, boost::dynamic_bitset<> &flags,
                                            SimpleNeighbor *cut_graph_) {
        auto range = index_->param_.get<unsigned>("R");
        index_->width = range;
        unsigned start = 0;

        for (unsigned nn = 0; nn < index_->final_graph_[q].size(); nn++) {
            unsigned id = index_->final_graph_[q][nn];
            if (flags[id]) continue;
            float dist =
                    index_->distance_->compare(index_->data_ + index_->dim_ * (size_t)q,
                                               index_->data_ + index_->dim_ * (size_t)id, (unsigned)index_->dim_);
            pool.emplace_back(id, dist, true);
        }

        std::sort(pool.begin(), pool.end());

        std::vector<Neighbor> picked;

        if(pool.size() > range){
            MinHeap<float, Neighbor> skipped;

            for(size_t i = pool.size() - 1; i >= 0; --i){
                bool skip = false;
                float cur_dist = pool[i].distance;
                for(size_t j = 0; j < picked.size(); j ++){
                    float dist = index_->distance_->compare(index_->data_ + index_->dim_ * (size_t)picked[j].id,
                                                            index_->data_ + index_->dim_ * (size_t) pool[i].id, (unsigned)index_->dim_);
                    if(alpha * dist < cur_dist) {
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
        }else{
            picked = pool;
        }


        SimpleNeighbor *des_pool = cut_graph_ + (size_t)q * (size_t)range;
        for (size_t t = 0; t < picked.size(); t++) {
            des_pool[t].id = picked[t].id;
            des_pool[t].distance = picked[t].distance;
        }
        if (picked.size() < range) {
            des_pool[picked.size()].distance = -1;
        }
    }


}


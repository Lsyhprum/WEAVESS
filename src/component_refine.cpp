//
// Created by Murph on 2020/9/14.
//

#include "weavess/component.h"

#define _CONTROL_NUM 100

namespace weavess {

    // NSG
    void ComponentRefineNSG::RefineInner() {

        SetConfigs();

        // ENTRY
        std::cout << "__ENTRY : Centroid__" << std::endl;
        auto *a = new ComponentEntryCentroidNSG(index);
        a->EntryInner();
        std::cout << "__ENTRY : FINISH" << std::endl;

        auto *cut_graph_ = new Index::SimpleNeighbor[index->getBaseLen() * (size_t) index->R_nsg];
        Link(cut_graph_);

        index->getFinalGraph().resize(index->getBaseLen());

        for (size_t i = 0; i < index->getBaseLen(); i++) {
            Index::SimpleNeighbor *pool = cut_graph_ + i * (size_t) index->R_nsg;
            unsigned pool_size = 0;
            for (unsigned j = 0; j < index->R_nsg; j++) {
                if (pool[j].distance == -1) break;
                pool_size = j;
            }
            pool_size++;
            index->getFinalGraph()[i].resize(1);
            index->getFinalGraph()[i][0].resize(pool_size);
            for (unsigned j = 0; j < pool_size; j++) {
                index->getFinalGraph()[i][0][j] = pool[j].id;
            }
        }

        // CONN
        std::cout << "__CONN : DFS__" << std::endl;
        auto *c = new ComponentConnNSGDFS(index);

        c->ConnInner();
    }

    void ComponentRefineNSG::SetConfigs() {
        index->R_nsg = index->getParam().get<unsigned>("R_nsg");
        index->L_nsg = index->getParam().get<unsigned>("L_nsg");
        index->C_nsg = index->getParam().get<unsigned>("C_nsg");
    }

    void ComponentRefineNSG::Link(Index::SimpleNeighbor *cut_graph_) {
        std::vector<std::mutex> locks(index->getBaseLen());

        // CANDIDATE
        std::cout << "__CANDIDATE : GREEDY(NSG)__" << std::endl;
        auto *a = new ComponentCandidateNSG(index);

        // PRUNE
        std::cout << "__PRUNE : NSG__" << std::endl;
        auto *b = new ComponentPruneNSG(index);

#pragma omp parallel
        {
            std::vector<Index::Neighbor> pool;
            boost::dynamic_bitset<> flags{index->getBaseLen(), 0};

#pragma omp for schedule(dynamic, 100)
            for (unsigned n = 0; n < index->getBaseLen(); ++n){
                //std::cout << n << std::endl;
                pool.clear();
                flags.reset();

                a->CandidateInner(n, index->ep_, flags, pool, 0);
                //std::cout << n << " candidate : " << pool.size() << std::endl;
                b->PruneInner(n, index->R_nsg, flags, pool, cut_graph_, 0);
                //std::cout << n << " prune : " << pool.size() << std::endl;
            }
        }

#pragma omp for schedule(dynamic, 100)
        for (unsigned n = 0; n < index->getBaseLen(); ++n) {
            InterInsert(n, index->R_nsg, locks, cut_graph_);
        }
    }

    void ComponentRefineNSG::InterInsert(unsigned n, unsigned range, std::vector<std::mutex> &locks,
                                         Index::SimpleNeighbor *cut_graph_) {
        Index::SimpleNeighbor *src_pool = cut_graph_ + (size_t) n * (size_t) range;
        for (size_t i = 0; i < range; i++) {
            if (src_pool[i].distance == -1) break;

            Index::SimpleNeighbor sn(n, src_pool[i].distance);
            size_t des = src_pool[i].id;
            Index::SimpleNeighbor *des_pool = cut_graph_ + des * (size_t) range;

            std::vector<Index::SimpleNeighbor> temp_pool;
            int dup = 0;
            {
                Index::LockGuard guard(locks[des]);
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
                std::vector<Index::SimpleNeighbor> result;
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
                {
                    Index::LockGuard guard(locks[des]);
                    for (unsigned t = 0; t < result.size(); t++) {
                        des_pool[t] = result[t];
                    }
                }
            } else {
                Index::LockGuard guard(locks[des]);
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
    void ComponentRefineNSSG::RefineInner() {
        SetConfigs();

        // ENTRY
        auto *a = new ComponentEntryCentroidNSSG(index);
        a->EntryInner();

        auto *cut_graph_ = new Index::SimpleNeighbor[index->getBaseLen() * (size_t) index->R_nsg];
        Link(cut_graph_);

        index->getFinalGraph().resize(index->getBaseLen());

        for (size_t i = 0; i < index->getBaseLen(); i++) {
            Index::SimpleNeighbor *pool = cut_graph_ + i * (size_t) index->R_nsg;
            unsigned pool_size = 0;
            for (unsigned j = 0; j < index->R_nsg; j++) {
                if (pool[j].distance == -1) break;
                pool_size = j;
            }
            pool_size++;
            index->getFinalGraph()[i].resize(1);
            index->getFinalGraph()[i][0].resize(pool_size);
            for (unsigned j = 0; j < pool_size; j++) {
                index->getFinalGraph()[i][0][j] = pool[j].id;
            }
        }

        // CONN
        std::cout << "__CONN : DFS__" << std::endl;
        auto *c = new ComponentConnSSGDFS(index);

        c->ConnInner();
    }

    void ComponentRefineNSSG::SetConfigs() {
        index->R_nsg = index->getParam().get<unsigned>("R_nsg");
        index->L_nsg = index->getParam().get<unsigned>("L_nsg");
        index->A = index->getParam().get<float>("A");
        index->n_try = index->getParam().get<unsigned>("n_try");
    }

    void ComponentRefineNSSG::Link(Index::SimpleNeighbor *cut_graph_) {
        /*
         std::cerr << "Graph Link" << std::endl;
         unsigned progress = 0;
         unsigned percent = 100;
         unsigned step_size = nd_ / percent;
         std::mutex progress_lock;
         */
        std::vector<std::mutex> locks(index->getBaseLen());

        // CANDIDATE
        std::cout << "__CANDIDATE : PROPAGATION 2__" << std::endl;
        ComponentCandidate *a = new ComponentCandidateNSSG(index);

        // PRUNE
        std::cout << "__PRUNE : NSSG__" << std::endl;
        ComponentPrune *b = new ComponentPruneNSSG(index);

#pragma omp parallel
        {
            // unsigned cnt = 0;
            std::vector<Index::Neighbor> pool;
            boost::dynamic_bitset<> flags{index->getBaseLen(), 0};

#pragma omp for schedule(dynamic, 100)
            for (unsigned n = 0; n < index->getBaseLen(); ++n) {
                pool.clear();
                flags.reset();

                a->CandidateInner(n, 0, flags, pool, 0);
                //std::cout << "candidate : " << pool.size() << std::endl;

                flags.reset();

                b->PruneInner(n, index->R_nsg, flags, pool, cut_graph_, 0);
                //std::cout << "prune : " << pool.size() << std::endl;

                /*
                cnt++;
                if (cnt % step_size == 0) {
                  LockGuard g(progress_lock);
                  std::cout << progress++ << "/" << percent << " completed" << std::endl;
                }
                */
            }
        }

        double kPi = std::acos(-1);
        float threshold = std::cos(index->A / 180 * kPi);

#pragma omp for schedule(dynamic, 100)
        for (unsigned n = 0; n < index->getBaseLen(); ++n) {
            InterInsert(n, index->R_nsg, threshold, locks, cut_graph_);
        }
    }

    void ComponentRefineNSSG::InterInsert(unsigned n, unsigned range, float threshold, std::vector<std::mutex> &locks,
                                          Index::SimpleNeighbor *cut_graph_) {
        Index::SimpleNeighbor *src_pool = cut_graph_ + (size_t)n * (size_t)range;
        for (size_t i = 0; i < range; i++) {
            if (src_pool[i].distance == -1) break;

            Index::SimpleNeighbor sn(n, src_pool[i].distance);
            size_t des = src_pool[i].id;
            Index::SimpleNeighbor *des_pool = cut_graph_ + des * (size_t)range;

            std::vector<Index::SimpleNeighbor> temp_pool;
            int dup = 0;
            {
                Index::LockGuard guard(locks[des]);
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
                std::vector<Index::SimpleNeighbor> result;
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
                        float djk = index->getDist()->compare(index->getBaseData() + index->getBaseDim() * (size_t)result[t].id,
                                                              index->getBaseData() + index->getBaseDim() * (size_t)p.id, (unsigned)index->getBaseDim());
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
                    Index::LockGuard guard(locks[des]);
                    for (unsigned t = 0; t < result.size(); t++) {
                        des_pool[t] = result[t];
                    }
                    if (result.size() < range) {
                        des_pool[result.size()].distance = -1;
                    }
                }
            } else {
                Index::LockGuard guard(locks[des]);
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


    // DPG
    void ComponentRefineDPG::RefineInner() {
        SetConfigs();
        auto *cut_graph_ = new Index::SimpleNeighbor[index->getBaseLen() * (size_t) index->L_dpg];
        Link(cut_graph_);

        index->getFinalGraph().resize(index->getBaseLen());

        for (size_t i = 0; i < index->getBaseLen(); i++) {
            Index::SimpleNeighbor *pool = cut_graph_ + i * (size_t) index->L_dpg;
            unsigned pool_size = 0;
            for (unsigned j = 0; j < index->L_dpg; j++) {
                if (pool[j].distance == -1) break;
                pool_size = j;
            }
            pool_size++;
            index->getFinalGraph()[i].resize(1);
            index->getFinalGraph()[i][0].resize(pool_size);
            for (unsigned j = 0; j < pool_size; j++) {
                index->getFinalGraph()[i][0][j] = pool[j].id;
            }
        }

        // CONN
        std::cout << "__CONN : DFS__" << std::endl;
        auto *c = new ComponentConnReverse(index);
        c->ConnInner();
    }

    void ComponentRefineDPG::SetConfigs() {
        index->K = index->getParam().get<unsigned>("K");
        index->L = index->getParam().get<unsigned>("L");
        index->S = index->getParam().get<unsigned>("S");
        index->R = index->getParam().get<unsigned>("R");
        index->ITER = index->getParam().get<unsigned>("ITER");

        index->L_dpg = index->K / 2;
    }

    void ComponentRefineDPG::Link(Index::SimpleNeighbor *cut_graph_) {
        std::vector<std::mutex> locks(index->getBaseLen());

        // CANDIDATE
        ComponentCandidateNone *a = new ComponentCandidateNone(index);

        // PRUNE
        ComponentPrune *b = new ComponentPruneDPG(index);

#pragma omp parallel
        {
            boost::dynamic_bitset<> flags{index->getBaseLen(), 0};
            std::vector<Index::Neighbor> pool;
#pragma omp for schedule(dynamic, 100)
            for (unsigned n = 0; n < index->getBaseLen(); ++n) {
                //std::cout << n << std::endl;

                pool.clear();

                flags.reset();

                a->CandidateInner(n, 0, flags, pool, 0);
                //std::cout << "candidate : " << pool.size() << std::endl;
                b->PruneInner(n, index->L_dpg, flags, pool, cut_graph_, 0);
            }
        }
    }


    // EFANNA
    void ComponentRefineEFANNA::RefineInner() {
        SetConfigs();

        init();

        NNDescent();

        index->getFinalGraph().reserve(index->getBaseLen());

        unsigned K = index->K;
        for (unsigned i = 0; i < index->getBaseLen(); i++) {
            std::vector<unsigned> tmp;
            std::sort(index->graph_[i].pool.begin(), index->graph_[i].pool.end());
            for (unsigned j = 0; j < K; j++) {
                tmp.push_back(index->graph_[i].pool[j].id);
            }
            tmp.reserve(K);
            std::vector<std::vector<unsigned>> level_tmp;
            level_tmp.reserve(1);
            level_tmp.push_back(tmp);
            level_tmp.resize(1);
            index->getFinalGraph().push_back(level_tmp);
            std::vector<Index::Neighbor>().swap(index->graph_[i].pool);
            std::vector<unsigned>().swap(index->graph_[i].nn_new);
            std::vector<unsigned>().swap(index->graph_[i].nn_old);
            std::vector<unsigned>().swap(index->graph_[i].rnn_new);
            std::vector<unsigned>().swap(index->graph_[i].rnn_new);
        }
        std::vector<Index::nhood>().swap(index->graph_);
    }

    void ComponentRefineEFANNA::SetConfigs() {
        index->L = index->getParam().get<unsigned>("L");
        index->K = index->getParam().get<unsigned>("K");
        index->R = index->getParam().get<unsigned>("R");
        index->S = index->getParam().get<unsigned>("S");
        index->ITER = index->getParam().get<unsigned>("ITER");
    }

    void ComponentRefineEFANNA::init() {
        index->graph_.reserve(index->getBaseLen());
        std::mt19937 rng(rand());

        for (unsigned i = 0; i < index->getBaseLen(); i++) {
            index->graph_.push_back(Index::nhood(index->L, index->S, rng, (unsigned) index->getBaseLen()));
        }
#pragma omp parallel for
        for (unsigned i = 0; i < index->getBaseLen(); i++) {
            //std::cout << i << std::endl;
            auto &ids = index->getFinalGraph()[i][0];
            std::sort(ids.begin(), ids.end());

            size_t K = ids.size();

            for (unsigned j = 0; j < K; j++) {
                unsigned id = ids[j];
                if (id == i || (j > 0 && id == ids[j - 1]))continue;
                float dist = index->getDist()->compare(index->getBaseData() + i * index->getBaseDim(),
                                                       index->getBaseData() + id * index->getBaseDim(),
                                                       (unsigned) index->getBaseDim());
                index->graph_[i].pool.push_back(Index::Neighbor(id, dist, true));
            }
            std::make_heap(index->graph_[i].pool.begin(), index->graph_[i].pool.end());
            index->graph_[i].pool.reserve(index->L);
            std::vector<unsigned>().swap(ids);
        }
        Index::CompactGraph().swap(index->getFinalGraph());
    }

    void ComponentRefineEFANNA::NNDescent() {

        std::mt19937 rng(rand());

        // 采样用于评估每次迭代效果，与算法无关
        std::vector<unsigned> control_points(_CONTROL_NUM);
        std::vector<std::vector<unsigned> > acc_eval_set(_CONTROL_NUM);
        GenRandom(rng, &control_points[0], control_points.size(), index->getBaseLen());
        generate_control_set(control_points, acc_eval_set, index->getBaseLen());

        for (unsigned it = 0; it < index->ITER; it++) {
            join();
            update();
            //checkDup();
            eval_recall(control_points, acc_eval_set);
            std::cout << "iter: " << it << std::endl;
        }
    }

    void ComponentRefineEFANNA::join() {
#pragma omp parallel for default(shared) schedule(dynamic, 100)
        for (unsigned n = 0; n < index->getBaseLen(); n++) {
            index->graph_[n].join([&](unsigned i, unsigned j) {
                if (i != j) {
                    float dist = index->getDist()->compare(index->getBaseData() + i * index->getBaseDim(),
                                                           index->getBaseData() + j * index->getBaseDim(),
                                                           index->getBaseDim());

                    index->graph_[i].insert(j, dist);
                    index->graph_[j].insert(i, dist);
                }
            });
        }
    }

    void ComponentRefineEFANNA::update() {
        // 清空内存
#pragma omp parallel for
        for (unsigned i = 0; i < index->getBaseLen(); i++) {
            std::vector<unsigned>().swap(index->graph_[i].nn_new);
            std::vector<unsigned>().swap(index->graph_[i].nn_old);
            //std::vector<unsigned>().swap(graph_[i].rnn_new);
            //std::vector<unsigned>().swap(graph_[i].rnn_old);
            //graph_[i].nn_new.clear();
            //graph_[i].nn_old.clear();
            //graph_[i].rnn_new.clear();
            //graph_[i].rnn_old.clear();
        }
        // 确定候选个数
#pragma omp parallel for
        for (unsigned n = 0; n < index->getBaseLen(); ++n) {
            auto &nn = index->graph_[n];
            std::sort(nn.pool.begin(), nn.pool.end());
            if (nn.pool.size() > index->L)nn.pool.resize(index->L);
            nn.pool.reserve(index->L);
            unsigned maxl = std::min(nn.M + index->S, (unsigned) nn.pool.size());
            unsigned c = 0;
            unsigned l = 0;
            //std::sort(nn.pool.begin(), nn.pool.end());
            //if(n==0)std::cout << nn.pool[0].distance<<","<< nn.pool[1].distance<<","<< nn.pool[2].distance<< std::endl;
            while ((l < maxl) && (c < index->S)) {
                if (nn.pool[l].flag) ++c;
                ++l;
            }
            nn.M = l;
        }
#pragma omp parallel for
        for (unsigned n = 0; n < index->getBaseLen(); ++n) {
            auto &nnhd = index->graph_[n];
            auto &nn_new = nnhd.nn_new;
            auto &nn_old = nnhd.nn_old;
            for (unsigned l = 0; l < nnhd.M; ++l) {
                auto &nn = nnhd.pool[l];
                auto &nhood_o = index->graph_[nn.id];  // nn on the other side of the edge

                if (nn.flag) {
                    nn_new.push_back(nn.id);
                    if (nn.distance > nhood_o.pool.back().distance) {
                        Index::LockGuard guard(nhood_o.lock);
                        if (nhood_o.rnn_new.size() < index->R)nhood_o.rnn_new.push_back(n);
                        else {
                            unsigned int pos = rand() % index->R;
                            nhood_o.rnn_new[pos] = n;
                        }
                    }
                    nn.flag = false;
                } else {
                    nn_old.push_back(nn.id);
                    if (nn.distance > nhood_o.pool.back().distance) {
                        Index::LockGuard guard(nhood_o.lock);
                        if (nhood_o.rnn_old.size() < index->R)nhood_o.rnn_old.push_back(n);
                        else {
                            unsigned int pos = rand() % index->R;
                            nhood_o.rnn_old[pos] = n;
                        }
                    }
                }
            }
            std::make_heap(nnhd.pool.begin(), nnhd.pool.end());
        }
#pragma omp parallel for
        for (unsigned i = 0; i < index->getBaseLen(); ++i) {
            auto &nn_new = index->graph_[i].nn_new;
            auto &nn_old = index->graph_[i].nn_old;
            auto &rnn_new = index->graph_[i].rnn_new;
            auto &rnn_old = index->graph_[i].rnn_old;
            if (index->R && rnn_new.size() > index->R) {
                std::random_shuffle(rnn_new.begin(), rnn_new.end());
                rnn_new.resize(index->R);
            }
            nn_new.insert(nn_new.end(), rnn_new.begin(), rnn_new.end());
            if (index->R && rnn_old.size() > index->R) {
                std::random_shuffle(rnn_old.begin(), rnn_old.end());
                rnn_old.resize(index->R);
            }
            nn_old.insert(nn_old.end(), rnn_old.begin(), rnn_old.end());
            if (nn_old.size() > index->R * 2) {
                nn_old.resize(index->R * 2);
                nn_old.reserve(index->R * 2);
            }
            std::vector<unsigned>().swap(index->graph_[i].rnn_new);
            std::vector<unsigned>().swap(index->graph_[i].rnn_old);
        }
    }

    void ComponentRefineEFANNA::generate_control_set(std::vector<unsigned> &c, std::vector<std::vector<unsigned> > &v,
                                                     unsigned N) {
#pragma omp parallel for
        for (unsigned i = 0; i < c.size(); i++) {
            std::vector<Index::Neighbor> tmp;
            for (unsigned j = 0; j < N; j++) {
                float dist = index->getDist()->compare(index->getBaseData() + c[i] * index->getBaseDim(),
                                                       index->getBaseData() + j * index->getBaseDim(),
                                                       index->getBaseDim());
                tmp.push_back(Index::Neighbor(j, dist, true));
            }
            std::partial_sort(tmp.begin(), tmp.begin() + _CONTROL_NUM, tmp.end());
            for (unsigned j = 0; j < _CONTROL_NUM; j++) {
                v[i].push_back(tmp[j].id);
            }
        }
    }

    void ComponentRefineEFANNA::eval_recall(std::vector<unsigned> &ctrl_points,
                                            std::vector<std::vector<unsigned> > &acc_eval_set) {
        float mean_acc = 0;
        for (unsigned i = 0; i < ctrl_points.size(); i++) {
            float acc = 0;
            auto &g = index->graph_[ctrl_points[i]].pool;
            auto &v = acc_eval_set[i];
            for (unsigned j = 0; j < g.size(); j++) {
                for (unsigned k = 0; k < v.size(); k++) {
                    if (g[j].id == v[k]) {
                        acc++;
                        break;
                    }
                }
            }
            mean_acc += acc / v.size();
        }
        std::cout << "recall : " << mean_acc / ctrl_points.size() << std::endl;
    }


    // HNSW
    void ComponentRefineHNSW::RefineInner() {
        SetConfigs();

        Build(false);

//        if (post_graph_process_ == GraphPostProcessing::MERGE_LEVEL0) {
//            std::cout << "graph post processing: merge_level0" << std::endl;
//
//            std::vector<Index::HnswNode*> nodes_backup;
//            nodes_backup.swap(index->nodes_);
//            Build(true);
//            MergeEdgesOfTwoGraphs(nodes_backup);
//
//            for (size_t i = 0; i < nodes_backup.size(); ++i) {
//                delete nodes_backup[i];
//            }
//            nodes_backup.clear();
//        }

        for (size_t i = 0; i < index->nodes_.size(); ++i) {
            delete index->nodes_[i];
        }
        index->nodes_.clear();
    }

    void ComponentRefineHNSW::SetConfigs() {
        index->max_m_ = index->getParam().get<unsigned>("max_m");
        index->m_ = index->max_m_;
        index->max_m0_ = index->getParam().get<unsigned>("max_m0");
        index->ef_construction_ = index->getParam().get<unsigned>("ef_construction");
        index->n_threads_ = index->getParam().get<unsigned>("n_threads");
        index->mult = index->getParam().get<unsigned>("mult");
        index->level_mult_ = index->mult > 0 ? index->mult : 1 / log(1.0 * index->m_);
    }

    void ComponentRefineHNSW::Build(bool reverse) {
        index->nodes_.resize(index->getBaseLen());
        int level = GetRandomNodeLevel();
        Index::HnswNode *first = new Index::HnswNode(0, level, index->max_m_, index->max_m0_);
        index->nodes_[0] = first;
        index->max_level_ = level;
        index->enterpoint_ = first;
//#pragma omp parallel num_threads(index->n_threads_)
//        {
            Index::VisitedList *visited_list = new Index::VisitedList(index->getBaseLen());
            if (reverse) {
//#pragma omp for schedule(dynamic, 128)
                for (size_t i = index->getBaseLen() - 1; i >= 1; --i) {
                    int level = GetRandomNodeLevel();
                    Index::HnswNode *qnode = new Index::HnswNode(i, level, index->max_m_, index->max_m0_);
                    index->nodes_[i] = qnode;
                    InsertNode(qnode, visited_list);
                }
            } else {
//#pragma omp for schedule(dynamic, 128)
                for (size_t i = 1; i < index->getBaseLen(); ++i) {
                    int level = GetRandomNodeLevel();
                    auto *qnode = new Index::HnswNode(i, level, index->max_m_, index->max_m0_);
                    index->nodes_[i] = qnode;
                    InsertNode(qnode, visited_list);
                }
            }
            delete visited_list;
//        }
    }

    void ComponentRefineHNSW::InsertNode(Index::HnswNode *qnode, Index::VisitedList *visited_list) {
        int cur_level = qnode->GetLevel();
        std::unique_lock<std::mutex> max_level_lock(index->max_level_guard_, std::defer_lock);
        if (cur_level > index->max_level_)
            max_level_lock.lock();

        int max_level_copy = index->max_level_;
        Index::HnswNode *enterpoint = index->enterpoint_;

        if (cur_level < max_level_copy) {
            Index::HnswNode *cur_node = enterpoint;

            float d = index->getDist()->compare(index->getBaseData() + qnode->GetId() * index->getBaseDim(),
                                                index->getBaseData() + cur_node->GetId() * index->getBaseDim(),
                                                index->getBaseDim());
            float cur_dist = d;
            for (auto i = max_level_copy; i > cur_level; --i) {
                bool changed = true;
                while (changed) {
                    changed = false;
                    std::unique_lock<std::mutex> local_lock(cur_node->GetAccessGuard());
                    const std::vector<Index::HnswNode *> &neighbors = cur_node->GetFriends(i);
//                    for (auto iter = neighbors.begin(); iter != neighbors.end(); ++iter) {
//                        _mm_prefetch((*iter)->GetData(), _MM_HINT_T0);
//                    }
                    for (auto iter = neighbors.begin(); iter != neighbors.end(); ++iter) {
                        d = index->getDist()->compare(index->getBaseData() + qnode->GetId() * index->getBaseDim(),
                                                      index->getBaseData() + (*iter)->GetId() * index->getBaseDim(),
                                                      index->getBaseDim());

                        if (d < cur_dist) {
                            cur_dist = d;
                            cur_node = *iter;
                            changed = true;
                        }
                    }
                }
            }
            enterpoint = cur_node;
        }

        // PRUNE
        //std::cout << "wtf1" << std::endl;
        ComponentPrune *a = new ComponentPruneHeuristic(index);

        //_mm_prefetch(&selecting_policy_, _MM_HINT_T0);
        for (auto i = std::min(max_level_copy, cur_level); i >= 0; --i) {
            std::priority_queue<Index::FurtherFirst> result;
            SearchAtLayer(qnode, enterpoint, i, visited_list, result);

            a->Hnsw2Neighbor(index->m_, result);
            //std::cout << "level : " << i << " " << result.size() << std::endl;

            while (result.size() > 0) {
                //std::cout << "result1" << result.size() << std::endl;
                auto *top_node = result.top().GetNode();
                result.pop();
                //std::cout << "result2" << result.size() << std::endl;
                Link(top_node, qnode, i);
                //std::cout << "result3" << result.size() << std::endl;
                Link(qnode, top_node, i);
                //std::cout << "result4" << result.size() << std::endl;
            }
            //std::cout << "level : wtf" << std::endl;
        }
        //std::cout << "wtf2" << std::endl;
        if (cur_level > index->enterpoint_->GetLevel()) {
            index->enterpoint_ = qnode;
            index->max_level_ = cur_level;
        }
        //std::cout << "wtf1" << std::endl;
    }

    void ComponentRefineHNSW::SearchAtLayer(Index::HnswNode *qnode, Index::HnswNode *enterpoint, int level,
                                            Index::VisitedList *visited_list,
                                            std::priority_queue<Index::FurtherFirst> &result) {
        // TODO: check Node 12bytes => 8bytes
        std::priority_queue<Index::CloserFirst> candidates;
        float d = index->getDist()->compare(index->getBaseData() + qnode->GetId() * index->getBaseDim(),
                                            index->getBaseData() + enterpoint->GetId() * index->getBaseDim(),
                                            index->getBaseDim());
        result.emplace(enterpoint, d);
        candidates.emplace(enterpoint, d);

        visited_list->Reset();
        visited_list->MarkAsVisited(enterpoint->GetId());

        while (!candidates.empty()) {
            const Index::CloserFirst &candidate = candidates.top();
            float lower_bound = result.top().GetDistance();
            if (candidate.GetDistance() > lower_bound)
                break;

            Index::HnswNode *candidate_node = candidate.GetNode();
            std::unique_lock<std::mutex> lock(candidate_node->GetAccessGuard());
            const std::vector<Index::HnswNode *> &neighbors = candidate_node->GetFriends(level);
            candidates.pop();
//            for (const auto& neighbor : neighbors) {
//                _mm_prefetch(neighbor->GetData(), _MM_HINT_T0);
//            }
            for (const auto &neighbor : neighbors) {
                int id = neighbor->GetId();
                if (visited_list->NotVisited(id)) {
                    //_mm_prefetch(neighbor->GetData(), _MM_HINT_T0);
                    visited_list->MarkAsVisited(id);
                    d = index->getDist()->compare(index->getBaseData() + qnode->GetId() * index->getBaseDim(),
                                                  index->getBaseData() + neighbor->GetId() * index->getBaseDim(),
                                                  index->getBaseDim());
                    if (result.size() < index->ef_construction_ || result.top().GetDistance() > d) {
                        result.emplace(neighbor, d);
                        candidates.emplace(neighbor, d);
                        if (result.size() > index->ef_construction_)
                            result.pop();
                    }
                }
            }
        }
    }

    void ComponentRefineHNSW::Link(Index::HnswNode *source, Index::HnswNode *target, int level) {
        std::unique_lock<std::mutex> lock(source->GetAccessGuard());
        std::vector<Index::HnswNode *> &neighbors = source->GetFriends(level);
        neighbors.push_back(target);
        bool shrink = (level > 0 && neighbors.size() > source->GetMaxM()) ||
                      (level <= 0 && neighbors.size() > source->GetMaxM0());
        //std::cout << "shrink : " << shrink << std::endl;
        if (!shrink) return;

//        float max = index->getDist()->compare(index->getBaseData() + source->GetId() * index->getBaseDim(),
//                                              index->getBaseData() + neighbors[0]->GetId() * index->getBaseDim(),
//                                              index->getBaseDim());
//        int maxi = 0;
//        for(size_t i = 1; i < neighbors.size(); i ++) {
//            float curd = index->getDist()->compare(index->getBaseData() + source->GetId() * index->getBaseDim(),
//                                                   index->getBaseData() + neighbors[i]->GetId() * index->getBaseDim(),
//                                                   index->getBaseDim());
//
//            if(curd > max) {
//                max = curd;
//                maxi = i;
//            }
//        }
//        neighbors.erase(neighbors.begin() + maxi);

        std::priority_queue<Index::FurtherFirst> tempres;
//            for (const auto& neighbor : neighbors) {
//                _mm_prefetch(neighbor->GetData(), _MM_HINT_T0);
//            }
        std::cout << neighbors.size() << std::endl;
        for (const auto &neighbor : neighbors) {
            std::cout << "neighbors : " << neighbor->GetId() << std::endl;
            float tmp = index->getDist()->compare(index->getBaseData() + source->GetId() * index->getBaseDim(),
                                                  index->getBaseData() + neighbor->GetId() * index->getBaseDim(),
                                                  index->getBaseDim());
            std::cout << tmp << std::endl;
            std::cout << neighbor->GetId() << std::endl;
            std::cout << tempres.size() << std::endl;
            if(!tempres.empty())
                std::cout << tempres.top().GetNode()->GetId() << std::endl;
            tempres.push(Index::FurtherFirst(neighbors[0], tmp));
            std::cout << "mm" << std::endl;
        }
        std::cout << "tempres : " << tempres.size() << std::endl;

        // PRUNE
        ComponentPrune *a = new ComponentPruneHeuristic(index);
        std::cout << "wtf" << std::endl;

        a->Hnsw2Neighbor(tempres.size() - 1, tempres);
        std::cout << "ff" << tempres.size() << std::endl;
        neighbors.clear();
        while (tempres.size()) {
            neighbors.emplace_back(tempres.top().GetNode());
            tempres.pop();
        }
        std::priority_queue<Index::FurtherFirst>().swap(tempres);
    }

    int ComponentRefineHNSW::GetRandomSeedPerThread() {
        int tid = omp_get_thread_num();
        int g_seed = 17;
        for (int i = 0; i <= tid; ++i)
            g_seed = 214013 * g_seed + 2531011;
        return (g_seed >> 16) & 0x7FFF;
    }

    int ComponentRefineHNSW::GetRandomNodeLevel() {
        static thread_local std::mt19937 rng(GetRandomSeedPerThread());
        static thread_local std::uniform_real_distribution<double> uniform_distribution(0.0, 1.0);
        double r = uniform_distribution(rng);

        if (r < std::numeric_limits<double>::epsilon())
            r = 1.0;
        return (int) (-log(r) * index->level_mult_);
    }


    // VAMANA
    void ComponentRefineVAMANA::RefineInner() {
        SetConfigs();

        // ENTRY
        auto *a = new ComponentEntryCentroidNSG(index);
        a->EntryInner();

        auto *cut_graph_ = new Index::SimpleNeighbor[index->getBaseLen() * (size_t) index->R_nsg];
        Link(cut_graph_);

        index->getFinalGraph().resize(index->getBaseLen());

        for (size_t i = 0; i < index->getBaseLen(); i++) {
            Index::SimpleNeighbor *pool = cut_graph_ + i * (size_t) index->R_nsg;
            unsigned pool_size = 0;
            for (unsigned j = 0; j < index->R_nsg; j++) {
                if (pool[j].distance == -1) break;
                pool_size = j;
            }
            pool_size++;
            index->getFinalGraph()[i].resize(1);
            index->getFinalGraph()[i][0].resize(pool_size);
            for (unsigned j = 0; j < pool_size; j++) {
                index->getFinalGraph()[i][0][j] = pool[j].id;
            }
        }
    }

    void ComponentRefineVAMANA::SetConfigs() {
        index->alpha = index->getParam().get<float>("alpha");
    }

    void ComponentRefineVAMANA::Link(Index::SimpleNeighbor *cut_graph_) {
        std::vector<std::mutex> locks(index->getBaseLen());

        // CANDIDATE
        std::cout << "__CANDIDATE : GREEDY__" << std::endl;
        ComponentCandidate *a = new ComponentCandidateNSG(index);

        // PRUNE
        std::cout << "__PRUNE : VAMANA__" << std::endl;
        ComponentPrune *b1 = new ComponentPruneHeuristic(index);
        ComponentPrune *b2 = new ComponentPruneVAMANA(index);

        for(unsigned t = 0; t < 2; t ++) {
#pragma omp parallel
            {
                std::vector<std::vector<Index::Neighbor>> pool;
                pool.resize(index->getBaseLen());
                std::vector<boost::dynamic_bitset<>> flags(index->getBaseLen(),
                                                          boost::dynamic_bitset<>{index->getBaseLen(), 0});

                std::vector<Index::Neighbor> tmp;

#pragma omp for schedule(dynamic, 100)
                for (unsigned n = 0; n < index->getBaseLen(); ++n) {
                    a->CandidateInner(n, index->ep_, flags[n], pool[n], 0);

                    if(t == 0) {
                        b1->PruneInner(n, index->R_nsg, flags[n], pool[n], cut_graph_, 0);
                    }else {
                        b2->PruneInner(n, index->R_nsg, flags[n], pool[n], cut_graph_, 0);
                    }

//                    for (int i = 0; i < pool.size(); i++) {
//                        visited[n][i] = true;
//                    }
                }

                // 添加反向边
//#pragma omp for schedule(dynamic, 100)
//                for (unsigned n = 0; n < index->getBaseLen(); ++n) {
//                    for (unsigned m = 0; m < temp_graph[n].size(); m++) {
//                        if (visited[temp_graph[n][m].id][n] == false) {
//                            temp_graph[n].push_back(Index::Neighbor(n, temp_graph[n][m].distance, true));
//                            visited[temp_graph[n][m].id][n] = true;
//                        }
//                    }
//                }
//                std::vector<std::vector<bool>>().swap(visited);
//
//                // 裁边
//#pragma omp for schedule(dynamic, 100)
//                for (unsigned n = 0; n < index->getBaseLen(); ++n) {
//                    visited_list->Reset();
//                    if (t == 0) {
//                        b1->PruneInner(n, index->R_nsg, visited_list, temp_graph[n], cut_graph_, 0);
//                    } else {
//                        b2->PruneInner(n, index->R_nsg, visited_list, temp_graph[n], cut_graph_, 0);
//                    }
//                }
            }
        }

//#pragma omp for schedule(dynamic, 100)
//        for (unsigned n = 0; n < index->getBaseLen(); ++n) {
//            InterInsert(n, index->R_nsg, locks, cut_graph_);
//        }
    }


    // Test
    void ComponentRefineTest::RefineInner() {
        SetConfigs();

        // ENTRY
        auto *a = new ComponentEntryCentroidNSG(index);
        a->EntryInner();

        auto *cut_graph_ = new Index::SimpleNeighbor[index->getBaseLen() * (size_t) index->R_nsg];
        Link(cut_graph_);

        index->getFinalGraph().resize(index->getBaseLen());

        for (size_t i = 0; i < index->getBaseLen(); i++) {
            Index::SimpleNeighbor *pool = cut_graph_ + i * (size_t) index->R_nsg;
            unsigned pool_size = 0;
            for (unsigned j = 0; j < index->R_nsg; j++) {
                if (pool[j].distance == -1) break;
                pool_size = j;
            }
            pool_size++;
            index->getFinalGraph()[i].resize(1);
            index->getFinalGraph()[i][0].resize(pool_size);
            for (unsigned j = 0; j < pool_size; j++) {
                index->getFinalGraph()[i][0][j] = pool[j].id;
            }
        }
    }

    void ComponentRefineTest::SetConfigs() {

    }

    void ComponentRefineTest::Link(Index::SimpleNeighbor *cut_graph_) {
//        /*
// std::cerr << "Graph Link" << std::endl;
// unsigned progress = 0;
// unsigned percent = 100;
// unsigned step_size = nd_ / percent;
// std::mutex progress_lock;
// */
//        std::vector<std::mutex> locks(index->getBaseLen());
//
//#pragma omp parallel
//        {
//            std::vector<Index::Neighbor> pool;
//            auto *visited_list = new Index::VisitedList(index->getBaseLen());
//
//            ComponentCandidate *a = nullptr;
//            if (index->getCandidateType() == CANDIDATE_NAIVE)
//                a = new ComponentCandidateNSSG(index);
//
//            ComponentPrune *b = nullptr;
//            if (index->getPruneType() == PRUNE_NAIVE)
//                b = new ComponentPruneNaive(index);
//
//#pragma omp for schedule(dynamic, 100)
//            for (unsigned n = 0; n < index->getBaseLen(); ++n) {
//                std::cout << n << std::endl;
//                pool.clear();
//
//                visited_list->Reset();
//
//                a->CandidateInner(n, 0, visited_list, pool, 0);
//
//                visited_list->Reset();
//
//                b->PruneInner(n, index->R_nsg, visited_list, pool, cut_graph_, 0);
//
//                //std::cout << pool.size() << std::endl;
//                /*
//                cnt++;
//                if (cnt % step_size == 0) {
//                  LockGuard g(progress_lock);
//                  std::cout << progress++ << "/" << percent << " completed" << std::endl;
//                }
//                */
//            }
//            delete visited_list;
//        }
//
//#pragma omp for schedule(dynamic, 100)
//        for (unsigned n = 0; n < index->getBaseLen(); ++n) {
//            InterInsert(n, index->R_nsg, 0, locks, cut_graph_);
//        }
    }

    void ComponentRefineTest::InterInsert(unsigned n, unsigned range, float threshold, std::vector<std::mutex> &locks,
                                          Index::SimpleNeighbor *cut_graph_) {
        Index::SimpleNeighbor *src_pool = cut_graph_ + (size_t) n * (size_t) range;
        for (size_t i = 0; i < range; i++) {
            if (src_pool[i].distance == -1) break;

            Index::SimpleNeighbor sn(n, src_pool[i].distance);
            size_t des = src_pool[i].id;
            Index::SimpleNeighbor *des_pool = cut_graph_ + des * (size_t) range;

            std::vector<Index::SimpleNeighbor> temp_pool;
            int dup = 0;
            {
                Index::LockGuard guard(locks[des]);
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
                std::vector<Index::SimpleNeighbor> result;
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
                        float djk = index->getDist()->compare(
                                index->getBaseData() + index->getBaseDim() * (size_t) result[t].id,
                                index->getBaseData() + index->getBaseDim() * (size_t) p.id,
                                (unsigned) index->getBaseDim());
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
                    Index::LockGuard guard(locks[des]);
                    for (unsigned t = 0; t < result.size(); t++) {
                        des_pool[t] = result[t];
                    }
                    if (result.size() < range) {
                        des_pool[result.size()].distance = -1;
                    }
                }
            } else {
                Index::LockGuard guard(locks[des]);
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




}
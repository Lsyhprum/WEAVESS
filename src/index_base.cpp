//
// Created by Murph on 2020/8/12.
//
#include <weavess/index_base.h>
#include <weavess/parameters.h>
#include <queue>
#include <omp.h>
#include <set>

namespace weavess {
#define _CONTROL_NUM 100
    IndexGraph::IndexGraph(const size_t dimension, const size_t n, Metric m, Index *initializer)
            : Index(dimension, n, m),
              initializer_{initializer} {
        assert(dimension == initializer->GetDimension());
    }
    IndexGraph::~IndexGraph() {}

    void IndexGraph::join() {
#pragma omp parallel for default(shared) schedule(dynamic, 100)
        for (unsigned n = 0; n < nd_; n++) {
            graph_[n].join([&](unsigned i, unsigned j) {
                if(i != j){
                    float dist = distance_->compare(data_ + i * dimension_, data_ + j * dimension_, dimension_);
                    graph_[i].insert(j, dist);
                    graph_[j].insert(i, dist);
                }
            });
        }
    }
    void IndexGraph::update(const Parameters &parameters) {
        unsigned S = parameters.Get<unsigned>("S");
        unsigned R = parameters.Get<unsigned>("R");
        unsigned L = parameters.Get<unsigned>("L");
#pragma omp parallel for
        for (unsigned i = 0; i < nd_; i++) {
            std::vector<unsigned>().swap(graph_[i].nn_new);
            std::vector<unsigned>().swap(graph_[i].nn_old);
            //std::vector<unsigned>().swap(graph_[i].rnn_new);
            //std::vector<unsigned>().swap(graph_[i].rnn_old);
            //graph_[i].nn_new.clear();
            //graph_[i].nn_old.clear();
            //graph_[i].rnn_new.clear();
            //graph_[i].rnn_old.clear();
        }
#pragma omp parallel for
        for (unsigned n = 0; n < nd_; ++n) {
            auto &nn = graph_[n];
            // std::sort(nn.pool.begin(), nn.pool.end());
            if(nn.pool.size()>L)nn.pool.resize(L);
            nn.pool.reserve(L + 1);
            unsigned maxl = std::min(nn.M + S, (unsigned) nn.pool.size());
            unsigned c = 0;
            unsigned l = 0;
            //std::sort(nn.pool.begin(), nn.pool.end());
            //if(n==0)std::cout << nn.pool[0].distance<<","<< nn.pool[1].distance<<","<< nn.pool[2].distance<< std::endl;
            while ((l < maxl) && (c < S)) {
                if (nn.pool[l].flag) ++c;
                ++l;
            }
            nn.M = l;
        }
#pragma omp parallel for
        for (unsigned n = 0; n < nd_; ++n) {
            auto &nnhd = graph_[n];
            auto &nn_new = nnhd.nn_new;
            auto &nn_old = nnhd.nn_old;
            for (unsigned l = 0; l < nnhd.M; ++l) {
                auto &nn = nnhd.pool[l];
                auto &nhood_o = graph_[nn.id];  // nn on the other side of the edge
                if (nn.flag) {
                    nn_new.push_back(nn.id);
                    if (nn.distance > nhood_o.pool.back().distance)
                    {
                        LockGuard guard(nhood_o.lock);
                        if(nhood_o.rnn_new.size() < R)nhood_o.rnn_new.push_back(n);
                        else{
                            unsigned int pos = rand() % R;
                            nhood_o.rnn_new[pos] = n;
                        }
                    }
                    nn.flag = false;
                } else {
                    nn_old.push_back(nn.id);
                    if (nn.distance > nhood_o.pool.back().distance)
                    {
                        LockGuard guard(nhood_o.lock);
                        if(nhood_o.rnn_old.size() < R)nhood_o.rnn_old.push_back(n);
                        else{
                            unsigned int pos = rand() % R;
                            nhood_o.rnn_old[pos] = n;
                        }
                    }
                }
            }
            // std::make_heap(nnhd.pool.begin(), nnhd.pool.end());
        }
#pragma omp parallel for
        for (unsigned i = 0; i < nd_; ++i) {
            auto &nn_new = graph_[i].nn_new;
            auto &nn_old = graph_[i].nn_old;
            auto &rnn_new = graph_[i].rnn_new;
            auto &rnn_old = graph_[i].rnn_old;
            if (R && rnn_new.size() > R) {
                std::random_shuffle(rnn_new.begin(), rnn_new.end());
                rnn_new.resize(R);
            }
            nn_new.insert(nn_new.end(), rnn_new.begin(), rnn_new.end());
            if (R && rnn_old.size() > R) {
                std::random_shuffle(rnn_old.begin(), rnn_old.end());
                rnn_old.resize(R);
            }
            nn_old.insert(nn_old.end(), rnn_old.begin(), rnn_old.end());
            if(nn_old.size() > R * 2){nn_old.resize(R * 2);nn_old.reserve(R*2);}
            std::vector<unsigned>().swap(graph_[i].rnn_new);
            std::vector<unsigned>().swap(graph_[i].rnn_old);
        }
    } //update

    void IndexGraph::NNDescent(const Parameters &parameters) {
        unsigned iter = parameters.Get<unsigned>("iter");
        std::mt19937 rng(rand());
        std::vector<unsigned> control_points(_CONTROL_NUM);
        std::vector<std::vector<unsigned> > acc_eval_set(_CONTROL_NUM);
        GenRandom(rng, &control_points[0], control_points.size(), nd_);
        generate_control_set(control_points, acc_eval_set, nd_);
        for (unsigned it = 0; it < iter; it++) {
            join();
            update(parameters);
            //checkDup();
            eval_recall(control_points, acc_eval_set);
            std::cout << "iter: " << it << std::endl;
        }
    }

    void IndexGraph::Cut_Link(const Parameters &parameters, SimpleNeighbor *cut_graph_) {
        unsigned range = parameters.Get<unsigned>("RANGE");
        float m = parameters.Get<float>("M");
        std::vector<std::mutex> locks(nd_);

        // std::cout << "test1\n";
#pragma omp parallel
        {
            std::vector<Neighbor> pool;
#pragma omp for schedule(dynamic, 100)
            for (unsigned n = 0; n < nd_; ++n) {
                pool.clear();
                sync_prune(n, pool, m, parameters, cut_graph_); //cut edge
            }
#pragma omp for schedule(dynamic, 100)
            for (unsigned n = 0; n < nd_; ++n) {
                InterInsert(n, range, m, locks, cut_graph_); //reverse connection
            }
        }
    }

    void IndexGraph::get_neighbors(const float *query, const Parameters &parameter,
                                   std::vector<Neighbor> &retset,
                                   std::vector<Neighbor> &fullset,
                                   boost::dynamic_bitset<> cflags) {
        unsigned L = parameter.Get<unsigned>("PL");

        retset.resize(L + 1);
        std::vector<unsigned> init_ids(L);
        // initializer_->Search(query, nullptr, L, parameter, init_ids.data());
        std::mt19937 rng(rand());
        GenRandom(rng, init_ids.data(), L, (unsigned)nd_);

        boost::dynamic_bitset<> flags{nd_, 0};
        L = 0;
        for (unsigned i = 0; i < init_ids.size(); i++) {
            unsigned id = init_ids[i];
            if (id >= nd_ || !cflags[id]) continue;
            // std::cout<<id<<std::endl;
            float dist = distance_->compare(data_ + dimension_ * (size_t)id, query,
                                            (unsigned)dimension_);
            retset[i] = Neighbor(id, dist, true);
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

                for (unsigned m = 0; m < final_graph_[n].size(); ++m) {
                    unsigned id = final_graph_[n][m];
                    if (flags[id]) continue;
                    flags[id] = 1;

                    float dist = distance_->compare(query, data_ + dimension_ * (size_t)id,
                                                    (unsigned)dimension_);
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

    void IndexGraph::get_neighbors(const unsigned q, const Parameters &parameter,
                                   std::vector<Neighbor> &pool, boost::dynamic_bitset<> &flagss) {
        boost::dynamic_bitset<> flags{nd_, 0};
        unsigned PL = parameter.Get<unsigned>("PL");
        unsigned K = parameter.Get<unsigned>("K");
        unsigned b = parameter.Get<float>("B");
        unsigned ML = PL + pool.size();
        float bK = (float)K * b;
        flags[q] = true;
        for (unsigned i = 0; i < graph_[q].pool.size() && i < bK; i++) {
            unsigned nid = graph_[q].pool[i].id;
            for (unsigned nn = 0; nn < graph_[nid].pool.size() && i < bK; nn++) {
                unsigned nnid = graph_[nid].pool[nn].id;
                if (flags[nnid] || flagss[nnid]) continue;
                flags[nnid] = true;
                float d1 = graph_[q].pool[i].distance;
                float d2 = graph_[nid].pool[nn].distance;
                if (d1 < d2 && d2 - d1 > d1) continue;
                float dist = distance_->compare(data_ + dimension_ * q,
                                                data_ + dimension_ * nnid, dimension_);
                pool.push_back(Neighbor(nnid, dist, true));
                if (pool.size() >= ML) break;
            }
            if (pool.size() >= ML) break;
        }
        // if (pool.size() > ML) pool.resize(ML);
    }

    void IndexGraph::sync_prune(unsigned q, std::vector<Neighbor> &pool, float m,
                                const Parameters &parameters, SimpleNeighbor *cut_graph_) {
        unsigned range = parameters.Get<unsigned>("RANGE");
        width = range;
        unsigned start = 0;

        boost::dynamic_bitset<> flags{nd_, 0};
        for (unsigned nn = 0; nn < graph_[q].pool.size(); nn++) {
            unsigned id = graph_[q].pool[nn].id;
            flags[id] = 1;
            float dist = graph_[q].pool[nn].distance;
            bool f = graph_[q].pool[nn].flag;
            pool.push_back(Neighbor(id, dist, f));
        }
        get_neighbors(q, parameters, pool, flags);
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
                float djk = distance_->compare(data_ + dimension_ * (size_t)result[t].id,
                                               data_ + dimension_ * (size_t)p.id,
                                               (unsigned)dimension_);

                // float cos_ij = (p.distance + result[t].distance - djk) / 2 /
                //                sqrt(p.distance * result[t].distance);
                // if (cos_ij > threshold) {
                //   occlude = true;
                //   break;
                // }
                if (m * djk < p.distance) {
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

    void IndexGraph::InterInsert(unsigned n, unsigned range, float m,
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
                        float djk = distance_->compare(
                                data_ + dimension_ * (size_t)result[t].id,
                                data_ + dimension_ * (size_t)p.id, (unsigned)dimension_);

                        if (djk < p.distance) {
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

    void IndexGraph::get_cluster_center(const Parameters &parameter, boost::dynamic_bitset<> cflags, unsigned &cc) {
        float *center = new float[dimension_];
        for (unsigned j = 0; j < dimension_; j++) center[j] = 0;
        for (unsigned i = 0; i < nd_; i++) {
            if (cflags[i]) {
                for (unsigned j = 0; j < dimension_; j++) {
                    center[j] += data_[i * dimension_ + j];
                }
            }
        }
        for (unsigned j = 0; j < dimension_; j++) {
            center[j] /= nd_;
        }
        std::vector<Neighbor> tmp, pool;
        cc = rand() % nd_;  // random initialize navigating point
        get_neighbors(center, parameter, tmp, pool, cflags);
        cc = tmp[0].id;
    }

    void IndexGraph::DFS_expand(const Parameters &parameter) {  //partition dataset
        // unsigned n_try = parameter.Get<unsigned>("n_try");
        // unsigned range = parameter.Get<unsigned>("RANGE");
        unsigned id = rand() % nd_;


        boost::dynamic_bitset<> flags{nd_, 0};
        boost::dynamic_bitset<> cflags{nd_, 0};
        std::queue<unsigned> myqueue;
        myqueue.push(id);
        flags[id]=true;
        cflags[id] = true;

        std::vector<unsigned> uncheck_set(1);

        while(uncheck_set.size() >0){

            while(!myqueue.empty()){
                unsigned q_front=myqueue.front();
                myqueue.pop();

                for(unsigned j=0; j<final_graph_[q_front].size(); j++){
                    unsigned child = final_graph_[q_front][j];
                    if(flags[child])continue;
                    flags[child] = true;
                    cflags[child] = true;
                    myqueue.push(child);
                }
            }
            unsigned cc;
            get_cluster_center(parameter, cflags, cc);
            eps_.push_back(cc);

            uncheck_set.clear();
            for(unsigned j=0; j<nd_; j++){
                if(flags[j])continue;
                uncheck_set.push_back(j);
            }
            //std::cout <<i<<":"<< uncheck_set.size() << '\n';
            if(uncheck_set.size()>0){
                // for(unsigned j=0; j<nd_; j++){
                //   if(flags[j] && final_graph_[j].size()<range){
                //     final_graph_[j].push_back(uncheck_set[0]);
                //     break;
                //   }
                // }
                myqueue.push(uncheck_set[0]);
                flags[uncheck_set[0]]=true;
            }
            cflags.resize(nd_, 0);
        }
        std::cout << "navigation_points: " << eps_.size() << "\n";
    }

    void IndexGraph::generate_control_set(std::vector<unsigned> &c,
                                          std::vector<std::vector<unsigned> > &v,
                                          unsigned N){
#pragma omp parallel for
        for(unsigned i=0; i<c.size(); i++){
            std::vector<Neighbor> tmp;
            for(unsigned j=0; j<N; j++){
                float dist = distance_->compare(data_ + c[i] * dimension_, data_ + j * dimension_, dimension_);
                tmp.push_back(Neighbor(j, dist, true));
            }
            std::partial_sort(tmp.begin(), tmp.begin() + _CONTROL_NUM, tmp.end());
            for(unsigned j=0; j<_CONTROL_NUM; j++){
                v[i].push_back(tmp[j].id);
            }
        }
    }

    float IndexGraph::eval_recall(std::vector<unsigned>& ctrl_points, std::vector<std::vector<unsigned> > &acc_eval_set){
        float mean_acc=0;
        for(unsigned i=0; i<ctrl_points.size(); i++){
            float acc = 0;
            auto &g = graph_[ctrl_points[i]].pool;
            auto &v = acc_eval_set[i];
            for(unsigned j=0; j<g.size(); j++){
                for(unsigned k=0; k<v.size(); k++){
                    if(g[j].id == v[k]){
                        acc++;
                        break;
                    }
                }
            }
            mean_acc += acc / v.size();
        }
        float ret_recall = mean_acc / ctrl_points.size();
        std::cout<<"recall : "<< ret_recall <<std::endl;
        return ret_recall;
    }


    void IndexGraph::InitializeGraph(const Parameters &parameters) {

        const unsigned L = parameters.Get<unsigned>("L");
        const unsigned S = parameters.Get<unsigned>("S");

        graph_.reserve(nd_);
        std::mt19937 rng(rand());
        for (unsigned i = 0; i < nd_; i++) {
            graph_.push_back(nhood(L, S, rng, (unsigned) nd_));
        }
#pragma omp parallel for
        for (unsigned i = 0; i < nd_; i++) {
            const float *query = data_ + i * dimension_;
            std::vector<unsigned> tmp(S + 1);
            initializer_->Search(query, data_, S + 1, parameters, tmp.data());

            for (unsigned j = 0; j < S; j++) {
                unsigned id = tmp[j];
                if (id == i)continue;
                float dist = distance_->compare(data_ + i * dimension_, data_ + id * dimension_, (unsigned) dimension_);

                graph_[i].pool.push_back(Neighbor(id, dist, true));
            }
            std::sort(graph_[i].pool.begin(), graph_[i].pool.end());
            graph_[i].pool.reserve(L + 1);
        }
    }

    void IndexGraph::InitializeGraph_Refine(const Parameters &parameters) {
        assert(final_graph_.size() == nd_);

        const unsigned L = parameters.Get<unsigned>("L");
        const unsigned S = parameters.Get<unsigned>("S");

        graph_.reserve(nd_);
        std::mt19937 rng(rand());
        for (unsigned i = 0; i < nd_; i++) {
            graph_.push_back(nhood(L, S, rng, (unsigned) nd_));
        }
#pragma omp parallel for
        for (unsigned i = 0; i < nd_; i++) {
            auto& ids = final_graph_[i];
            std::sort(ids.begin(), ids.end());

            size_t K = ids.size();

            for (unsigned j = 0; j < K; j++) {
                unsigned id = ids[j];
                if (id == i || (j>0 &&id == ids[j-1]))continue;
                float dist = distance_->compare(data_ + i * dimension_, data_ + id * dimension_, (unsigned) dimension_);
                graph_[i].pool.push_back(Neighbor(id, dist, true));
            }
            std::make_heap(graph_[i].pool.begin(), graph_[i].pool.end());
            graph_[i].pool.reserve(L);
            std::vector<unsigned>().swap(ids);
        }
        CompactGraph().swap(final_graph_);
    }


    void IndexGraph::RefineGraph(const float* data, const Parameters &parameters) {
        data_ = data;
        assert(initializer_->HasBuilt());

        InitializeGraph_Refine(parameters);
        NNDescent(parameters);

        final_graph_.reserve(nd_);
        std::cout << nd_ << std::endl;
        unsigned K = parameters.Get<unsigned>("K");
        for (unsigned i = 0; i < nd_; i++) {
            std::vector<unsigned> tmp;
            std::sort(graph_[i].pool.begin(), graph_[i].pool.end());
            for (unsigned j = 0; j < K; j++) {
                tmp.push_back(graph_[i].pool[j].id);
            }
            tmp.reserve(K);
            final_graph_.push_back(tmp);
            std::vector<Neighbor>().swap(graph_[i].pool);
            std::vector<unsigned>().swap(graph_[i].nn_new);
            std::vector<unsigned>().swap(graph_[i].nn_old);
            std::vector<unsigned>().swap(graph_[i].rnn_new);
            std::vector<unsigned>().swap(graph_[i].rnn_new);
        }
        std::vector<nhood>().swap(graph_);
        has_built = true;

    }


    void IndexGraph::Build(size_t n, const float *data, const Parameters &parameters) {

        //assert(initializer_->GetDataset() == data);
        data_ = data;
        // assert(initializer_->HasBuilt());
        unsigned range = parameters.Get<unsigned>("RANGE");
        InitializeGraph(parameters);
        NNDescent(parameters);
        SimpleNeighbor *cut_graph_ = new SimpleNeighbor[nd_ * (size_t)range];
        Cut_Link(parameters, cut_graph_);
        final_graph_.resize(nd_);

        for (size_t i = 0; i < nd_; i++) {
            SimpleNeighbor *pool = cut_graph_ + i * (size_t)range;
            unsigned pool_size = 0;
            for (unsigned j = 0; j < range; j++) {
                if (pool[j].distance == -1) break;
                pool_size = j;
            }
            pool_size++;
            final_graph_[i].resize(pool_size);
            for (unsigned j = 0; j < pool_size; j++) {
                final_graph_[i][j] = pool[j].id;
            }
            std::vector<Neighbor>().swap(graph_[i].pool);
            std::vector<unsigned>().swap(graph_[i].nn_new);
            std::vector<unsigned>().swap(graph_[i].nn_old);
            std::vector<unsigned>().swap(graph_[i].rnn_new);
            std::vector<unsigned>().swap(graph_[i].rnn_new);
        }
        std::vector<nhood>().swap(graph_);
        //RefineGraph(parameters);

        DFS_expand(parameters);
        delete cut_graph_;
        unsigned max, min, avg;
        max = 0;
        min = nd_;
        avg = 0;
        for (size_t i = 0; i < nd_; i++) {
            auto size = final_graph_[i].size();
            max = max < size ? size : max;
            min = min > size ? size : min;
            avg += size;
        }
        avg /= 1.0 * nd_;
        printf("Degree Statistics: Max = %d, Min = %d, Avg = %d\n",
               max, min, avg);
        // has_built = true;
    }

    void IndexGraph::Search(
            const float *query,
            const float *x,
            size_t K,
            const Parameters &parameter,
            unsigned *indices) {
        const unsigned L = parameter.Get<unsigned>("L_search");

        std::vector<Neighbor> retset(L+1);
        std::vector<unsigned> init_ids(L);
        std::mt19937 rng(rand());
        GenRandom(rng, init_ids.data(), L, (unsigned)nd_);

        std::vector<char> flags(nd_);
        memset(flags.data(), 0, nd_ * sizeof(char));
        for(unsigned i=0; i<L; i++){
            unsigned id = init_ids[i];
            float dist = distance_->compare(data_ + dimension_*id, query, (unsigned)dimension_);
            retset[i]=Neighbor(id, dist, true);
        }

        std::sort(retset.begin(), retset.begin()+L);
        int k=0;
        while(k < (int)L) {
            int nk = L;

            if (retset[k].flag) {
                retset[k].flag = false;
                unsigned n = retset[k].id;

                for (unsigned m = 0; m < final_graph_[n].size(); ++m) {
                    unsigned id = final_graph_[n][m];
                    if(flags[id])continue;
                    flags[id] = 1;
                    float dist = distance_->compare(query, data_ + dimension_ * id, (unsigned)dimension_);
                    if(dist >= retset[L-1].distance)continue;
                    Neighbor nn(id, dist, true);
                    int r = InsertIntoPool(retset.data(), L, nn);

                    //if(L+1 < retset.size()) ++L;
                    if(r < nk)nk=r;
                }
                //lock to here
            }
            if(nk <= k)k = nk;
            else ++k;
        }
        for(size_t i=0; i < K; i++){
            indices[i] = retset[i].id;
        }
    }

    void IndexGraph::Save(const char *filename) {
        std::ofstream out(filename, std::ios::binary | std::ios::out);
        assert(final_graph_.size() == nd_);

        out.write((char *)&width, sizeof(unsigned));
        unsigned n_ep=eps_.size();
        out.write((char *)&n_ep, sizeof(unsigned));
        out.write((char *)eps_.data(), n_ep*sizeof(unsigned));
        for (unsigned i = 0; i < nd_; i++) {
            unsigned GK = (unsigned)final_graph_[i].size();
            out.write((char *)&GK, sizeof(unsigned));
            out.write((char *)final_graph_[i].data(), GK * sizeof(unsigned));
        }
        out.close();
    }

    void IndexGraph::Load(const char *filename) {
        std::ifstream in(filename, std::ios::binary);
        in.read((char *)&width, sizeof(unsigned));
        unsigned n_ep=0;
        in.read((char *)&n_ep, sizeof(unsigned));
        eps_.resize(n_ep);
        in.read((char *)eps_.data(), n_ep*sizeof(unsigned));
        // width=100;
        unsigned cc = 0;
        while (!in.eof()) {
            unsigned k;
            in.read((char *)&k, sizeof(unsigned));
            if (in.eof()) break;
            cc += k;
            std::vector<unsigned> tmp(k);
            in.read((char *)tmp.data(), k * sizeof(unsigned));
            final_graph_.push_back(tmp);
        }
        cc /= nd_;
        std::cerr << "Average Degree = " << cc << std::endl;
    }

    void IndexGraph::SearchWithOptGraph(const float *query, size_t K,
                                        const Parameters &parameters,
                                        unsigned *indices) {
        unsigned L = parameters.Get<unsigned>("L_search");
        DistanceFastL2 *dist_fast = (DistanceFastL2 *)distance_;

        std::vector<Neighbor> retset(L + 1);
        std::vector<unsigned> init_ids(L);
        std::mt19937 rng(rand());
        GenRandom(rng, init_ids.data(), L, (unsigned)nd_);
        // assert(eps_.size() < L);
        for(unsigned i=0; i<eps_.size() && i < L; i++){
            init_ids[i] = eps_[i];
        }

        boost::dynamic_bitset<> flags{nd_, 0};
        for (unsigned i = 0; i < init_ids.size(); i++) {
            unsigned id = init_ids[i];
            if (id >= nd_) continue;
            _mm_prefetch(opt_graph_ + node_size * id, _MM_HINT_T0);
        }
        L = 0;
        for (unsigned i = 0; i < init_ids.size(); i++) {
            unsigned id = init_ids[i];
            if (id >= nd_) continue;
            float *x = (float *)(opt_graph_ + node_size * id);
            float norm_x = *x;
            x++;
            float dist = dist_fast->compare(x, query, norm_x, (unsigned)dimension_);
            // float d = distance_ -> compare(x, query, (unsigned)dimension_);
            // std::cout << d << std::endl;
            dist_cout++;
            retset[i] = Neighbor(id, dist, true);
            flags[id] = true;
            L++;
        }
        // std::cout<<L<<std::endl;

        std::sort(retset.begin(), retset.begin() + L);
        int k = 0;
        while (k < (int)L) {
            int nk = L;

            if (retset[k].flag) {
                retset[k].flag = false;
                unsigned n = retset[k].id;

                _mm_prefetch(opt_graph_ + node_size * n + data_len, _MM_HINT_T0);
                unsigned *neighbors = (unsigned *)(opt_graph_ + node_size * n + data_len);
                neighbors++;
                unsigned MaxM = *neighbors;
                neighbors++;
                for (unsigned m = 0; m < MaxM; ++m)
                    _mm_prefetch(opt_graph_ + node_size * neighbors[m], _MM_HINT_T0);
                for (unsigned m = 0; m < MaxM; ++m) {
                    unsigned id = neighbors[m];
                    if (flags[id]) continue;
                    flags[id] = 1;
                    float *data = (float *)(opt_graph_ + node_size * id);
                    float norm = *data;
                    data++;
                    float dist =
                            dist_fast->compare(query, data, norm, (unsigned)dimension_);
                    dist_cout++;
                    if (dist >= retset[L - 1].distance) continue;
                    Neighbor nn(id, dist, true);
                    int r = InsertIntoPool(retset.data(), L, nn);

                    // if(L+1 < retset.size()) ++L;
                    if (r < nk) nk = r;
                }
            }
            if (nk <= k)
                k = nk;
            else
                ++k;
        }
        // for (size_t i = 0; i < L; i++) {  //重置候选集中的点为未访问
        //   retset[i].flag = true;
        // }
        k = 0;
        while (k < (int)L) {
            int nk = L;

            if (!retset[k].flag) {
                retset[k].flag = true;
                unsigned n = retset[k].id;

                _mm_prefetch(opt_graph_ + node_size * n + data_len, _MM_HINT_T0);
                unsigned *neighbors = (unsigned *)(opt_graph_ + node_size * n + data_len);
                unsigned MaxM = *neighbors;
                neighbors += 2;
                for (unsigned m = 0; m < MaxM; ++m)
                    _mm_prefetch(opt_graph_ + node_size * neighbors[m], _MM_HINT_T0);
                for (unsigned m = 0; m < MaxM; ++m) {
                    unsigned id = neighbors[m];
                    if (flags[id]) continue;
                    flags[id] = 1;
                    float *data = (float *)(opt_graph_ + node_size * id);
                    float norm = *data;
                    data++;
                    float dist =
                            dist_fast->compare(query, data, norm, (unsigned)dimension_);
                    dist_cout++;
                    if (dist >= retset[L - 1].distance) continue;
                    Neighbor nn(id, dist, false);
                    int r = InsertIntoPool(retset.data(), L, nn);

                    // if(L+1 < retset.size()) ++L;
                    if (r < nk) nk = r;
                }
            }
            if (nk <= k)
                k = nk;
            else
                ++k;
        }
        for (size_t i = 0; i < K; i++) {
            indices[i] = retset[i].id;
        }
    }

    void IndexGraph::FreeGraphData() {
        free(opt_graph_);
    }

    void IndexGraph::OptimizeGraph(float *data) {  // use after build or load

        data_ = data;
        data_len = (dimension_ + 1) * sizeof(float);
        neighbor_len = (width + 2) * sizeof(unsigned);
        node_size = data_len + neighbor_len;
        opt_graph_ = (char *)malloc(node_size * nd_);
        DistanceFastL2 *dist_fast = (DistanceFastL2 *)distance_;
        for (unsigned i = 0; i < nd_; i++) {
            char *cur_node_offset = opt_graph_ + i * node_size;
            float cur_norm = dist_fast->norm(data_ + i * dimension_, dimension_);
            std::memcpy(cur_node_offset, &cur_norm, sizeof(float));
            std::memcpy(cur_node_offset + sizeof(float), data_ + i * dimension_,
                        data_len - sizeof(float));

            cur_node_offset += data_len;
            unsigned k = final_graph_[i].size();
            std::memcpy(cur_node_offset, &k, sizeof(unsigned));
            unsigned kk = k/2;
            std::memcpy(cur_node_offset + sizeof(unsigned), &kk, sizeof(unsigned));
            std::memcpy(cur_node_offset + 2 * sizeof(unsigned), final_graph_[i].data(),
                        k * sizeof(unsigned));
            // std::vector<unsigned>().swap(final_graph_[i]);
        }
        // free(data);
        // data_ = nullptr;
        // CompactGraph().swap(final_graph_);
    }

    void IndexGraph::parallel_graph_insert(unsigned id, Neighbor nn, LockGraph& g, size_t K){
        LockGuard guard(g[id].lock);
        size_t l = g[id].pool.size();
        if(l == 0)g[id].pool.push_back(nn);
        else{
            g[id].pool.resize(l+1);
            g[id].pool.reserve(l+1);
            InsertIntoPool(g[id].pool.data(), (unsigned)l, nn);
            if(g[id].pool.size() > K)g[id].pool.reserve(K);
        }

    }

    void IndexGraph::GraphAdd(const float* data, unsigned n_new, unsigned dim, const Parameters &parameters) {
        data_ = data;
        data += nd_ * dimension_;
        assert(final_graph_.size() == nd_);
        assert(dim == dimension_);
        unsigned total = n_new + (unsigned)nd_;
        LockGraph graph_tmp(total);
        size_t K = final_graph_[0].size();
        compact_to_Lockgraph(graph_tmp);
        unsigned seed = 19930808;
#pragma omp parallel
        {
            std::mt19937 rng(seed ^ omp_get_thread_num());
#pragma omp for
            for(unsigned i = 0; i < n_new; i++){
                std::vector<Neighbor> res;
                get_neighbor_to_add(data + i * dim, parameters, graph_tmp, rng, res, n_new);

                for(unsigned j=0; j<K; j++){
                    parallel_graph_insert(i + (unsigned)nd_, res[j], graph_tmp, K);
                    parallel_graph_insert(res[j].id, Neighbor(i + (unsigned)nd_, res[j].distance, true), graph_tmp, K);
                }

            }
        };


        std::cout<<"complete: "<<std::endl;
        nd_ = total;
        final_graph_.resize(total);
        for(unsigned i=0; i<total; i++){
            for(unsigned m=0; m<K; m++){
                final_graph_[i].push_back(graph_tmp[i].pool[m].id);
            }
        }

    }

    void IndexGraph::get_neighbor_to_add(const float* point,
                                         const Parameters &parameters,
                                         LockGraph& g,
                                         std::mt19937& rng,
                                         std::vector<Neighbor>& retset,
                                         unsigned n_new){
        const unsigned L = parameters.Get<unsigned>("L_ADD");

        retset.resize(L+1);
        std::vector<unsigned> init_ids(L);
        GenRandom(rng, init_ids.data(), L/2, n_new);
        for(unsigned i=0; i<L/2; i++)init_ids[i] += nd_;

        GenRandom(rng, init_ids.data() + L/2, L - L/2, (unsigned)nd_);

        unsigned n_total = (unsigned)nd_ + n_new;
        std::vector<char> flags(n_new + n_total);
        memset(flags.data(), 0, n_total * sizeof(char));
        for(unsigned i=0; i<L; i++){
            unsigned id = init_ids[i];
            float dist = distance_->compare(data_ + dimension_*id, point, (unsigned)dimension_);
            retset[i]=Neighbor(id, dist, true);
        }

        std::sort(retset.begin(), retset.begin()+L);
        int k=0;
        while(k < (int)L) {
            int nk = L;

            if (retset[k].flag) {
                retset[k].flag = false;
                unsigned n = retset[k].id;

                LockGuard guard(g[n].lock);//lock start
                for (unsigned m = 0; m < g[n].pool.size(); ++m) {
                    unsigned id = g[n].pool[m].id;
                    if(flags[id])continue;
                    flags[id] = 1;
                    float dist = distance_->compare(point, data_ + dimension_ * id, (unsigned)dimension_);
                    if(dist >= retset[L-1].distance)continue;
                    Neighbor nn(id, dist, true);
                    int r = InsertIntoPool(retset.data(), L, nn);

                    //if(L+1 < retset.size()) ++L;
                    if(r < nk)nk=r;
                }
                //lock to here
            }
            if(nk <= k)k = nk;
            else ++k;
        }


    }

    void IndexGraph::compact_to_Lockgraph(LockGraph &g){

        //g.resize(final_graph_.size());
        for(unsigned i=0; i<final_graph_.size(); i++){
            g[i].pool.reserve(final_graph_[i].size()+1);
            for(unsigned j=0; j<final_graph_[i].size(); j++){
                float dist = distance_->compare(data_ + i*dimension_,
                                                data_ + final_graph_[i][j]*dimension_, (unsigned)dimension_);
                g[i].pool.push_back(Neighbor(final_graph_[i][j], dist, true));
            }
            std::vector<unsigned>().swap(final_graph_[i]);
        }
        CompactGraph().swap(final_graph_);
    }

}


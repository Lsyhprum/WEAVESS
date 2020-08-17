//
// Created by Murph on 2020/8/14.
//

#ifndef WEAVESS_INDEX_H
#define WEAVESS_INDEX_H

#define _CONTROL_NUM 100

#include <utility>
#include <vector>
#include <cstring>
#include "util.h"
#include "distance.h"
#include "neighbor.h"
#include "parameters.h"

namespace weavess {
    class Index {
    public:
        explicit Index(Parameters param) : param_(std::move(param)) {
            std::cout << param_.ToString() << std::endl;
        }

        /**
         * 内部实现类 ：
         */

        // Component
        class IndexComponent {
        public:
            explicit IndexComponent(Index *index) : index_(index) {}
            explicit IndexComponent(Index *index, Parameters param) : index_(index), param_(std::move(param)) {}

        protected:
            Index *index_;
            Parameters param_;
        };

        // Load
        enum FILE_TYPE {
            VECS
        };

        class IndexComponentLoad : public IndexComponent {
        public:
            explicit IndexComponentLoad(Index *index) : IndexComponent(index) {}

            virtual void LoadDataInner(char *data_file, float *&data, unsigned int &dim, unsigned int &n) = 0;
        };

        class IndexComponentLoadVECS : public IndexComponentLoad {
        public:
            explicit IndexComponentLoadVECS(Index *index) : IndexComponentLoad(index) {}

            void LoadDataInner(char *data_file, float *&data, unsigned int &dim, unsigned int &n) override {
                load_data(data_file, data, n, dim);
            }
        };

        // Init
        enum INIT_TYPE {
            Random, Other
        };

        class IndexComponentInit : public IndexComponent {
        public:
            explicit IndexComponentInit(Index *index, Parameters param) : IndexComponent(index, std::move(param)) {}

            virtual void InitInner() = 0;
        };

        class IndexComponentInitRandom : public IndexComponentInit {
        public:
            explicit IndexComponentInitRandom(Index *index, Parameters param) : IndexComponentInit(index, std::move(param)) {}

            void InitInner() override {
                const unsigned L = param_.Get<unsigned>("L");
                const unsigned S = param_.Get<unsigned>("S");

                index_->graph_.reserve(index_->n_);
                std::mt19937 rng(rand());
                for (unsigned i = 0; i < index_->n_; i++) {
                    index_->graph_.push_back(nhood(L, S, rng, (unsigned) index_->n_));
                }

#pragma omp parallel for
                for (unsigned i = 0; i < index_->n_; i++) {
                    const float *query = index_->data_ + i * index_->dim_;
                    std::vector<unsigned> tmp(S + 1);

                    weavess::GenRandom(rng, tmp.data(), S + 1, index_->n_);

                    for (unsigned j = 0; j < S; j++) {
                        unsigned id = tmp[j];
                        if (id == i)continue;
                        float dist = index_->distance_->compare(index_->data_ + i * index_->dim_,
                                                                index_->data_ + id * index_->dim_,
                                                                (unsigned) index_->dim_);

                        index_->graph_[i].pool.push_back(Neighbor(id, dist, true));
                    }
                    std::make_heap(index_->graph_[i].pool.begin(), index_->graph_[i].pool.end());
                    index_->graph_[i].pool.reserve(L);
                }
            }
        };

        // Coarse
        enum COARSE_TYPE{
            COARSE_NONE, NN_Descent
        };

        class IndexComponentCoarse : public IndexComponent {
        public:
            IndexComponentCoarse(Index *index, Parameters param) : IndexComponent(index, std::move(param)) {}

            virtual void CoarseInner() = 0;
        };

        class IndexComponentCoarseNNDescent : public IndexComponentCoarse {
        public:
            IndexComponentCoarseNNDescent(Index *index, Parameters param) : IndexComponentCoarse(index, param) {}

            void CoarseInner() override {
                unsigned iter = param_.Get<unsigned>("iter");
                std::mt19937 rng(rand());
                std::vector<unsigned> control_points(_CONTROL_NUM);
                std::vector<std::vector<unsigned> > acc_eval_set(_CONTROL_NUM);
                GenRandom(rng, &control_points[0], control_points.size(), index_->n_);
                generate_control_set(control_points, acc_eval_set, index_->n_);
                for (unsigned it = 0; it < iter; it++) {
                    join();
                    update(param_);
                    //checkDup();
                    eval_recall(control_points, acc_eval_set);
                    std::cout << "iter: " << it << std::endl;
                }

                index_->final_graph_.reserve(index_->n_);
                unsigned K = param_.Get<unsigned>("K");
                for (unsigned i = 0; i < index_->n_; i++) {
                    std::vector<unsigned> tmp;
                    std::sort(index_->graph_[i].pool.begin(), index_->graph_[i].pool.end());
                    for (unsigned j = 0; j < K; j++) {
                        tmp.push_back(index_->graph_[i].pool[j].id);
                    }
                    tmp.reserve(K);
                    index_->final_graph_.push_back(tmp);
                    std::vector<Neighbor>().swap(index_->graph_[i].pool);
                    std::vector<unsigned>().swap(index_->graph_[i].nn_new);
                    std::vector<unsigned>().swap(index_->graph_[i].nn_old);
                    std::vector<unsigned>().swap(index_->graph_[i].rnn_new);
                    std::vector<unsigned>().swap(index_->graph_[i].rnn_new);
                }
                std::vector<nhood>().swap(index_->graph_);
            }

        private:
            void join() {
#pragma omp parallel for default(shared) schedule(dynamic, 100)
                for (unsigned n = 0; n < index_->n_; n++) {
                    index_->graph_[n].join([&](unsigned i, unsigned j) {
                        if(i != j){
                            float dist = index_->distance_->compare(index_->data_ + i * index_->dim_, index_->data_ + j * index_->dim_, index_->dim_);
                            index_->graph_[i].insert(j, dist);
                            index_->graph_[j].insert(i, dist);
                        }
                    });
                }
            }

            void update(const Parameters &parameters) {
                unsigned S = parameters.Get<unsigned>("S");
                unsigned R = parameters.Get<unsigned>("R");
                unsigned L = parameters.Get<unsigned>("L");
#pragma omp parallel for
                for (unsigned i = 0; i < index_->n_; i++) {
                    std::vector<unsigned>().swap(index_->graph_[i].nn_new);
                    std::vector<unsigned>().swap(index_->graph_[i].nn_old);
                    //std::vector<unsigned>().swap(graph_[i].rnn_new);
                    //std::vector<unsigned>().swap(graph_[i].rnn_old);
                    //graph_[i].nn_new.clear();
                    //graph_[i].nn_old.clear();
                    //graph_[i].rnn_new.clear();
                    //graph_[i].rnn_old.clear();
                }
#pragma omp parallel for
                for (unsigned n = 0; n < index_->n_; ++n) {
                    auto &nn = index_->graph_[n];
                    std::sort(nn.pool.begin(), nn.pool.end());
                    if(nn.pool.size()>L)nn.pool.resize(L);
                    nn.pool.reserve(L);
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
                for (unsigned n = 0; n < index_->n_; ++n) {
                    auto &nnhd = index_->graph_[n];
                    auto &nn_new = nnhd.nn_new;
                    auto &nn_old = nnhd.nn_old;
                    for (unsigned l = 0; l < nnhd.M; ++l) {
                        auto &nn = nnhd.pool[l];
                        auto &nhood_o = index_->graph_[nn.id];  // nn on the other side of the edge

                        if (nn.flag) {
                            nn_new.push_back(nn.id);
                            if (nn.distance > nhood_o.pool.back().distance) {
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
                            if (nn.distance > nhood_o.pool.back().distance) {
                                LockGuard guard(nhood_o.lock);
                                if(nhood_o.rnn_old.size() < R)nhood_o.rnn_old.push_back(n);
                                else{
                                    unsigned int pos = rand() % R;
                                    nhood_o.rnn_old[pos] = n;
                                }
                            }
                        }
                    }
                    std::make_heap(nnhd.pool.begin(), nnhd.pool.end());
                }
#pragma omp parallel for
                for (unsigned i = 0; i < index_->n_; ++i) {
                    auto &nn_new = index_->graph_[i].nn_new;
                    auto &nn_old = index_->graph_[i].nn_old;
                    auto &rnn_new = index_->graph_[i].rnn_new;
                    auto &rnn_old = index_->graph_[i].rnn_old;
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
                    std::vector<unsigned>().swap(index_->graph_[i].rnn_new);
                    std::vector<unsigned>().swap(index_->graph_[i].rnn_old);
                }
            }

            void generate_control_set(std::vector<unsigned> &c, std::vector<std::vector<unsigned> > &v, unsigned N){
#pragma omp parallel for
                for(unsigned i=0; i<c.size(); i++){
                    std::vector<Neighbor> tmp;
                    for(unsigned j=0; j<N; j++){
                        float dist = index_->distance_->compare(index_->data_ + c[i] * index_->dim_, index_->data_ + j * index_->dim_, index_->dim_);
                        tmp.push_back(Neighbor(j, dist, true));
                    }
                    std::partial_sort(tmp.begin(), tmp.begin() + _CONTROL_NUM, tmp.end());
                    for(unsigned j=0; j<_CONTROL_NUM; j++){
                        v[i].push_back(tmp[j].id);
                    }
                }
            }

            void eval_recall(std::vector<unsigned>& ctrl_points, std::vector<std::vector<unsigned> > &acc_eval_set){
                float mean_acc=0;
                for(unsigned i=0; i<ctrl_points.size(); i++){
                    float acc = 0;
                    auto &g = index_->graph_[ctrl_points[i]].pool;
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
                std::cout<<"recall : "<<mean_acc / ctrl_points.size() <<std::endl;
            }

        };

        // prune
        enum PRUNE_TYPE{
            PRUNE_NONE, NSG, SSG
        };

        class IndexComponentPrune : public IndexComponent {
        public:
            IndexComponentPrune(Index *index, Parameters &param) : IndexComponent(index, param) {}

            virtual void PruneInner() = 0;
        };

        class IndexComponentPruneNone : public IndexComponentPrune {
        public:
            IndexComponentPruneNone(Index *index, Parameters param) : IndexComponentPrune(index, param) {}

            void PruneInner() override {}
        };

        // connect
        enum CONN_TYPE{
           CONN_NONE, DFS
        };
        class IndexComponentConnect : public IndexComponent {
        public:
            IndexComponentConnect(Index *index, Parameters param) : IndexComponent(index, param) {}

            virtual void ConnectInner() = 0;
        };

        class IndexComponentConnectNone : public IndexComponentConnect {
        public:
            explicit IndexComponentConnectNone(Index *index, Parameters param) : IndexComponentConnect(index, param) {}

            void ConnectInner() override {}
        };

        class IndexComponentConnectDFS : public IndexComponentConnect {
        public:
            IndexComponentConnectDFS(Index *index, Parameters &param) : IndexComponentConnect(index, param) {}

            void ConnectInner() override {

            }
        };

        inline void IndexLoadData(FILE_TYPE type, char *&data_file) {
            IndexComponentLoad *loader = nullptr;
            switch (type) {
                case Index::VECS:
                    std::cout << "__Loader I/FVECS__" << std::endl;
                    loader = new IndexComponentLoadVECS(this);
                    break;
                default:
                    std::cerr << "load data wrong type" << std::endl;
            }

            loader->LoadDataInner(data_file, data_, dim_, n_);
            std::cout << "data dimension: " << dim_ << std::endl;
            std::cout << "data num: " << n_ << std::endl;
        }

        inline void IndexInit(INIT_TYPE type) {
            Index::IndexComponentInit *a = nullptr;
            switch (type) {
                case Index::Random:
                    a = new Index::IndexComponentInitRandom(this, param_);
                    break;
                default:
                    std::cerr << "init index wrong type" << std::endl;
            }
            a->InitInner();
        }

        inline void IndexCoarseBuild(COARSE_TYPE type) {
            Index::IndexComponentCoarse *a = nullptr;
            switch (type) {
                case Index::NN_Descent:
                    a = new Index::IndexComponentCoarseNNDescent(this, param_);
                    break;
                default:
                    std::cerr << "coarse KNN wrong type" << std::endl;
            }
            a->CoarseInner();
        }

        inline void IndexPrune(PRUNE_TYPE type) {
            Index::IndexComponentPrune *a = nullptr;
            switch (type) {
                case Index::PRUNE_NONE:
                    a = new Index::IndexComponentPruneNone(this, param_);
                    break;
                default:
                    std::cerr << "prune wrong type" << std::endl;
            }
            a->PruneInner();
        }

        inline void IndexConnect(CONN_TYPE type) {
            Index::IndexComponentConnect *a = nullptr;

            switch (type) {
                case Index::CONN_NONE:

                    a = new Index::IndexComponentConnectNone(this, param_);
                    break;
                default:
                    std::cerr << "connect wrong type" << std::endl;
            }
            a->ConnectInner();
        }

        void Search(const float *query, const float *x, size_t K, const Parameters &parameter, unsigned *indices) {
            const unsigned L = parameter.Get<unsigned>("L_search");

            std::vector<Neighbor> retset(L+1);
            std::vector<unsigned> init_ids(L);
            std::mt19937 rng(rand());
            GenRandom(rng, init_ids.data(), L, (unsigned)n_);

            std::vector<char> flags(n_);
            memset(flags.data(), 0, n_ * sizeof(char));
            for(unsigned i=0; i<L; i++){
                unsigned id = init_ids[i];
                float dist = distance_->compare(data_ + dim_*id, query, (unsigned)dim_);
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
                        float dist = distance_->compare(query, data_ + dim_ * id, (unsigned)dim_);
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

        void FreeGraphData() {

        }

        inline float *GetData() { return data_; }

        inline unsigned int GetDim() const { return dim_; }

        inline unsigned GetNum() const { return n_; }

    private:
        float *data_ = nullptr;
        unsigned dim_{};
        unsigned n_{};
        const Parameters param_;
        Distance *distance_ = new Distance();

        typedef std::vector<nhood> KNNGraph;
        typedef std::vector<std::vector<unsigned> > CompactGraph;
        typedef std::vector<LockNeighbor> LockGraph;

        Index *initializer_{};
        KNNGraph graph_;
        CompactGraph final_graph_;
    };
}


#endif //WEAVESS_INDEX_H

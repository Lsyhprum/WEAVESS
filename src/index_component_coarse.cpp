//
// Created by Murph on 2020/8/24.
//
#include "weavess/index_builder.h"

namespace weavess {
    void IndexComponentCoarseNNDescent::CoarseInner() {
        const unsigned iter = index_->param_.get<unsigned>("iter");
        const unsigned K = index_->param_.get<unsigned>("K");

        std::mt19937 rng(rand());

        std::vector<unsigned> control_points(_CONTROL_NUM);
        std::vector<std::vector<unsigned> > acc_eval_set(_CONTROL_NUM);
        GenRandom(rng, &control_points[0], control_points.size(), index_->n_);
        generate_control_set(control_points, acc_eval_set, index_->n_);
        for (unsigned it = 0; it < iter; it++) {
            join();
            update(index_->param_);
            //checkDup();
            eval_recall(control_points, acc_eval_set);
            std::cout << "iter: " << it << std::endl;
        }

        index_->final_graph_.reserve(index_->n_);

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

    void IndexComponentCoarseNNDescent::IndexComponentCoarseNNDescent::join() {
#pragma omp parallel for default(shared) schedule(dynamic, 100)
        for (unsigned n = 0; n < index_->n_; n++) {
            index_->graph_[n].join([&](unsigned i, unsigned j) {
                if (i != j) {
                    float dist = index_->distance_->compare(index_->data_ + i * index_->dim_,
                                                            index_->data_ + j * index_->dim_, index_->dim_);

                    index_->graph_[i].insert(j, dist);
                    index_->graph_[j].insert(i, dist);
                }
            });
        }
    }

    void IndexComponentCoarseNNDescent::IndexComponentCoarseNNDescent::update(const Parameters &parameters) {
        unsigned S = parameters.get<unsigned>("S");
        unsigned R = parameters.get<unsigned>("R");
        unsigned L = parameters.get<unsigned>("L");
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
            if (nn.pool.size() > L)nn.pool.resize(L);
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
                        if (nhood_o.rnn_new.size() < R)nhood_o.rnn_new.push_back(n);
                        else {
                            unsigned int pos = rand() % R;
                            nhood_o.rnn_new[pos] = n;
                        }
                    }
                    nn.flag = false;
                } else {
                    nn_old.push_back(nn.id);
                    if (nn.distance > nhood_o.pool.back().distance) {
                        LockGuard guard(nhood_o.lock);
                        if (nhood_o.rnn_old.size() < R)nhood_o.rnn_old.push_back(n);
                        else {
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
            if (nn_old.size() > R * 2) {
                nn_old.resize(R * 2);
                nn_old.reserve(R * 2);
            }
            std::vector<unsigned>().swap(index_->graph_[i].rnn_new);
            std::vector<unsigned>().swap(index_->graph_[i].rnn_old);
        }
    }

    void IndexComponentCoarseNNDescent::IndexComponentCoarseNNDescent::generate_control_set(std::vector<unsigned> &c, std::vector<std::vector<unsigned> > &v, unsigned N){
#pragma omp parallel for
        for (unsigned i = 0; i < c.size(); i++) {
            std::vector<Neighbor> tmp;
            for (unsigned j = 0; j < N; j++) {
                float dist = index_->distance_->compare(index_->data_ + c[i] * index_->dim_,
                                                        index_->data_ + j * index_->dim_, index_->dim_);
                tmp.push_back(Neighbor(j, dist, true));
            }
            std::partial_sort(tmp.begin(), tmp.begin() + _CONTROL_NUM, tmp.end());
            for (unsigned j = 0; j < _CONTROL_NUM; j++) {
                v[i].push_back(tmp[j].id);
            }
        }
    }

    void IndexComponentCoarseNNDescent::IndexComponentCoarseNNDescent::eval_recall(std::vector<unsigned>& ctrl_points, std::vector<std::vector<unsigned> > &acc_eval_set){
        float mean_acc = 0;
        for (unsigned i = 0; i < ctrl_points.size(); i++) {
            float acc = 0;
            auto &g = index_->graph_[ctrl_points[i]].pool;
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
}

//
// Created by Murph on 2020/8/24.
//
#include "weavess/index_builder.h"

namespace weavess {
    void IndexComponentEva::EvaInner(char *query_file, char *ground_truth_file) {
        float *query_data = nullptr;
        unsigned query_num, query_dim;
        unsigned *ground_load = nullptr;
        unsigned ground_num, ground_dim;

        load_data<float>(query_file, query_data, query_num, query_dim);
        load_data<unsigned>(ground_truth_file, ground_load, ground_num, ground_dim);

        unsigned K = 50;
        unsigned L_start = 50;
        unsigned L_end = 2000;
        unsigned experiment_num = 15;
        unsigned LI = (L_end - L_start) / experiment_num;

        for (unsigned L = L_start; L <= L_end; L += LI) {
            if (L < K) {
                std::cout << "search_L cannot be smaller than search_K!"
                          << std::endl;
                exit(-1);
            }

            index_->param_.set<unsigned>("L_search", L);
            index_->param_.set<unsigned>("P_search", L);

            auto s = std::chrono::high_resolution_clock::now();
            std::vector<std::vector<unsigned>> res;
            unsigned distcount = 0;
            for (unsigned i = 0; i < query_num; i++) {
                std::vector<unsigned> tmp(K);
                SearchInner(query_data + i * index_->dim_, index_->data_, K, index_->param_, tmp.data(), distcount);
                res.push_back(tmp);
            }
            auto e = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> diff = e - s;
            std::cout << "search time: " << diff.count() << "\n";

//            float speedup = (float)(index_->n_ * query_num) / (float)distcount;
            std::cout << "DistCount: " << distcount << std::endl;
            //结果评估
            int cnt = 0;
            for (unsigned i = 0; i < ground_num; i++) {
                for (unsigned j = 0; j < K; j++) {
                    unsigned k = 0;
                    for (; k < K; k++) {
                        if (res[i][j] == ground_load[i * ground_dim + k])
                            break;
                    }
                    if (k == K)
                        cnt++;
                }
            }
            float acc = 1 - (float) cnt / (ground_num * K);
            std::cout << K << " NN accuracy: " << acc << std::endl;
        }
    }

    void IndexComponentEvaRandom::SearchInner(const float *query, const float *x, size_t K,
                                              const Parameters &parameters, unsigned int *indices,
                                              unsigned int &distcount) {
        unsigned L = parameters.get<unsigned>("L_search");

        std::vector<Neighbor> retset(L+1);
        std::vector<unsigned> init_ids(L);
        std::mt19937 rng(rand());
        GenRandom(rng, init_ids.data(), L, (unsigned)index_->n_);

        std::vector<char> flags(index_->n_);
        memset(flags.data(), 0, index_->n_ * sizeof(char));
        for(unsigned i=0; i<L; i++){
            unsigned id = init_ids[i];
            float dist = index_->distance_->compare(index_->data_ + index_->dim_*id, query, (unsigned)index_->dim_);

            distcount ++;

            retset[i]=Neighbor(id, dist, true);
        }

        std::sort(retset.begin(), retset.begin()+L);

        int k=0;
        while(k < (int)L) {
            int nk = L;

            if (retset[k].flag) {
                retset[k].flag = false;
                unsigned n = retset[k].id;

                for (unsigned m = 0; m < index_->final_graph_[n].size(); ++m) {
                    unsigned id = index_->final_graph_[n][m];
                    if(flags[id])continue;
                    flags[id] = 1;
                    float dist = index_->distance_->compare(query, index_->data_ + index_->dim_ * id, (unsigned)index_->dim_);
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

    void IndexComponentEvaNSG::SearchInner(const float *query, const float *x, size_t K,
                                              const Parameters &parameters, unsigned int *indices,
                                              unsigned int &distcount) {
        const unsigned L = parameters.get<unsigned>("L_search");
        std::vector<Neighbor> retset(L + 1);
        std::vector<unsigned> init_ids(L);
        boost::dynamic_bitset<> flags{index_->n_, 0};
        // std::mt19937 rng(rand());
        // GenRandom(rng, init_ids.data(), L, (unsigned) index_->n_);

        unsigned tmp_l = 0;
        for (; tmp_l < L && tmp_l < index_->final_graph_[index_->ep_].size(); tmp_l++) {
            init_ids[tmp_l] = index_->final_graph_[index_->ep_][tmp_l];
            flags[init_ids[tmp_l]] = true;
        }

        while (tmp_l < L) {
            unsigned id = rand() % index_->n_;
            if (flags[id]) continue;
            flags[id] = true;
            init_ids[tmp_l] = id;
            tmp_l++;
        }

        for (unsigned i = 0; i < init_ids.size(); i++) {
            unsigned id = init_ids[i];
            float dist =
                    index_->distance_->compare(index_->data_ + index_->dim_ * id, query, (unsigned)index_->dim_);
            retset[i] = Neighbor(id, dist, true);
            // flags[id] = true;
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
                    float dist =
                            index_->distance_->compare(query, index_->data_ + index_->dim_ * id, (unsigned)index_->dim_);
                    if (dist >= retset[L - 1].distance) continue;
                    Neighbor nn(id, dist, true);
                    int r = InsertIntoPool(retset.data(), L, nn);

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
}

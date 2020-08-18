//
// Created by Murph on 2020/8/18.
//

#ifndef WEAVESS_INDEX_COMPONENT_CONNECT_H
#define WEAVESS_INDEX_COMPONENT_CONNECT_H

#include "index_component.h"
#include <stack>

namespace weavess {

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
        IndexComponentConnectDFS(Index *index, Parameters param) : IndexComponentConnect(index, param) {}

        void get_neighbors(const float *query, const Parameters &parameter,
                                     std::vector<Neighbor> &retset,
                                     std::vector<Neighbor> &fullset) {
            unsigned L = parameter.Get<unsigned>("L");

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



        void findroot(boost::dynamic_bitset<> &flag, unsigned &root,
                                const Parameters &parameter) {
            unsigned id = index_->n_;
            for (unsigned i = 0; i < index_->n_; i++) {
                if (flag[i] == false) {
                    id = i;
                    break;
                }
            }

            if (id == index_->n_) return;  // No Unlinked Node

            std::vector<Neighbor> tmp, pool;
            get_neighbors(index_->data_ + index_->dim_ * id, parameter, tmp, pool);
            std::sort(pool.begin(), pool.end());

            unsigned found = 0;
            for (unsigned i = 0; i < pool.size(); i++) {
                if (flag[pool[i].id]) {
                    // std::cout << pool[i].id << '\n';
                    root = pool[i].id;
                    found = 1;
                    break;
                }
            }
            if (found == 0) {
                while (true) {
                    unsigned rid = rand() % index_->n_;
                    if (flag[rid]) {
                        root = rid;
                        break;
                    }
                }
            }
            index_->final_graph_[root].push_back(id);
        }


        void DFS(boost::dynamic_bitset<> &flag, unsigned root, unsigned &cnt) {
            unsigned tmp = root;
            std::stack<unsigned> s;
            s.push(root);
            if (!flag[root]) cnt++;
            flag[root] = true;
            while (!s.empty()) {
                unsigned next = index_->n_ + 1;
                for (unsigned i = 0; i < index_->final_graph_[tmp].size(); i++) {
                    if (flag[index_->final_graph_[tmp][i]] == false) {
                        next = index_->final_graph_[tmp][i];
                        break;
                    }
                }
                // std::cout << next <<":"<<cnt <<":"<<tmp <<":"<<s.size()<< '\n';
                if (next == (index_->n_ + 1)) {
                    s.pop();
                    if (s.empty()) break;
                    tmp = s.top();
                    continue;
                }
                tmp = next;
                flag[tmp] = true;
                s.push(tmp);
                cnt++;
            }
        }

        void tree_grow(const Parameters &parameter) {
            unsigned root = index_->ep_;
            boost::dynamic_bitset<> flags{index_->n_, 0};
            unsigned unlinked_cnt = 0;
            while (unlinked_cnt < index_->n_) {
                DFS(flags, root, unlinked_cnt);
                // std::cout << unlinked_cnt << '\n';
                if (unlinked_cnt >= index_->n_) break;
                findroot(flags, root, parameter);
                // std::cout << "new root"<<":"<<root << '\n';
            }
            for (size_t i = 0; i < index_->n_; ++i) {
                if (index_->final_graph_[i].size() > index_->width) {
                    index_->width = index_->final_graph_[i].size();
                }
            }
        }

        void ConnectInner() override {
            tree_grow(param_);
        }
    };
}

#endif //WEAVESS_INDEX_COMPONENT_CONNECT_H

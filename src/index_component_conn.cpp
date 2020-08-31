//
// Created by Murph on 2020/8/25.
//
#include "weavess/index_builder.h"

namespace weavess {
    // NSG
    void IndexComponentConnDFS::ConnInner() {
        tree_grow(index_->param_);
    }

    void IndexComponentConnDFS::tree_grow(const Parameters &parameter) {
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

    void IndexComponentConnDFS::DFS(boost::dynamic_bitset<> &flag, unsigned root, unsigned &cnt) {
        unsigned tmp = root;
        std::stack<unsigned> s;
        s.push(root);
        if (!flag[root]) cnt++;
        flag[root] = true;
        while (!s.empty()) {
            unsigned next = index_->n_ + 1;
            for (unsigned i = 0; i < index_->final_graph_[tmp].size(); i++) {
                if (!flag[index_->final_graph_[tmp][i]]) {
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

    void IndexComponentConnDFS::findroot(boost::dynamic_bitset<> &flag, unsigned &root, const Parameters &parameter) {
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

    void IndexComponentConnDFS::get_neighbors(const float *query, const Parameters &parameter,
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


    // NSSG
    void IndexComponentConnDFS_EXPAND::ConnInner() {
        DFS_expand(index_->param_);
    }

    void IndexComponentConnDFS_EXPAND::DFS_expand(const Parameters &parameter) {
        unsigned n_try = parameter.get<unsigned>("n_try");
        unsigned range = parameter.get<unsigned>("R_nsg");

        std::vector<unsigned> ids(index_->n_);
        for(unsigned i=0; i<index_->n_; i++){
            ids[i]=i;
        }
        std::random_shuffle(ids.begin(), ids.end());
        for(unsigned i=0; i<n_try; i++){
            index_->eps_.push_back(ids[i]);
            //std::cout << eps_[i] << '\n';
        }
#pragma omp parallel for
        for(unsigned i=0; i<n_try; i++){
            unsigned rootid = index_->eps_[i];
            boost::dynamic_bitset<> flags{index_->n_, 0};
            std::queue<unsigned> myqueue;
            myqueue.push(rootid);
            flags[rootid]=true;

            std::vector<unsigned> uncheck_set(1);

            while(uncheck_set.size() >0){
                while(!myqueue.empty()){
                    unsigned q_front=myqueue.front();
                    myqueue.pop();

                    for(unsigned j=0; j<index_->final_graph_[q_front].size(); j++){
                        unsigned child = index_->final_graph_[q_front][j];
                        if(flags[child])continue;
                        flags[child] = true;
                        myqueue.push(child);
                    }
                }

                uncheck_set.clear();
                for(unsigned j=0; j<index_->n_; j++){
                    if(flags[j])continue;
                    uncheck_set.push_back(j);
                }
                //std::cout <<i<<":"<< uncheck_set.size() << '\n';
                if(uncheck_set.size()>0){
                    for(unsigned j=0; j<index_->n_; j++){
                        if(flags[j] && index_->final_graph_[j].size()<range){
                            index_->final_graph_[j].push_back(uncheck_set[0]);
                            break;
                        }
                    }
                    myqueue.push(uncheck_set[0]);
                    flags[uncheck_set[0]]=true;
                }
            }
        }
    }


    // DPG
    void IndexComponentConnDPG::ConnInner() {
        // ofstream os(hubs_path, ios::binary);
        // vector <hub_pair > hubs;
        std::vector<std::vector<unsigned>> rknn_graph;
        rknn_graph.resize(index_->n_);

        int count = 0;

        for (unsigned i = 0; i < index_->n_; ++i) {
            auto const &knn = index_->final_graph_[i];
            //uint32_t K = M[i]; // knn.size();
            for (unsigned j = 0; j < knn.size(); j++) {
                rknn_graph[knn[j]].push_back(i);
            }
        }

        for (unsigned i = 0; i < index_->n_; ++i) {
            std::vector<unsigned> rknn_list = rknn_graph[i];
            count += rknn_list.size();

            for (unsigned j = 0; j < rknn_list.size(); ++j) {
                index_->final_graph_[i].push_back(rknn_list[j]);
//                graph[i].push_back(Neighbor(rknn_list[j].id, rknn_list[j].dist,
//                                            true)); // rknn_list[j]);
                // sum += exp(-1 * sqrt(rknn_list[j].dist) * beta); // a
                // function with dist
            }

            std::sort(index_->final_graph_[i].begin(), index_->final_graph_[i].end());
            for(unsigned j = 1; j < index_->final_graph_[i].size(); j ++){
                if(index_->final_graph_[i][j] == index_->final_graph_[i][j-1]){
                    index_->final_graph_[i].erase(index_->final_graph_[i].begin() + j);
                    j -- ;
                }
            }

            //M[i] = graph[i].size();
        }
        fprintf(stderr, "inverse edges: %d\n", count);
    }


}

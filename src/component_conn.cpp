//
// Created by MurphySL on 2020/10/24.
//

#include "weavess/component.h"

namespace weavess {

    // DFS
    void ComponentConnNSGDFS::ConnInner() {
        tree_grow();
    }

    void ComponentConnNSGDFS::tree_grow() {
        unsigned root = index->ep_;
        boost::dynamic_bitset<> flags{index->getBaseLen(), 0};
        unsigned unlinked_cnt = 0;
        while (unlinked_cnt < index->getBaseLen()) {
            DFS(flags, root, unlinked_cnt);
            // std::cout << unlinked_cnt << '\n';
            if (unlinked_cnt >= index->getBaseLen()) break;
            findroot(flags, root);
            // std::cout << "new root"<<":"<<root << '\n';
        }
        for (size_t i = 0; i < index->getBaseLen(); ++i) {
            if (index->getFinalGraph()[i].size() > index->width) {
                index->width = index->getFinalGraph()[i].size();
            }
        }
    }

    void ComponentConnNSGDFS::DFS(boost::dynamic_bitset<> &flag, unsigned root, unsigned &cnt) {
        unsigned tmp = root;
        std::stack<unsigned> s;
        s.push(root);
        if (!flag[root]) cnt++;
        flag[root] = true;
        while (!s.empty()) {
            unsigned next = index->getBaseLen() + 1;
            for (unsigned i = 0; i < index->getFinalGraph()[tmp].size(); i++) {
                if (!flag[index->getFinalGraph()[tmp][i].id]) {
                    next = index->getFinalGraph()[tmp][i].id;
                    break;
                }
            }
            // std::cout << next <<":"<<cnt <<":"<<tmp <<":"<<s.size()<< '\n';
            if (next == (index->getBaseLen() + 1)) {
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

    void ComponentConnNSGDFS::findroot(boost::dynamic_bitset<> &flag, unsigned &root) {
        unsigned id = index->getBaseLen();
        for (unsigned i = 0; i < index->getBaseLen(); i++) {
            if (flag[i] == false) {
                id = i;
                break;
            }
        }

        if (id == index->getBaseLen()) return;  // No Unlinked Node

        std::vector<Index::Neighbor> tmp, pool;
        get_neighbors(index->getBaseData() + index->getBaseDim() * id, tmp, pool);
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
                unsigned rid = rand() % index->getBaseLen();
                if (flag[rid]) {
                    root = rid;
                    break;
                }
            }
        }
        float dist = index->getDist()->compare(index->getBaseData() + index->getBaseDim() * root,
                                         index->getBaseData() + index->getBaseDim() * id,
                                         index->getBaseDim());
        index->getFinalGraph()[root].push_back(Index::SimpleNeighbor(id, dist));
    }

    void ComponentConnNSGDFS::get_neighbors(const float *query, std::vector<Index::Neighbor> &retset,
                                            std::vector<Index::Neighbor> &fullset) {
        unsigned L = index->getParam().get<unsigned>("L_refine");

        retset.resize(L + 1);
        std::vector<unsigned> init_ids(L);
        // initializer_->Search(query, nullptr, L, parameter, init_ids.data());

        boost::dynamic_bitset<> flags{index->getBaseLen(), 0};
        L = 0;
        for (unsigned i = 0; i < init_ids.size() && i < index->getFinalGraph()[index->ep_].size(); i++) {
            init_ids[i] = index->getFinalGraph()[index->ep_][i].id;
            flags[init_ids[i]] = true;
            L++;
        }
        while (L < init_ids.size()) {
            unsigned id = rand() % index->getBaseLen();
            if (flags[id]) continue;
            init_ids[L] = id;
            L++;
            flags[id] = true;
        }

        L = 0;
        for (unsigned i = 0; i < init_ids.size(); i++) {
            unsigned id = init_ids[i];
            if (id >= index->getBaseLen()) continue;
            // std::cout<<id<<std::endl;
            float dist = index->getDist()->compare(index->getBaseData() + index->getBaseDim() * (size_t) id, query,
                                                   (unsigned) index->getBaseDim());
            retset[i] = Index::Neighbor(id, dist, true);
            // flags[id] = 1;
            L++;
        }

        std::sort(retset.begin(), retset.begin() + L);
        int k = 0;
        while (k < (int) L) {
            int nk = L;

            if (retset[k].flag) {
                retset[k].flag = false;
                unsigned n = retset[k].id;

                for (unsigned m = 0; m < index->getFinalGraph()[n].size(); ++m) {
                    unsigned id = index->getFinalGraph()[n][m].id;
                    if (flags[id]) continue;
                    flags[id] = 1;

                    float dist = index->getDist()->compare(query,
                                                           index->getBaseData() + index->getBaseDim() * (size_t) id,
                                                           (unsigned) index->getBaseDim());
                    Index::Neighbor nn(id, dist, true);
                    fullset.push_back(nn);
                    if (dist >= retset[L - 1].distance) continue;
                    int r = Index::InsertIntoPool(retset.data(), L, nn);

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

    // DFS EXPAND
    void ComponentConnSSGDFS::ConnInner() {
        unsigned n_try = index->getParam().get<unsigned>("n_try");
        unsigned range = index->getParam().get<unsigned>("R_refine");

        std::vector<unsigned> ids(index->getBaseLen());
        for (unsigned i = 0; i < index->getBaseLen(); i++) {
            ids[i] = i;
        }
        std::random_shuffle(ids.begin(), ids.end());
        for (unsigned i = 0; i < n_try; i++) {
            index->eps_.push_back(ids[i]);
        }
#pragma omp parallel for
        for(unsigned i=0; i<n_try; i++){
            unsigned rootid = index->eps_[i];
            boost::dynamic_bitset<> flags{index->getBaseLen(), 0};
            std::queue<unsigned> myqueue;
            myqueue.push(rootid);
            flags[rootid]=true;

            std::vector<unsigned> uncheck_set(1);

            while(uncheck_set.size() >0){
                while(!myqueue.empty()){
                    unsigned q_front=myqueue.front();
                    myqueue.pop();

                    for(unsigned j=0; j<index->getFinalGraph()[q_front].size(); j++){
                        unsigned child = index->getFinalGraph()[q_front][j].id;
                        if(flags[child])continue;
                        flags[child] = true;
                        myqueue.push(child);
                    }
                }

                uncheck_set.clear();
                for(unsigned j=0; j<index->getBaseLen(); j++){
                    if(flags[j])continue;
                    uncheck_set.push_back(j);
                }
                //std::cout <<i<<":"<< uncheck_set.size() << '\n';
                if(uncheck_set.size()>0){
                    for(unsigned j=0; j<index->getBaseLen(); j++){
                        if(flags[j] && index->getFinalGraph()[j].size()<range){
                            float dist = index->getDist()->compare(index->getBaseData() + index->getBaseDim() * j,
                                                                   index->getBaseData() + index->getBaseDim() * uncheck_set[0],
                                                                   index->getBaseDim());
                            index->getFinalGraph()[j].push_back(Index::SimpleNeighbor(uncheck_set[0], dist));
                            break;
                        }
                    }
                    myqueue.push(uncheck_set[0]);
                    flags[uncheck_set[0]]=true;
                }
            }
        }
    }

    // REVERSE
    void ComponentConnReverse::ConnInner() {
        //ofstream os(hubs_path, ios::binary);
        //vector <hub_pair > hubs;
        std::vector<std::vector<Index::SimpleNeighbor>> rknn_graph;
        rknn_graph.resize(index->getBaseLen());

        int count = 0;

        for (unsigned i = 0; i < index->getBaseLen(); ++i)
        {
            auto const &knn = index->getFinalGraph()[i];
            uint32_t K = index->getFinalGraph()[i].size(); //knn.size();
            for (unsigned j = 0; j < K; j++)
            {
                rknn_graph[knn[j].id].push_back(Index::SimpleNeighbor(i, knn[j].distance));
            }
        }

        for (unsigned i = 0; i < index->getBaseLen(); ++i)
        {
            std::vector<Index::SimpleNeighbor> rknn_list = rknn_graph[i];
            count += rknn_list.size();

            for (unsigned j = 0; j < rknn_list.size(); ++j)
            {
                index->getFinalGraph()[i].push_back(rknn_list[j]);
                //sum += exp(-1 * sqrt(rknn_list[j].dist) * beta); // a function with dist
            }

            sort(index->getFinalGraph()[i].begin(), index->getFinalGraph()[i].end());
            for (unsigned j = 1; j < index->getFinalGraph()[i].size(); ++j)
            {
                if (index->getFinalGraph()[i][j].id == index->getFinalGraph()[i][j - 1].id)
                {
                    index->getFinalGraph()[i].erase(index->getFinalGraph()[i].begin() + j);
                    j--;
                }
            }
        }
        // fprintf(stderr, "inverse edges: %d\n", count);
    }
}
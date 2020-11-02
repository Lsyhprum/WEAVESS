//
// Created by MurphySL on 2020/10/23.
//

#include "weavess/component.h"

namespace weavess {

    /**
     * 贪婪搜索
     * @param query 查询点
     * @param pool 侯选池
     * @param res 结果集
     */
    void ComponentSearchRouteGreedy::RouteInner(unsigned int query, std::vector<Index::Neighbor> &pool,
                                                std::vector<unsigned int> &res) {
        const auto L = index->getParam().get<unsigned>("L_search");
        const auto K = index->getParam().get<unsigned>("K_search");

        std::vector<char> flags(index->getBaseLen());

        int k = 0;
        while (k < (int) L) {
            int nk = L;

            if (pool[k].flag) {
                pool[k].flag = false;
                unsigned n = pool[k].id;

                // 查找邻居的邻居
                for (unsigned m = 0; m < index->getFinalGraph()[n].size(); ++m) {
                    unsigned id = index->getFinalGraph()[n][m].id;

                    if (flags[id])continue;
                    flags[id] = 1;

                    float dist = index->getDist()->compare(index->getQueryData() + index->getQueryDim() * query,
                                                           index->getBaseData() + index->getBaseDim() * id,
                                                           (unsigned) index->getBaseDim());
                    index->addDistCount();

                    if (dist >= pool[L - 1].distance) continue;
                    Index::Neighbor nn(id, dist, true);
                    int r = Index::InsertIntoPool(pool.data(), L, nn);

                    //if(L+1 < retset.size()) ++L;
                    if (r < nk)nk = r;
                }
                //lock to here
            }
            if (nk <= k)k = nk;
            else ++k;
        }

        for (size_t i = 0; i < K; i++) {
            res[i] = pool[i].id;
        }
    }


    /**
     * NSW 搜索
     * @param query 查询点
     * @param pool
     * @param res 结果集
     */
    void ComponentSearchRouteNSW::RouteInner(unsigned int query, std::vector<Index::Neighbor> &pool,
                                              std::vector<unsigned int> &res) {
        const auto K = index->getParam().get<unsigned>("K_search");

        auto *visited_list = new Index::VisitedList(index->getBaseLen());

        Index::HnswNode *enterpoint = index->enterpoint_;
        std::priority_queue<Index::FurtherFirst> result;
        std::priority_queue<Index::CloserFirst> tmp;

        SearchAtLayer(query, enterpoint, 0, visited_list, result);

        while(!result.empty()) {
            tmp.push(Index::CloserFirst(result.top().GetNode(), result.top().GetDistance()));
            result.pop();
        }

        int pos = 0;
        while (!tmp.empty() && pos < K) {
            auto *top_node = tmp.top().GetNode();
            tmp.pop();
            res[pos] = top_node->GetId();
            pos ++;
        }

        delete visited_list;
    }

    void ComponentSearchRouteNSW::SearchAtLayer(unsigned qnode, Index::HnswNode *enterpoint, int level,
                                         Index::VisitedList *visited_list,
                                         std::priority_queue<Index::FurtherFirst> &result) {
        const auto L = index->getParam().get<unsigned>("L_search");

        // TODO: check Node 12bytes => 8bytes
        std::priority_queue<Index::CloserFirst> candidates;
        float d = index->getDist()->compare(index->getQueryData() + qnode * index->getQueryDim(),
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
            for (const auto &neighbor : neighbors) {
                int id = neighbor->GetId();
                if (visited_list->NotVisited(id)) {
                    visited_list->MarkAsVisited(id);
                    d = index->getDist()->compare(index->getQueryData() + qnode * index->getQueryDim(),
                                                  index->getBaseData() + neighbor->GetId() * index->getBaseDim(),
                                                  index->getBaseDim());
                    if (result.size() < L || result.top().GetDistance() > d) {
                        result.emplace(neighbor, d);
                        candidates.emplace(neighbor, d);
                        if (result.size() > L)
                            result.pop();
                    }
                }
            }
        }


    }


    /**
     * HNSW 搜索
     * @param query 查询点
     * @param pool
     * @param res 结果集
     */
    void ComponentSearchRouteHNSW::RouteInner(unsigned int query, std::vector<Index::Neighbor> &pool,
                                              std::vector<unsigned int> &res) {
        const auto K = index->getParam().get<unsigned>("K_search");
        const auto L = index->getParam().get<unsigned>("L_search");

        auto *visited_list = new Index::VisitedList(index->getBaseLen());

        Index::HnswNode *enterpoint = index->enterpoint_;
        std::vector<std::pair<Index::HnswNode *, float>> ensure_k_path_;

        Index::HnswNode *cur_node = enterpoint;

        float d = index->getDist()->compare(index->getQueryData() + query * index->getQueryDim(),
                                            index->getBaseData() + cur_node->GetId() * index->getBaseDim(),
                                            index->getBaseDim());
        float cur_dist = d;

        ensure_k_path_.clear();
        ensure_k_path_.emplace_back(cur_node, cur_dist);

        for (auto i = index->max_level_; i >= 0; --i) {
            visited_list->Reset();
            unsigned visited_mark = visited_list->GetVisitMark();
            unsigned int* visited = visited_list->GetVisited();
            visited[cur_node->GetId()] = visited_mark;

            bool changed = true;
            while (changed) {
                changed = false;
                std::unique_lock<std::mutex> local_lock(cur_node->GetAccessGuard());
                const std::vector<Index::HnswNode *> &neighbors = cur_node->GetFriends(i);

                for (auto iter = neighbors.begin(); iter != neighbors.end(); ++iter) {
                    if(visited[(*iter)->GetId()] != visited_mark) {
                        visited[(*iter)->GetId()] = visited_mark;
                        d = index->getDist()->compare(index->getQueryData() + query * index->getQueryDim(),
                                                      index->getBaseData() + (*iter)->GetId() * index->getBaseDim(),
                                                      index->getBaseDim());

                        if (d < cur_dist) {
                            cur_dist = d;
                            cur_node = *iter;
                            changed = true;
                            ensure_k_path_.emplace_back(cur_node, cur_dist);
                        }
                    }
                }
            }
        }

        //std::cout << "ensure_k : " << ensure_k_path_.size() << " " << ensure_k_path_[0].first->GetId() << std::endl;

        std::vector<std::pair<Index::HnswNode*, float>> tmp;

        while(tmp.size() < K && !ensure_k_path_.empty()) {
            cur_dist = ensure_k_path_.back().second;
            //std::cout << ensure_k_path_.back().first->GetId() << " ";
            ensure_k_path_.pop_back();
            SearchById_(query, ensure_k_path_.back().first, cur_dist, K, L, tmp);
        }
        //std::cout << std::endl;

        for(auto ret : tmp) {
//            std::cout << ret.first << "";
            res.push_back(ret.first->GetId());
        }
//        std::cout << std::endl;

        delete visited_list;
    }

    void ComponentSearchRouteHNSW::SearchById_(unsigned query, Index::HnswNode* cur_node, float cur_dist, size_t k,
                                                   size_t ef_search, std::vector<std::pair<Index::HnswNode*, float>> &result) {
        Index::IdDistancePairMinHeap candidates;
        Index::IdDistancePairMinHeap visited_nodes;

        candidates.emplace(cur_node, cur_dist);

        auto *visited_list_ = new Index::VisitedList(index->getBaseLen());

        visited_list_->Reset();
        unsigned int visited_mark = visited_list_->GetVisitMark();
        unsigned int* visited = visited_list_->GetVisited();

        size_t already_visited_for_ensure_k = 0;
        if (!result.empty()) {
            already_visited_for_ensure_k = result.size();
            for (size_t i = 0; i < result.size(); ++i) {
                if (result[i].first->GetId() == cur_node->GetId()) {
                    return ;
                }
                visited[result[i].first->GetId()] = visited_mark;
                visited_nodes.emplace(std::move(result[i]));
            }
            result.clear();
        }
        visited[cur_node->GetId()] = visited_mark;
        //std::cout << "wtf" << std::endl;

        float farthest_distance = cur_dist;
        size_t total_size = 1;
        while (!candidates.empty() && visited_nodes.size() < ef_search+already_visited_for_ensure_k) {
            //std::cout << "wtf1" << std::endl;
            const Index::IdDistancePair& c = candidates.top();
            cur_node = c.first;
            visited_nodes.emplace(std::move(const_cast<Index::IdDistancePair&>(c)));
            candidates.pop();

            float minimum_distance = farthest_distance;
            int size = cur_node->GetFriends(0).size();

            //std::cout << "wtf2" << std::endl;
            for (auto j = 1; j < size; ++j) {
                int node_id = cur_node->GetFriends(0)[j]->GetId();
                //std::cout << "wtf4" << std::endl;
                if (visited[node_id] != visited_mark) {
                    visited[node_id] = visited_mark;
                    //std::cout << "wtf5" << std::endl;
                    float d = index->getDist()->compare(index->getQueryData() + index->getQueryDim() * query,
                                                        index->getBaseData() + index->getBaseDim() * node_id,
                                                        index->getBaseDim());
                    //std::cout << "wtf6" << std::endl;
                    if (d < minimum_distance || total_size < ef_search) {
                        candidates.emplace(cur_node->GetFriends(0)[j], d);
                        if (d > farthest_distance) {
                            farthest_distance = d;
                        }
                        ++total_size;
                    }
                    //std::cout << "wtf7" << std::endl;
                }
            }
            //std::cout << "wtf3" << std::endl;
        }

        //std::cout << "wtf" <<std::endl;

        while (result.size() < k) {
            if (!candidates.empty() && !visited_nodes.empty()) {
                const Index::IdDistancePair& c = candidates.top();
                const Index::IdDistancePair& v = visited_nodes.top();
                if (c.second < v.second) {
                    result.emplace_back(std::move(const_cast<Index::IdDistancePair&>(c)));
                    candidates.pop();
                } else {
                    result.emplace_back(std::move(const_cast<Index::IdDistancePair&>(v)));
                    visited_nodes.pop();
                }
            } else if (!candidates.empty()) {
                const Index::IdDistancePair& c = candidates.top();
                result.emplace_back(std::move(const_cast<Index::IdDistancePair&>(c)));
                candidates.pop();
            } else if (!visited_nodes.empty()) {
                const Index::IdDistancePair& v = visited_nodes.top();
                result.emplace_back(std::move(const_cast<Index::IdDistancePair&>(v)));
                visited_nodes.pop();
            } else {
                break;
            }
        }
    }


    /**
     * IEH 搜索
     * @param query 查询点
     * @param pool
     * @param res
     */
    void ComponentSearchRouteIEH::RouteInner(unsigned int query, std::vector<Index::Neighbor> &pool,
                                             std::vector<unsigned int> &result) {

        const auto L = index->getParam().get<unsigned>("L_search");
        const auto K = index->getParam().get<unsigned>("K_search");

        //GNNS
        Index::CandidateHeap2 cands;
        for (size_t j = 0; j < pool.size(); j++) {
            int neighbor = pool[j].id;
            Index::Candidate2<float> c(neighbor,
                                       index->getDist()->compare(&index->test[query][0], &index->train[neighbor][0], index->test[query].size()));
            cands.insert(c);
            if (cands.size() > L)cands.erase(cands.begin());
        }

        //iteration
        auto expand = index->getParam().get<unsigned>("expand");
        auto iterlimit = index->getParam().get<unsigned>("iterlimit");

        int niter = 0;
        while (niter++ < iterlimit) {
            auto it = cands.rbegin();
            std::vector<int> ids;
            for (int j = 0; it != cands.rend() && j < expand; it++, j++) {
                int neighbor = it->row_id;
                auto nnit = index->knntable[neighbor].rbegin();
                for (int k = 0; nnit != index->knntable[neighbor].rend() && k < expand; nnit++, k++) {
                    int nn = nnit->row_id;
                    ids.push_back(nn);
                }
            }
            for (size_t j = 0; j < ids.size(); j++) {
                Index::Candidate2<float> c(ids[j], index->getDist()->compare(&index->test[query][0], &index->train[ids[j]][0],
                                                                             index->test[query].size()));
                cands.insert(c);
                if (cands.size() > L)cands.erase(cands.begin());
            }
        }//cout<<i<<endl;

        auto it = cands.rbegin();
        for(int j = 0; it != cands.rend() && j < K; it++, j++){
            result.push_back(it->row_id);
        }
    }


    /**
     * Backtrack 搜索
     * @param query 查询点
     * @param pool 入口点
     * @param res 结果集
     */
    void ComponentSearchRouteBacktrack::RouteInner(unsigned int query, std::vector<Index::Neighbor> &pool,
                                                   std::vector<unsigned int> &res) {
        const auto L = index->getParam().get<unsigned>("L_search");
        const auto K = index->getParam().get<unsigned>("K_search");

        std::priority_queue<Index::FANNGCloserFirst> queue;
        std::priority_queue<Index::FANNGCloserFirst> full;
        std::vector<char> flags(index->getBaseLen());
        for(int i = 0; i < index->getBaseLen(); i ++) flags[i] = false;
        std::unordered_map<unsigned, int> mp; // 记录结点近邻访问位置
        std::unordered_map<unsigned, unsigned> relation; // 记录终止结点和起始结点关系

        unsigned start = pool[0].id;
        flags[start] = true;
        float dist = index->getDist()->compare(index->getBaseData() + index->getBaseDim() * start,
                                               index->getBaseData() + index->getBaseDim() * query,
                                               index->getBaseDim());

        int m = 1;
        queue.push(Index::FANNGCloserFirst(start, dist));
        full.push(Index::FANNGCloserFirst(start, dist));

        while(!queue.empty() && m < L) {
            unsigned top_node = queue.top().GetNode();
            mp[top_node] = 0;
            queue.pop();
            unsigned top_node_nn = index->getFinalGraph()[top_node][0].id;

            if(flags[top_node_nn] == false) {
                relation[top_node_nn] = top_node;
                flags[top_node_nn] = true;
                m += 1;
                float dist = index->getDist()->compare(index->getBaseData() + index->getBaseDim() * query,
                                                       index->getBaseData() + index->getBaseDim() * top_node_nn,
                                                       index->getBaseDim());
                queue.push(Index::FANNGCloserFirst(top_node_nn, dist));
                full.push(Index::FANNGCloserFirst(top_node_nn, dist));
            }

            unsigned start_node = relation[top_node_nn];
            bool flag = false;
            while(!flag) {
                if(mp[start_node] + 1 != index->getFinalGraph()[start_node].size() && flags[index->getFinalGraph()[start_node][mp[start_node] + 1].id] == false) {
                    flag = true;
                    break;
                }
                while(mp[start_node] + 1 == index->getFinalGraph()[start_node].size())
                    start_node = relation[start_node];
                while(flags[index->getFinalGraph()[start_node][mp[start_node] + 1].id]) {
                    mp[start_node] += 1;
                    if(mp[start_node] == index->getFinalGraph()[start_node].size()) {
                        start_node = relation[start_node];
                        break;
                    }
                }
            }
            if(flag) {
                unsigned node = index->getFinalGraph()[start_node][mp[start_node] + 1].id;
                float dist = index->getDist()->compare(index->getBaseData() + index->getBaseDim() * query,
                                                       index->getBaseData() + index->getBaseDim() * node,
                                                       index->getBaseDim());
                queue.push(Index::FANNGCloserFirst(node, dist));
                full.push(Index::FANNGCloserFirst(node, dist));
                relation[node] = start_node;
                mp[start_node] += 1;
            }
        }

        int i = 0;
        while(!full.empty() && i < K) {
            res.push_back(full.top().GetNode());
            full.pop();
            i ++;
        }
    }
}
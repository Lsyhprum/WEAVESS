//
// Created by MurphySL on 2020/10/23.
//

#include "weavess/component.h"

namespace weavess {

    void ComponentSearchRouteGreedy::RouteInner(unsigned int query, std::vector<Index::Neighbor> &pool,
                                                std::vector<unsigned int> &res) {
        const auto L = index->getParam().get<unsigned>("L_search");
        const auto K = index->getParam().get<unsigned>("K_search");

        std::vector<char> flags(index->getBaseLen(), 0);

        int k = 0;
        while (k < (int) L) {
            int nk = L;

            if (pool[k].flag) {
                pool[k].flag = false;
                unsigned n = pool[k].id;

                index->addHopCount();
                for (unsigned m = 0; m < index->getLoadGraph()[n].size(); ++m) {
                    unsigned id = index->getLoadGraph()[n][m];

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

        res.resize(K);
        for (size_t i = 0; i < K; i++) {
            res[i] = pool[i].id;
        }
    }


    void ComponentSearchRouteNSW::RouteInner(unsigned int query, std::vector<Index::Neighbor> &pool,
                                              std::vector<unsigned int> &res) {
        const auto K = index->getParam().get<unsigned>("K_search");

        auto *visited_list = new Index::VisitedList(index->getBaseLen());

        Index::HnswNode *enterpoint = index->nodes_[0];
        std::priority_queue<Index::FurtherFirst> result;
        std::priority_queue<Index::CloserFirst> tmp;

        SearchAtLayer(query, enterpoint, 0, visited_list, result);

        while(!result.empty()) {
            tmp.push(Index::CloserFirst(result.top().GetNode(), result.top().GetDistance()));
            result.pop();
        }

        res.resize(K);
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
        index->addDistCount();
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
            index->addHopCount();
            for (const auto &neighbor : neighbors) {
                int id = neighbor->GetId();
                if (visited_list->NotVisited(id)) {
                    visited_list->MarkAsVisited(id);
                    d = index->getDist()->compare(index->getQueryData() + qnode * index->getQueryDim(),
                                                  index->getBaseData() + neighbor->GetId() * index->getBaseDim(),
                                                  index->getBaseDim());
                    index->addDistCount();
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


    void ComponentSearchRouteHNSW::RouteInner(unsigned int query, std::vector<Index::Neighbor> &pool,
                                              std::vector<unsigned int> &res) {
        const auto K = index->getParam().get<unsigned>("K_search");
        // const auto L = index->getParam().get<unsigned>("L_search");

        auto *visited_list = new Index::VisitedList(index->getBaseLen());

        Index::HnswNode *enterpoint = index->enterpoint_;
        std::vector<std::pair<Index::HnswNode *, float>> ensure_k_path_;

        Index::HnswNode *cur_node = enterpoint;

        float d = index->getDist()->compare(index->getQueryData() + query * index->getQueryDim(),
                                            index->getBaseData() + cur_node->GetId() * index->getBaseDim(),
                                            index->getBaseDim());
        index->addDistCount();
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

                index->addHopCount();
                for (auto iter = neighbors.begin(); iter != neighbors.end(); ++iter) {
                    if(visited[(*iter)->GetId()] != visited_mark) {
                        visited[(*iter)->GetId()] = visited_mark;
                        d = index->getDist()->compare(index->getQueryData() + query * index->getQueryDim(),
                                                      index->getBaseData() + (*iter)->GetId() * index->getBaseDim(),
                                                      index->getBaseDim());
                        index->addDistCount();
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

        // std::vector<std::pair<Index::HnswNode*, float>> tmp;
        std::priority_queue<Index::FurtherFirst> result;
        std::priority_queue<Index::CloserFirst> tmp;

        while(result.size() < K && !ensure_k_path_.empty()) {
            cur_dist = ensure_k_path_.back().second;
            ensure_k_path_.pop_back();
            SearchAtLayer(query, ensure_k_path_.back().first, 0, visited_list, result);
        }
        while(!result.empty()) {
            tmp.push(Index::CloserFirst(result.top().GetNode(), result.top().GetDistance()));
            result.pop();
        }

        res.resize(K);
        int pos = 0;
        while (!tmp.empty() && pos < K) {
            auto *top_node = tmp.top().GetNode();
            tmp.pop();
            res[pos] = top_node->GetId();
            pos ++;
        }

        delete visited_list;
    }

    void ComponentSearchRouteHNSW::SearchAtLayer(unsigned qnode, Index::HnswNode *enterpoint, int level,
                                         Index::VisitedList *visited_list,
                                         std::priority_queue<Index::FurtherFirst> &result) {
        const auto L = index->getParam().get<unsigned>("L_search");

        // TODO: check Node 12bytes => 8bytes
        std::priority_queue<Index::CloserFirst> candidates;
        float d = index->getDist()->compare(index->getQueryData() + qnode * index->getQueryDim(),
                                            index->getBaseData() + enterpoint->GetId() * index->getBaseDim(),
                                            index->getBaseDim());
        index->addDistCount();
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
            index->addHopCount();
            for (const auto &neighbor : neighbors) {
                int id = neighbor->GetId();
                if (visited_list->NotVisited(id)) {
                    visited_list->MarkAsVisited(id);
                    d = index->getDist()->compare(index->getQueryData() + qnode * index->getQueryDim(),
                                                  index->getBaseData() + neighbor->GetId() * index->getBaseDim(),
                                                  index->getBaseDim());
                    index->addDistCount();
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

    // void ComponentSearchRouteHNSW::SearchById_(unsigned query, Index::HnswNode* cur_node, float cur_dist, size_t k,
    //                                                size_t ef_search, std::vector<std::pair<Index::HnswNode*, float>> &result) {
    //     Index::IdDistancePairMinHeap candidates;
    //     Index::IdDistancePairMinHeap visited_nodes;

    //     candidates.emplace(cur_node, cur_dist);

    //     auto *visited_list_ = new Index::VisitedList(index->getBaseLen());

    //     visited_list_->Reset();
    //     unsigned int visited_mark = visited_list_->GetVisitMark();
    //     unsigned int* visited = visited_list_->GetVisited();

    //     size_t already_visited_for_ensure_k = 0;
    //     if (!result.empty()) {
    //         already_visited_for_ensure_k = result.size();
    //         for (size_t i = 0; i < result.size(); ++i) {
    //             if (result[i].first->GetId() == cur_node->GetId()) {
    //                 return ;
    //             }
    //             visited[result[i].first->GetId()] = visited_mark;
    //             visited_nodes.emplace(std::move(result[i]));
    //         }
    //         result.clear();
    //     }
    //     visited[cur_node->GetId()] = visited_mark;
    //     //std::cout << "wtf" << std::endl;

    //     float farthest_distance = cur_dist;
    //     size_t total_size = 1;
    //     while (!candidates.empty() && visited_nodes.size() < ef_search+already_visited_for_ensure_k) {
    //         //std::cout << "wtf1" << std::endl;
    //         const Index::IdDistancePair& c = candidates.top();
    //         cur_node = c.first;
    //         visited_nodes.emplace(std::move(const_cast<Index::IdDistancePair&>(c)));
    //         candidates.pop();

    //         float minimum_distance = farthest_distance;
    //         int size = cur_node->GetFriends(0).size();

    //         index->addHopCount();
    //         //std::cout << "wtf2" << std::endl;
    //         for (auto j = 1; j < size; ++j) {
    //             int node_id = cur_node->GetFriends(0)[j]->GetId();
    //             //std::cout << "wtf4" << std::endl;
    //             if (visited[node_id] != visited_mark) {
    //                 visited[node_id] = visited_mark;
    //                 //std::cout << "wtf5" << std::endl;
    //                 float d = index->getDist()->compare(index->getQueryData() + index->getQueryDim() * query,
    //                                                     index->getBaseData() + index->getBaseDim() * node_id,
    //                                                     index->getBaseDim());
    //                 index->addDistCount();
    //                 //std::cout << "wtf6" << std::endl;
    //                 if (d < minimum_distance || total_size < ef_search) {
    //                     candidates.emplace(cur_node->GetFriends(0)[j], d);
    //                     if (d > farthest_distance) {
    //                         farthest_distance = d;
    //                     }
    //                     ++total_size;
    //                 }
    //                 //std::cout << "wtf7" << std::endl;
    //             }
    //         }
    //         //std::cout << "wtf3" << std::endl;
    //     }

    //     //std::cout << "wtf" <<std::endl;

    //     while (result.size() < k) {
    //         if (!candidates.empty() && !visited_nodes.empty()) {
    //             const Index::IdDistancePair& c = candidates.top();
    //             const Index::IdDistancePair& v = visited_nodes.top();
    //             if (c.second < v.second) {
    //                 result.emplace_back(std::move(const_cast<Index::IdDistancePair&>(c)));
    //                 candidates.pop();
    //             } else {
    //                 result.emplace_back(std::move(const_cast<Index::IdDistancePair&>(v)));
    //                 visited_nodes.pop();
    //             }
    //         } else if (!candidates.empty()) {
    //             const Index::IdDistancePair& c = candidates.top();
    //             result.emplace_back(std::move(const_cast<Index::IdDistancePair&>(c)));
    //             candidates.pop();
    //         } else if (!visited_nodes.empty()) {
    //             const Index::IdDistancePair& v = visited_nodes.top();
    //             result.emplace_back(std::move(const_cast<Index::IdDistancePair&>(v)));
    //             visited_nodes.pop();
    //         } else {
    //             break;
    //         }
    //     }
    // }


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
            index->addDistCount();
            cands.insert(c);
            if (cands.size() > L)cands.erase(cands.begin());
        }

        //iteration
        // auto expand = index->getParam().get<unsigned>("expand");
        auto iterlimit = index->getParam().get<unsigned>("iterlimit");

        int niter = 0;
        while (niter++ < iterlimit) {
            auto it = cands.rbegin();
            std::vector<int> ids;
            index->addHopCount();
            for (int j = 0; it != cands.rend() && j < L; it++, j++) {
                int neighbor = it->row_id;
                auto nnit = index->knntable[neighbor].rbegin();
                for (int k = 0; nnit != index->knntable[neighbor].rend(); nnit++, k++) {
                    int nn = nnit->row_id;
                    ids.push_back(nn);
                }
            }
            for (size_t j = 0; j < ids.size(); j++) {
                Index::Candidate2<float> c(ids[j], index->getDist()->compare(&index->test[query][0], &index->train[ids[j]][0],
                                                                             index->test[query].size()));
                index->addDistCount();
                cands.insert(c);
                if (cands.size() > L)cands.erase(cands.begin());
            }
        }//cout<<i<<endl;

        auto it = cands.rbegin();
        for(int j = 0; it != cands.rend() && j < K; it++, j++){
            result.push_back(it->row_id);
        }
    }


    void ComponentSearchRouteBacktrack::RouteInner(unsigned int query, std::vector<Index::Neighbor> &pool,
                                                   std::vector<unsigned int> &res) {
        const auto L = index->getParam().get<unsigned>("L_search");
        const auto K = index->getParam().get<unsigned>("K_search");

        std::priority_queue<Index::FANNGCloserFirst> queue;
        std::priority_queue<Index::FANNGCloserFirst> full;
        std::vector<char> flags(index->getBaseLen());
        for(int i = 0; i < index->getBaseLen(); i ++) flags[i] = false;
        std::unordered_map<unsigned, int> mp;
        std::unordered_map<unsigned, unsigned> relation;

        unsigned enter = pool[0].id;
        unsigned start = index->getLoadGraph()[enter][0];
        relation[start] = enter;
        mp[enter] = 0;
        float dist = index->getDist()->compare(index->getBaseData() + index->getBaseDim() * start,
                                               index->getQueryData() + index->getQueryDim() * query,
                                               index->getBaseDim());

        index->addDistCount();
        int m = 1;
        queue.push(Index::FANNGCloserFirst(start, dist));
        full.push(Index::FANNGCloserFirst(start, dist));

        while(!queue.empty() && m < L) {
            //std::cout << 1 << std::endl;
            unsigned top_node = queue.top().GetNode();
            queue.pop();

            if(!flags[top_node]) {
                flags[top_node] = true;

                unsigned nnid = index->getLoadGraph()[top_node][0];
                relation[nnid] = top_node;
                mp[top_node] = 0;
                m += 1;
                float dist = index->getDist()->compare(index->getQueryData() + index->getQueryDim() * query,
                                                       index->getBaseData() + index->getBaseDim() * nnid,
                                                       index->getBaseDim());
                index->addDistCount();
                queue.push(Index::FANNGCloserFirst(nnid, dist));
                full.push(Index::FANNGCloserFirst(nnid, dist));
            }
            //std::cout << 2 << std::endl;

            unsigned start_node = relation[top_node];

            //std::cout << 3 << " " << start_node << std::endl;

            unsigned pos = 0;
            auto iter = mp.find(start_node);
            //std::cout << 3.11 << " " << (*iter).second << std::endl;
            //std::cout << index->getFinalGraph()[start_node].size() << std::endl;
            if((*iter).second < index->getLoadGraph()[start_node].size() - 1) {
                //std::cout << 3.1 << std::endl;
                pos = (*iter).second + 1;
                mp[start_node] = pos;
                unsigned nnid = index->getLoadGraph()[start_node][pos];
                //std::cout << 3.2 << " " << nnid << std::endl;
                relation[nnid] = start_node;
                float dist = index->getDist()->compare(index->getQueryData() + index->getQueryDim() * query,
                                                      index->getBaseData() + index->getBaseDim() * nnid,
                                                      index->getBaseDim());
                index->addDistCount();
                //std::cout << 3.3 << std::endl;
                queue.push(Index::FANNGCloserFirst(nnid, dist));
                full.push(Index::FANNGCloserFirst(nnid, dist));
            }
            index->addHopCount();

            //std::cout << 4 << std::endl;
        }

        int i = 0;
        while(!full.empty() && i < K) {
            res.push_back(full.top().GetNode());
            full.pop();
            i ++;
        }
        //std::cout << 5 << std::endl;
    }


    void ComponentSearchRouteGuided::RouteInner(unsigned int query, std::vector<Index::Neighbor> &pool,
                                                std::vector<unsigned int> &res) {
        const auto L = index->getParam().get<unsigned>("L_search");
        const auto K = index->getParam().get<unsigned>("K_search");

        std::vector<char> flags(index->getBaseLen(), 0);

        int k = 0;
        while (k < (int)L) {
            int nk = L;

            if (pool[k].flag) {
                pool[k].flag = false;
                unsigned n = pool[k].id;

                unsigned div_dim_ = index->Tn[n].div_dim;
                unsigned left_len = index->Tn[n].left.size();
                // std::cout << "left_len: " << left_len << std::endl;
                unsigned right_len = index->Tn[n].right.size();
                // std::cout << "right_len: " << right_len << std::endl;
                std::vector<unsigned> nn;
                unsigned MaxM;
                if ((index->getQueryData() + index->getQueryDim() * query)[div_dim_] < (index->getBaseData() + index->getBaseDim() * n)[div_dim_]) {
                    MaxM = left_len;
                    nn = index->Tn[n].left;
                }
                else {
                    MaxM = right_len;
                    nn = index->Tn[n].right;
                }

                index->addHopCount();
                for (unsigned m = 0; m < MaxM; ++m) {
                    unsigned id = nn[m];
                    if (flags[id]) continue;
                    flags[id] = 1;
                    float dist = index->getDist()->compare(index->getQueryData() + query * index->getQueryDim(),
                                                           index->getBaseData() + id * index->getBaseDim(),
                                                           (unsigned)index->getBaseDim());
                    index->addDistCount();
                    if (dist >= pool[L - 1].distance) continue;
                    Index::Neighbor nn(id, dist, true);
                    int r = Index::InsertIntoPool(pool.data(), L, nn);

                    // if(L+1 < retset.size()) ++L;
                    if (r < nk) nk = r;
                }
            }
            if (nk <= k)
                k = nk;
            else
                ++k;
        }

//        for(int i = 0; i < pool.size(); i ++) {
//            std::cout << pool[i].id << "|" << pool[i].distance << " ";
//        }
//        std::cout << std::endl;
//        std::cout << std::endl;

        res.resize(K);
        for (size_t i = 0; i < K; i++) {
            res[i] = pool[i].id;
        }
    }


    void ComponentSearchRouteSPTAG_KDT::KDTSearch(unsigned int query, int node, Index::Heap &m_NGQueue,
                                                  Index::Heap &m_SPTQueue, Index::OptHashPosVector &nodeCheckStatus,
                                                  unsigned int &m_iNumberOfCheckedLeaves,
                                                  unsigned int &m_iNumberOfTreeCheckedLeaves) {
        if (node < 0)
        {
            int tmp = -node - 1;
            if (tmp >= index->getBaseLen()) return;
            if (nodeCheckStatus.CheckAndSet(tmp)) return;

            ++m_iNumberOfTreeCheckedLeaves;
            ++m_iNumberOfCheckedLeaves;
            m_NGQueue.insert(Index::HeapCell(tmp, index->getDist()->compare(index->getQueryData() + index->getQueryDim() * query,
                                                                            index->getBaseData() + index->getBaseDim() * tmp,
                                                                            index->getBaseDim())));
            index->addDistCount();
            return;
        }

        auto& tnode = index->m_pKDTreeRoots[node];

        float distBound = 0;
        float diff = (index->getQueryData() + index->getQueryDim() * query)[tnode.split_dim] - tnode.split_value;
        float distanceBound = distBound + diff * diff;
        int otherChild, bestChild;
        if (diff < 0)
        {
            bestChild = tnode.left;
            otherChild = tnode.right;
        }
        else
        {
            otherChild = tnode.left;
            bestChild = tnode.right;
        }

        m_SPTQueue.insert(Index::HeapCell(otherChild, distanceBound));
        KDTSearch(query, bestChild, m_NGQueue, m_SPTQueue, nodeCheckStatus, m_iNumberOfCheckedLeaves, m_iNumberOfTreeCheckedLeaves);
    }

    void ComponentSearchRouteSPTAG_KDT::RouteInner(unsigned int query, std::vector<Index::Neighbor> &pool,
                                                   std::vector<unsigned int> &result) {

        const auto L = index->getParam().get<unsigned>("L_search");
        const auto K = index->getParam().get<unsigned>("K_search");

        unsigned m_iNumberOfCheckedLeaves = 0;
        unsigned m_iNumberOfTreeCheckedLeaves = 0;
        unsigned m_iNumOfContinuousNoBetterPropagation = 0;
        const float MaxDist = (std::numeric_limits<float>::max)();
        unsigned m_iThresholdOfNumberOfContinuousNoBetterPropagation = 3;
        unsigned m_iNumberOfOtherDynamicPivots = 4;
        unsigned m_iNumberOfInitialDynamicPivots = 50;
        unsigned maxCheck = index->m_iMaxCheckForRefineGraph > index->m_iMaxCheck ? index->m_iMaxCheckForRefineGraph : index->m_iMaxCheck;

        // Prioriy queue used for neighborhood graph
        Index::Heap m_NGQueue;
        m_NGQueue.Resize(maxCheck * 30);

        // Priority queue Used for Tree
        Index::Heap m_SPTQueue;
        m_SPTQueue.Resize(maxCheck * 10);

        Index::OptHashPosVector nodeCheckStatus;
        nodeCheckStatus.Init(maxCheck, index->m_iHashTableExp);
        nodeCheckStatus.CheckAndSet(query);

        Index::QueryResultSet p_query(L);

        for(int i = 0; i < index->m_iTreeNumber; i ++) {
            int node = index->m_pTreeStart[i];

            KDTSearch(query, node, m_NGQueue, m_SPTQueue, nodeCheckStatus, m_iNumberOfCheckedLeaves, m_iNumberOfTreeCheckedLeaves);
        }

        unsigned p_limits = m_iNumberOfInitialDynamicPivots;
        while (!m_SPTQueue.empty() && m_iNumberOfCheckedLeaves < p_limits)
        {
            auto& tcell = m_SPTQueue.pop();
            KDTSearch(query, tcell.node, m_NGQueue, m_SPTQueue, nodeCheckStatus, m_iNumberOfCheckedLeaves, m_iNumberOfTreeCheckedLeaves);
        }

        while (!m_NGQueue.empty()) {
            Index::HeapCell gnode = m_NGQueue.pop();
            std::vector<unsigned> node = index->getLoadGraph()[gnode.node];

            if (!p_query.AddPoint(gnode.node, gnode.distance) && m_iNumberOfCheckedLeaves > index->m_iMaxCheck) {
                p_query.SortResult();
                for(int i = 0; i < p_query.GetResultNum(); i ++) {
                    if(p_query.GetResult(i)->Dist == MaxDist) break;
                    result.emplace_back(p_query.GetResult(i)->VID);
                }

                result.resize(result.size() > K ? K : result.size());
                return;
            }
            float upperBound = std::max(p_query.worstDist(), gnode.distance);
            bool bLocalOpt = true;
            index->addHopCount();
            for (unsigned i = 0; i < node.size(); i++) {
                unsigned nn_index = node[i];
                if (nn_index < 0) break;
                if (nodeCheckStatus.CheckAndSet(nn_index)) continue;
                float distance2leaf = index->getDist()->compare(index->getQueryData() + index->getQueryDim() * query,
                                                                index->getBaseData() + index->getBaseDim() * nn_index,
                                                                index->getBaseDim());
                index->addDistCount();
                if (distance2leaf <= upperBound) bLocalOpt = false;
                m_iNumberOfCheckedLeaves++;
                m_NGQueue.insert(Index::HeapCell(nn_index, distance2leaf));
            }
            if (bLocalOpt) m_iNumOfContinuousNoBetterPropagation++;
            else m_iNumOfContinuousNoBetterPropagation = 0;
            if (m_iNumOfContinuousNoBetterPropagation > m_iThresholdOfNumberOfContinuousNoBetterPropagation) {
                if (m_iNumberOfTreeCheckedLeaves <= m_iNumberOfCheckedLeaves / 10) {
                    auto& tcell = m_SPTQueue.pop();
                    KDTSearch(query, tcell.node, m_NGQueue, m_SPTQueue, nodeCheckStatus, m_iNumberOfCheckedLeaves, m_iNumberOfTreeCheckedLeaves);
                } else if (gnode.distance > p_query.worstDist()) {
                    break;
                }
            }
        }

        p_query.SortResult();
        for(int i = 0; i < p_query.GetResultNum(); i ++) {
            if(p_query.GetResult(i)->Dist == MaxDist) break;
            result.emplace_back(p_query.GetResult(i)->VID);
//            std::cout << p_query.GetResult(i)->VID << "|" << p_query.GetResult(i)->Dist << " ";
        }
//        std::cout << std::endl;
        result.resize(result.size() > K ? K : result.size());
    }


    void ComponentSearchRouteSPTAG_BKT::RouteInner(unsigned int query, std::vector<Index::Neighbor> &pool,
                                                   std::vector<unsigned int> &result) {
        const auto L = index->getParam().get<unsigned>("L_search");
        const auto K = index->getParam().get<unsigned>("K_search");

        unsigned maxCheck = index->m_iMaxCheckForRefineGraph > index->m_iMaxCheck ? index->m_iMaxCheckForRefineGraph : index->m_iMaxCheck;
        unsigned m_iContinuousLimit = maxCheck / 64;
        const float MaxDist = (std::numeric_limits<float>::max)();

        unsigned m_iNumOfContinuousNoBetterPropagation = 0;
        unsigned m_iNumberOfCheckedLeaves = 0;
        unsigned m_iNumberOfTreeCheckedLeaves = 0;

        // Prioriy queue used for neighborhood graph
        Index::Heap m_NGQueue;
        m_NGQueue.Resize(maxCheck * 30);

        // Priority queue Used for Tree
        Index::Heap m_SPTQueue;
        m_SPTQueue.Resize(maxCheck * 10);

        Index::OptHashPosVector nodeCheckStatus;
        nodeCheckStatus.Init(maxCheck, index->m_iHashTableExp);
        nodeCheckStatus.CheckAndSet(query);

        Index::QueryResultSet p_query(L);


        for (char i = 0; i < index->m_iTreeNumber; i++) {
            const Index::BKTNode& node = index->m_pBKTreeRoots[index->m_pTreeStart[i]];
            if (node.childStart < 0) {
                float dist = index->getDist()->compare(index->getQueryData() + index->getQueryDim() * query,
                                                       index->getBaseData() + index->getBaseDim() * node.centerid,
                                                       index->getBaseDim());
                index->addDistCount();
                m_SPTQueue.insert(Index::HeapCell(index->m_pTreeStart[i], dist));
            }
            else {
                for (int begin = node.childStart; begin < node.childEnd; begin++) {
                    int tmp = index->m_pBKTreeRoots[begin].centerid;
                    float dist = index->getDist()->compare(index->getQueryData() + index->getQueryDim() * query,
                                                           index->getBaseData() + index->getBaseDim() * tmp,
                                                           index->getBaseDim());
                    index->addDistCount();
                    m_SPTQueue.insert(Index::HeapCell(begin, dist));
                }
            }
        }

        BKTSearch(query, m_NGQueue, m_SPTQueue, nodeCheckStatus, m_iNumberOfCheckedLeaves, m_iNumberOfTreeCheckedLeaves, index->m_iNumberOfInitialDynamicPivots);

        const unsigned checkPos = index->getLoadGraph()[0].size() - 1;
        while (!m_NGQueue.empty()) {
            Index::HeapCell gnode = m_NGQueue.pop();
            int tmpNode = gnode.node;
            std::vector<unsigned> node = index->getLoadGraph()[tmpNode];
            if (gnode.distance <= p_query.worstDist()) {
                int checkNode = node[checkPos];
                if (checkNode < -1) {
                    const Index::BKTNode& tnode = index->m_pBKTreeRoots[-2 - checkNode];
                    m_iNumOfContinuousNoBetterPropagation = 0;
                    p_query.AddPoint(tmpNode, gnode.distance);
                } else {
                    m_iNumOfContinuousNoBetterPropagation = 0;
                    p_query.AddPoint(tmpNode, gnode.distance);
                }
            } else {
                m_iNumOfContinuousNoBetterPropagation++;
                if (m_iNumOfContinuousNoBetterPropagation > m_iContinuousLimit || m_iNumberOfCheckedLeaves > maxCheck) {
                    p_query.SortResult();
                    for(int i = 0; i < p_query.GetResultNum(); i ++) {
                        if(p_query.GetResult(i)->Dist == MaxDist) break;
                        result.emplace_back(p_query.GetResult(i)->VID);
                    }
                    result.resize(result.size() > K ? K : result.size());
                    return;
                }
            }
            index->addHopCount();
            for (unsigned i = 0; i <= checkPos; i++) {
                int nn_index = node[i];
                if (nn_index < 0) break;
                if (nodeCheckStatus.CheckAndSet(nn_index)) continue;
                float distance2leaf = index->getDist()->compare(index->getQueryData() + index->getQueryDim() * query,
                                                                index->getBaseData() + index->getBaseDim() * nn_index,
                                                                index->getBaseDim());
                index->addDistCount();
                m_iNumberOfCheckedLeaves++;
                m_NGQueue.insert(Index::HeapCell(nn_index, distance2leaf));
            }
            if (m_NGQueue.Top().distance > m_SPTQueue.Top().distance) {
                BKTSearch(query, m_NGQueue, m_SPTQueue, nodeCheckStatus, m_iNumberOfCheckedLeaves, m_iNumberOfTreeCheckedLeaves, index->m_iNumberOfOtherDynamicPivots + m_iNumberOfCheckedLeaves);
            }
        }
        p_query.SortResult();
        for(int i = 0; i < p_query.GetResultNum(); i ++) {
            if(p_query.GetResult(i)->Dist == MaxDist) break;
            result.emplace_back(p_query.GetResult(i)->VID);
        }
        result.resize(result.size() > K ? K : result.size());
    }

    void ComponentSearchRouteSPTAG_BKT::BKTSearch(unsigned int query, Index::Heap &m_NGQueue,
                                                  Index::Heap &m_SPTQueue, Index::OptHashPosVector &nodeCheckStatus,
                                                  unsigned int &m_iNumberOfCheckedLeaves,
                                                  unsigned int &m_iNumberOfTreeCheckedLeaves,
                                                  int p_limits) {
        while (!m_SPTQueue.empty())
        {
            Index::HeapCell bcell = m_SPTQueue.pop();
            const Index::BKTNode& tnode = index->m_pBKTreeRoots[bcell.node];
            if (tnode.childStart < 0) {
                if (!nodeCheckStatus.CheckAndSet(tnode.centerid)) {
                    m_iNumberOfCheckedLeaves++;
                    m_NGQueue.insert(Index::HeapCell(tnode.centerid, bcell.distance));
                }
                if (m_iNumberOfCheckedLeaves >= p_limits) break;
            }
            else {
                if (!nodeCheckStatus.CheckAndSet(tnode.centerid)) {
                    m_NGQueue.insert(Index::HeapCell(tnode.centerid, bcell.distance));
                }
                for (int begin = tnode.childStart; begin < tnode.childEnd; begin++) {
                    int tmp = index->m_pBKTreeRoots[begin].centerid;
                    float dist = index->getDist()->compare(index->getQueryData() + index->getQueryDim() * query,
                                                           index->getBaseData() + index->getBaseDim() * tmp,
                                                           index->getBaseDim());
                    index->addDistCount();
                    m_SPTQueue.insert(Index::HeapCell(begin, dist));
                }
            }
        }
    }


    void ComponentSearchRouteNGT::RouteInner(unsigned int query, std::vector<Index::Neighbor> &pool,
                                             std::vector<unsigned int> &res) {
        const auto L = index->getParam().get<unsigned>("L_search");
        const auto K = index->getParam().get<unsigned>("K_search");

        float radius = static_cast<float>(FLT_MAX);

        // setup edgeSize
        size_t edgeSize = pool.size();

        std::priority_queue<Index::Neighbor, std::vector<Index::Neighbor>, std::greater<Index::Neighbor>> unchecked;
        Index::DistanceCheckedSet distanceChecked;

        std::priority_queue<Index::Neighbor, std::vector<Index::Neighbor>, std::less<Index::Neighbor>> results;
        //setupDistances(sc, seeds);

        //setupSeeds(sc, seeds, results, unchecked, distanceChecked);
        std::sort(pool.begin(), pool.end());

        for (auto ri = pool.begin(); ri != pool.end(); ri++){
            if ((results.size() < L) && ((*ri).distance <= radius)){
                results.push((*ri));
            }else{
                break;
            }
        }

        if (results.size() >= L){
            radius = results.top().distance;
        }

        for (auto ri = pool.begin(); ri != pool.end(); ri++){
            distanceChecked.insert((*ri).id);
            unchecked.push(*ri);
        }

        float explorationRadius = index->explorationCoefficient * radius;

        while (!unchecked.empty()){
            //std::cout << "radius: " << explorationRadius << std::endl;
            Index::Neighbor target = unchecked.top();

            unchecked.pop();
            if (target.distance > explorationRadius){
                break;
            }
            std::vector<unsigned> neighbors = index->getLoadGraph()[target.id];
            if (neighbors.empty()){
                continue;
            }

            index->addHopCount();
            for (unsigned neighborptr = 0; neighborptr < neighbors.size(); ++neighborptr){
                //sc.visitCount++;
                unsigned neighbor = index->getLoadGraph()[target.id][neighborptr];
                if (distanceChecked[neighbor]){
                    continue;
                }
                distanceChecked.insert(neighbor);

                float distance = index->getDist()->compare(index->getQueryData() + index->getQueryDim() * query,
                                                           index->getBaseData() + index->getBaseDim() * neighbor,
                                                           index->getBaseDim());
                index->addDistCount();
                //sc.distanceComputationCount++;
                if (distance <= explorationRadius){
                    unchecked.push(Index::Neighbor(neighbor, distance, true));
                    if (distance <= radius){
                        results.push(Index::Neighbor(neighbor, distance, true));
                        if (results.size() >= L){
                            if (results.top().distance >= distance){
                                if (results.size() > L){
                                    results.pop();
                                }
                                radius = results.top().distance;
                                explorationRadius = index->explorationCoefficient * radius;
                            }
                        }
                    }
                }
            }
        }
        //std::cout << "res : " << results.size() << std::endl;
        while(!results.empty()) {
            //std::cout << results.top().id << "|" << results.top().distance << " ";
            if(results.size() <= K) {
                res.push_back(results.top().id);
            }

            results.pop();
        }
        //std::cout << std::endl;

        sort(res.begin(), res.end());

        //sc.distanceComputationCount = so.distanceComputationCount;
        //sc.visitCount = so.visitCount;
    }
}
//
// Created by MurphySL on 2020/10/23.
//

#include "weavess/component.h"

namespace weavess {

    // NO LIMIT GREEDY
    void
    ComponentCandidateNSG::CandidateInner(const unsigned query, const unsigned enter, boost::dynamic_bitset<> flags,
                                          std::vector<Index::SimpleNeighbor> &result) {
        auto L = index->getParam().get<unsigned>("L_refine");

        std::vector<unsigned> init_ids(L);
        std::vector<Index::Neighbor> retset;
        retset.resize(L + 1);

        L = 0;
        // 选取质点近邻作为初始候选点
        for (unsigned i = 0; i < init_ids.size() && i < index->getFinalGraph()[enter].size(); i++) {
            init_ids[i] = index->getFinalGraph()[enter][i].id;
            flags[init_ids[i]] = true;
            L++;
        }
        // 候选点不足填入随机点
        while (L < init_ids.size()) {
            unsigned id = rand() % index->getBaseLen();
            if (flags[id]) continue;
            init_ids[L] = id;
            L++;
            flags[id] = true;
        }
        L = 0;
        // unsinged -> SimpleNeighbor
        for (unsigned i = 0; i < init_ids.size(); i++) {
            unsigned id = init_ids[i];
            if (id >= index->getBaseLen()) continue;
            float dist = index->getDist()->compare(index->getBaseData() + index->getBaseDim() * id,
                                                   index->getBaseData() + index->getBaseDim() * query,
                                                   (unsigned) index->getBaseDim());

            retset[i] = Index::Neighbor(id, dist, true);
            result.emplace_back(Index::SimpleNeighbor(id, dist));

            L++;
        }
        std::sort(retset.begin(), retset.begin() + L);
        index->i++;

        int k = 0;
        while (k < (int) L) {
            int nk = L;
            if (retset[k].flag) {
                retset[k].flag = false;
                unsigned n = retset[k].id;
                for (unsigned m = 0; m < index->getFinalGraph()[n].size(); ++m) {

                    unsigned id = index->getFinalGraph()[n][m].id;

                    if (flags[id]) continue;
                    flags[id] = true;

                    float dist = index->getDist()->compare(index->getBaseData() + index->getBaseDim() * query,
                                                           index->getBaseData() + index->getBaseDim() * (size_t) id,
                                                           (unsigned) index->getBaseDim());

                    Index::Neighbor nn(id, dist, true);
                    result.push_back(Index::SimpleNeighbor(id, dist));

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

        std::vector<Index::Neighbor>().swap(retset);
        std::vector<unsigned>().swap(init_ids);
    }

    // PROPAGATION 2
    void ComponentCandidatePropagation2::CandidateInner(const unsigned query, const unsigned enter,
                                                        boost::dynamic_bitset<> flags,
                                                        std::vector<Index::SimpleNeighbor> &pool) {
        flags[enter] = true;

        for (unsigned i = 0; i < index->getFinalGraph()[enter].size(); i++) {
            unsigned nid = index->getFinalGraph()[enter][i].id;
            for (unsigned nn = 0; nn < index->getFinalGraph()[nid].size(); nn++) {
                unsigned nnid = index->getFinalGraph()[nid][nn].id;
                if (flags[nnid]) continue;
                flags[nnid] = true;
                float dist = index->getDist()->compare(index->getBaseData() + index->getBaseDim() * query,
                                                       index->getBaseData() + index->getBaseDim() * nnid,
                                                       index->getBaseDim());
                pool.emplace_back(nnid, dist);
                if (pool.size() >= index->L_refine) break;
            }
            if (pool.size() >= index->L_refine) break;
        }
    }

    // SPTAG_BKT
    void ComponentCandidateSPTAG_BKT::CandidateInner(unsigned int query, unsigned int enter,
                                                     boost::dynamic_bitset<> flags,
                                                     std::vector<Index::SimpleNeighbor> &result) {

        //how many nodes will be visited for a query in the search stage
        unsigned m_iMaxCheck = 8192L;
        //how many nodes each node will visit during graph refine in the build stage
        unsigned maxCheck = index->m_iMaxCheckForRefineGraph > m_iMaxCheck ? index->m_iMaxCheckForRefineGraph : m_iMaxCheck;

        // Prioriy queue used for neighborhood graph
        Index::Heap m_NGQueue;
        // 初始化
        m_NGQueue.Resize(maxCheck * 30);

        // Priority queue Used for Tree
        Index::Heap m_SPTQueue;
        m_SPTQueue.Resize(maxCheck * 10);

        Index::OptHashPosVector nodeCheckStatus;
        nodeCheckStatus.Init(maxCheck, index->m_iHashTableExp);

        Index::QueryResultSet p_query(index->L_refine);

        unsigned m_iNumOfContinuousNoBetterPropagation = 0;
        unsigned m_iNumberOfCheckedLeaves = 0;
        unsigned m_iContinuousLimit = maxCheck / 64;

        // init_search_trees
        for(unsigned i = 0; i < index->m_iTreeNumber; i ++) {
            const Index::BKTNode& node = index->m_pBKTreeRoots[index->m_pTreeStart[i]];
            if (node.childStart < 0) {
                m_SPTQueue.insert(Index::HeapCell(index->m_pTreeStart[i],
                                                  index->getDist()->compare(index->getBaseData() + index->getBaseDim() * query,
                                                                            index->getBaseData() + index->getBaseDim() * node.centerid,
                                                                            index->getBaseDim())));
            }
            else {
                for (unsigned begin = node.childStart; begin < node.childEnd; begin++) {
                    unsigned cen = index->m_pBKTreeRoots[begin].centerid;
                    m_SPTQueue.insert(Index::HeapCell(begin,
                                                      index->getDist()->compare(index->getBaseData() + index->getBaseDim() * query,
                                                                                index->getBaseData() + index->getBaseDim() * cen,
                                                                                index->getBaseDim())));
                }
            }
        }

        // search_trees
        while (!m_SPTQueue.empty())
        {
            Index::HeapCell bcell = m_SPTQueue.pop();
            const Index::BKTNode& tnode = index->m_pBKTreeRoots[bcell.node];
            if (tnode.childStart < 0) {
                if (!nodeCheckStatus.CheckAndSet(tnode.centerid)) {
                    m_iNumberOfCheckedLeaves++;
                    m_NGQueue.insert(Index::HeapCell(tnode.centerid, bcell.distance));
                }
                if (m_iNumberOfCheckedLeaves >= index->m_iNumberOfInitialDynamicPivots) break;
            }
            else {
                if (!nodeCheckStatus.CheckAndSet(tnode.centerid)) {
                    m_NGQueue.insert(Index::HeapCell(tnode.centerid, bcell.distance));
                }
                for (unsigned begin = tnode.childStart; begin < tnode.childEnd; begin++) {
                    unsigned cen = index->m_pBKTreeRoots[begin].centerid;
                    m_SPTQueue.insert(Index::HeapCell(begin,
                                                      index->getDist()->compare(index->getBaseData() + index->getBaseDim() * query,
                                                                                index->getBaseData() + index->getBaseDim() * cen,
                                                                                index->getBaseDim())));
                }
            }
        }

        const unsigned checkPos = index->m_iNeighborhoodSize - 1;
        while (!m_NGQueue.empty()) {
            Index::HeapCell gnode = m_NGQueue.pop();
            unsigned tmpNode = gnode.node;
            std::vector<Index::SimpleNeighbor> node = index->getFinalGraph()[tmpNode];
            if (gnode.distance <= p_query.worstDist()) {
                unsigned checkNode = node[checkPos].id;
                if ((int)checkNode < -1) {
                    //std::cout << 2.211 << std::endl;
                    const Index::BKTNode& tnode = index->m_pBKTreeRoots[-2 - (int)checkNode];
                    unsigned i = -tnode.childStart;

                    do {
                        m_iNumOfContinuousNoBetterPropagation = 0;
                        p_query.AddPoint(tmpNode, gnode.distance);
                        break;
                        tmpNode = index->m_pBKTreeRoots[i].centerid;
                    }while(i++ < tnode.childEnd);
                } else {
                    m_iNumOfContinuousNoBetterPropagation = 0;
                    p_query.AddPoint(tmpNode, gnode.distance);
                }
            } else {
                m_iNumOfContinuousNoBetterPropagation++;
                if (m_iNumOfContinuousNoBetterPropagation > m_iContinuousLimit || m_iNumberOfCheckedLeaves > index->m_iMaxCheck) {
                    p_query.SortResult();

                    for(auto & iter : p_query) {
                        result.emplace_back(iter.VID, iter.Dist);
                    }
                    return;
                }
            }

            for (unsigned i = 0; i <= checkPos; i++) {
                unsigned nn_index = node[i].id;
                if ((int)nn_index < 0) break;
                if (nodeCheckStatus.CheckAndSet(nn_index)) continue;
                float distance2leaf = index->getDist()->compare(index->getBaseData() + index->getBaseDim() * query,
                                                                index->getBaseData() + index->getBaseDim() * nn_index,
                                                                index->getBaseDim());
                m_iNumberOfCheckedLeaves++;
                m_NGQueue.insert(Index::HeapCell(nn_index, distance2leaf));
            }
            if (m_NGQueue.Top().distance >m_SPTQueue.Top().distance) {
                //SearchTrees(m_pSamples, m_fComputeDistance, p_query, p_space, m_iNumberOfOtherDynamicPivots + p_space.m_iNumberOfCheckedLeaves);
                while (!m_SPTQueue.empty())
                {
                    Index::HeapCell bcell = m_SPTQueue.pop();
                    const Index::BKTNode& tnode = index->m_pBKTreeRoots[bcell.node];
                    if (tnode.childStart < 0) {
                        if (!nodeCheckStatus.CheckAndSet(tnode.centerid)) {
                            m_iNumberOfCheckedLeaves++;
                            m_NGQueue.insert(Index::HeapCell(tnode.centerid, bcell.distance));
                        }
                        if (m_iNumberOfCheckedLeaves >= index->m_iNumberOfInitialDynamicPivots) break;
                    }
                    else {
                        if (!nodeCheckStatus.CheckAndSet(tnode.centerid)) {
                            m_NGQueue.insert(Index::HeapCell(tnode.centerid, bcell.distance));
                        }
                        for (unsigned begin = tnode.childStart; begin < tnode.childEnd; begin++) {
                            unsigned cen = index->m_pBKTreeRoots[begin].centerid;
                            m_SPTQueue.insert(Index::HeapCell(begin,
                                                              index->getDist()->compare(index->getBaseData() + index->getBaseDim() * query,
                                                                                        index->getBaseData() + index->getBaseDim() * cen,
                                                                                        index->getBaseDim())));
                        }
                    }
                }
            }
        }

        p_query.SortResult();
        for(auto & iter : p_query) {
            result.emplace_back(iter.VID, iter.Dist);
        }
    }

    // SPTAG_KDT
    void ComponentCandidateSPTAG_KDT::CandidateInner(unsigned int query, unsigned int enter,
                                                     boost::dynamic_bitset<> flags,
                                                     std::vector<Index::SimpleNeighbor> &result) {
        unsigned m_iNumberOfCheckedLeaves = 0;
        unsigned m_iNumberOfTreeCheckedLeaves = 0;
        unsigned m_iNumberOfInitialDynamicPivots = 50;
        unsigned m_iMaxCheck = 8192L;
        unsigned m_iNumOfContinuousNoBetterPropagation = 0;
        unsigned m_iThresholdOfNumberOfContinuousNoBetterPropagation = 3;
        unsigned m_iNumberOfOtherDynamicPivots = 4;
        unsigned maxCheck = index->m_iMaxCheckForRefineGraph > m_iMaxCheck ? index->m_iMaxCheckForRefineGraph : m_iMaxCheck;

        // Prioriy queue used for neighborhood graph
        Index::Heap m_NGQueue;
        // 初始化
        m_NGQueue.Resize(maxCheck * 30);

        // Priority queue Used for Tree
        Index::Heap m_SPTQueue;
        m_SPTQueue.Resize(maxCheck * 10);

        // Priority queue used for result
        std::priority_queue<Index::SPTAGFurtherFirst> tmp;
        std::priority_queue<Index::SPTAGCloserFirst> res;

        Index::OptHashPosVector nodeCheckStatus;
        nodeCheckStatus.Init(maxCheck, index->m_iHashTableExp);

        // 根据 KDT 获取入口点
        for(int i = 0; i < index->m_iTreeNumber; i ++) {
            unsigned node = index->m_pTreeStart[i];

            KDTSearch(query, node, m_NGQueue, m_SPTQueue, nodeCheckStatus, m_iNumberOfCheckedLeaves, m_iNumberOfTreeCheckedLeaves);
        }

        // 查询足够的 KDT 结点
        unsigned p_limits = m_iNumberOfInitialDynamicPivots;
        while (!m_SPTQueue.empty() && m_iNumberOfCheckedLeaves < p_limits)
        {
            auto& tcell = m_SPTQueue.pop();
            KDTSearch(query, tcell.node, m_NGQueue, m_SPTQueue, nodeCheckStatus, m_iNumberOfCheckedLeaves, m_iNumberOfTreeCheckedLeaves);
        }

        while (!m_NGQueue.empty()) {
            Index::HeapCell gnode = m_NGQueue.pop();
            std::vector<Index::SimpleNeighbor> node = index->getFinalGraph()[gnode.node];

            if(tmp.size() < index->L_refine || (tmp.size() >= index->L_refine && tmp.top().GetDistance() > gnode.distance) || (tmp.size() >= index->L_refine && tmp.top().GetDistance() == gnode.distance && tmp.top().GetNode() > gnode.node)) {
                if(tmp.size() + 1 > index->L_refine)
                    tmp.pop();
                tmp.push(Index::SPTAGFurtherFirst(gnode.node, gnode.distance));
                if(m_iNumberOfCheckedLeaves > m_iMaxCheck) {
                    while (!tmp.empty()) {
                        res.push(Index::SPTAGCloserFirst(tmp.top().GetNode(), tmp.top().GetDistance()));
                        tmp.pop();
                    }

                    int num = 0;
                    while (!res.empty()) {
                        num ++;
                        auto top_node = res.top();
                        result.emplace_back(top_node.GetNode(), top_node.GetDistance());
                        res.pop();
                    }
                    ////std::cout << 24.64 << std::endl;

                    if(num < index->L_refine) {
                        result.emplace_back(-1, -1);
                    }
                    return;
                }
            }
            float upperBound = std::max(tmp.top().GetDistance(), gnode.distance);
            bool bLocalOpt = true;
            //if(query == 700)
                //std::cout << 4.7 << std::endl;

            // modify
            for (unsigned i = 0; i < node.size(); i++) {
                unsigned nn_index = node[i].id;
                if (nn_index < 0) break;
                //if(query == 700)
                    //std::cout << 4.71 << std::endl;
                if (nodeCheckStatus.CheckAndSet(nn_index)) continue;
                //if(query == 700)
                    //std::cout << 4.72 << " " << nn_index << std::endl;
                float distance2leaf = index->getDist()->compare(index->getBaseData() + index->getBaseDim() * query,
                                                                index->getBaseData() + index->getBaseDim() * nn_index,
                                                                index->getBaseDim());
                //if(query == 700)
                    //std::cout << 4.73 << " " << std::endl;
                if (distance2leaf <= upperBound) bLocalOpt = false;
                m_iNumberOfCheckedLeaves++;
                m_NGQueue.insert(Index::HeapCell(nn_index, distance2leaf));
            }
            //if(query == 700)
                //std::cout << 4.8 << std::endl;
            if (bLocalOpt) m_iNumOfContinuousNoBetterPropagation++;
            else m_iNumOfContinuousNoBetterPropagation = 0;
            if (m_iNumOfContinuousNoBetterPropagation > m_iThresholdOfNumberOfContinuousNoBetterPropagation) {
                if (m_iNumberOfTreeCheckedLeaves <= m_iNumberOfCheckedLeaves / 10) {
                    while(!m_SPTQueue.empty() && m_iNumberOfCheckedLeaves < m_iNumberOfCheckedLeaves + m_iNumberOfOtherDynamicPivots) {
                        auto& tcell = m_SPTQueue.pop();
                        KDTSearch(query, tcell.node, m_NGQueue, m_SPTQueue, nodeCheckStatus, m_iNumberOfCheckedLeaves, m_iNumberOfTreeCheckedLeaves);
                    }
                } else if (gnode.distance > tmp.top().GetDistance()) {
                    break;
                }
            }
            //if(query == 700)
                //std::cout << 4.9l << std::endl;
        }

        while (!tmp.empty()) {
            res.push(Index::SPTAGCloserFirst(tmp.top().GetNode(), tmp.top().GetDistance()));
            tmp.pop();
        }

        while (!res.empty()) {
            auto top_node = res.top();
            result.emplace_back(top_node.GetNode(), top_node.GetDistance());
            res.pop();
        }
    }

    void ComponentCandidateSPTAG_KDT::KDTSearch(unsigned query, unsigned node, Index::Heap &m_NGQueue, Index::Heap &m_SPTQueue, Index::OptHashPosVector &nodeCheckStatus,
                                                unsigned &m_iNumberOfCheckedLeaves, unsigned &m_iNumberOfTreeCheckedLeaves) {
        if((int)node < 0) {
            unsigned tmp = -(int)node - 1;
            if(tmp > index->getBaseLen()) return;

            if(nodeCheckStatus.CheckAndSet(tmp)) return;

            ++m_iNumberOfCheckedLeaves;
            ++m_iNumberOfTreeCheckedLeaves;

            m_NGQueue.insert(Index::HeapCell(tmp, index->getDist()->compare(index->getBaseData() + index->getBaseDim() * query,
                                                                            index->getBaseData() + index->getBaseDim() * tmp,
                                                                            index->getBaseDim())));
            return;
        }
        auto& tnode = index->m_pKDTreeRoots[node];

        float distBound = 0;
        float diff = (index->getBaseData() + index->getBaseDim() * query)[tnode.split_dim] - tnode.split_value;
        float distanceBound = distBound + diff * diff;
        unsigned otherChild, bestChild;
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

    void ComponentCandidateSPTAG_KDT_new::KDTSearch(unsigned int query, int node, Index::Heap &m_NGQueue,
                                                    Index::Heap &m_SPTQueue, Index::OptHashPosVector &nodeCheckStatus,
                                                    unsigned int &m_iNumberOfCheckedLeaves,
                                                    unsigned int &m_iNumberOfTreeCheckedLeaves) {
        //std::cout << 1 << std::endl;
        if (node < 0)
        {
            int tmp = -node - 1;
            if (tmp >= index->getBaseLen()) return;
            if (nodeCheckStatus.CheckAndSet(tmp)) return;

            ++m_iNumberOfTreeCheckedLeaves;
            ++m_iNumberOfCheckedLeaves;
            m_NGQueue.insert(Index::HeapCell(tmp, index->getDist()->compare(index->getBaseData() + index->getBaseDim() * query,
                                                                            index->getBaseData() + index->getBaseDim() * tmp,
                                                                            index->getBaseDim())));
            return;
        }
        //std::cout << 2 << std::endl;

        auto& tnode = index->m_pKDTreeRoots[node];
        //std::cout << 2 << " " << tnode.split_dim << std::endl;

        float distBound = 0;
        float diff = (index->getBaseData() + index->getBaseDim() * query)[tnode.split_dim] - tnode.split_value;
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

        //std::cout << 3 << std::endl;

        m_SPTQueue.insert(Index::HeapCell(otherChild, distanceBound));
        KDTSearch(query, bestChild, m_NGQueue, m_SPTQueue, nodeCheckStatus, m_iNumberOfCheckedLeaves, m_iNumberOfTreeCheckedLeaves);
    }

    void ComponentCandidateSPTAG_KDT_new::CandidateInner(unsigned int query, unsigned int enter,
                                                         boost::dynamic_bitset<> flags,
                                                         std::vector<Index::SimpleNeighbor> &result) {
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

        Index::QueryResultSet p_query(index->L_refine + 1);
        //std::priority_queue<Index::SPTAGCloserFirst> res;

        // InitSearchTrees 根据 KDT 获取入口点
        for(int i = 0; i < index->m_iTreeNumber; i ++) {
            int node = index->m_pTreeStart[i];

            KDTSearch(query, node, m_NGQueue, m_SPTQueue, nodeCheckStatus, m_iNumberOfCheckedLeaves, m_iNumberOfTreeCheckedLeaves);
        }

        // SearchTrees 查询足够的 KDT 结点
        unsigned p_limits = m_iNumberOfInitialDynamicPivots;
        while (!m_SPTQueue.empty() && m_iNumberOfCheckedLeaves < p_limits)
        {
            auto& tcell = m_SPTQueue.pop();
            KDTSearch(query, tcell.node, m_NGQueue, m_SPTQueue, nodeCheckStatus, m_iNumberOfCheckedLeaves, m_iNumberOfTreeCheckedLeaves);
        }

        while (!m_NGQueue.empty()) {
            Index::HeapCell gnode = m_NGQueue.pop();
            std::vector<Index::SimpleNeighbor> node = index->getFinalGraph()[gnode.node];

            if (!p_query.AddPoint(gnode.node, gnode.distance) && m_iNumberOfCheckedLeaves > index->m_iMaxCheck) {
                p_query.SortResult();
                for(int i = 0; i < p_query.GetResultNum(); i ++) {
                    if(p_query.GetResult(i)->Dist == MaxDist) break;
                    result.emplace_back(p_query.GetResult(i)->VID, p_query.GetResult(i)->Dist);
                }
                return;
            }
            float upperBound = std::max(p_query.worstDist(), gnode.distance);
            bool bLocalOpt = true;
            for (unsigned i = 0; i < index->R_refine; i++) {
                int nn_index = node[i].id;
                if (nn_index < 0) break;
                if (nodeCheckStatus.CheckAndSet(nn_index)) continue;
                float distance2leaf = index->getDist()->compare(index->getBaseData() + index->getBaseDim() * query,
                                                       index->getBaseData() + index->getBaseDim() * nn_index,
                                                       index->getBaseDim());
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
            result.emplace_back(p_query.GetResult(i)->VID, p_query.GetResult(i)->Dist);
        }
    }
}
//
// Created by MurphySL on 2020/10/23.
//

#include "weavess/component.h"

namespace weavess {

    void ComponentSearchEntryRand::SearchEntryInner(unsigned query, std::vector<Index::Neighbor> &pool) {
        const auto L = index->getParam().get<unsigned>("L_search");

        pool.resize(L + 1);

        std::vector<unsigned> init_ids(L);
        std::mt19937 rng(rand());

        GenRandom(rng, init_ids.data(), L, (unsigned) index->getBaseLen());
        std::vector<char> flags(index->getBaseLen());
        memset(flags.data(), 0, index->getBaseLen() * sizeof(char));
        for (unsigned i = 0; i < L; i++) {
            unsigned id = init_ids[i];
            float dist = index->getDist()->compare(index->getQueryData() + query * index->getQueryDim(),
                                                   index->getBaseData() + id * index->getBaseDim(),
                                                   (unsigned) index->getBaseDim());
            index->addDistCount();
            pool[i] = Index::Neighbor(id, dist, true);
        }

        std::sort(pool.begin(), pool.begin() + L);
    }


    void ComponentSearchEntryCentroid::SearchEntryInner(unsigned int query, std::vector<Index::Neighbor> &pool) {
        const auto L = index->getParam().get<unsigned>("L_search");
        pool.reserve(L + 1);
        std::vector<unsigned> init_ids(L);
        boost::dynamic_bitset<> flags{index->getBaseLen(), 0};
        // std::mt19937 rng(rand());
        // GenRandom(rng, init_ids.data(), L, (unsigned) index_->n_);

        unsigned tmp_l = 0;
        for (; tmp_l < L && tmp_l < index->getLoadGraph()[index->ep_].size(); tmp_l++) {
            init_ids[tmp_l] = index->getLoadGraph()[index->ep_][tmp_l];
            flags[init_ids[tmp_l]] = true;
        }

        while (tmp_l < L) {
            unsigned id = rand() % index->getBaseLen();
            if (flags[id]) continue;
            flags[id] = true;
            init_ids[tmp_l] = id;
            tmp_l++;
        }

        for (unsigned i = 0; i < init_ids.size(); i++) {
            unsigned id = init_ids[i];
            float dist =
                    index->getDist()->compare(index->getBaseData() + index->getBaseDim() * id,
                                              index->getQueryData() + index->getQueryDim() * query,
                                              (unsigned) index->getBaseDim());
            index->addDistCount();
            pool[i] = Index::Neighbor(id, dist, true);
            // flags[id] = true;
        }

        std::sort(pool.begin(), pool.begin() + L);
    }


    void ComponentSearchEntrySubCentroid::SearchEntryInner(unsigned int query, std::vector<Index::Neighbor> &pool) {
        auto L = index->getParam().get<unsigned>("L_search");
        pool.resize(L + 1);
        std::vector<unsigned> init_ids(L);
        std::mt19937 rng(rand());
        GenRandom(rng, init_ids.data(), L, (unsigned) index->getBaseLen());

        assert(index->eps_.size() <= L);
        for (unsigned i = 0; i < index->eps_.size(); i++) {
            init_ids[i] = index->eps_[i];
        }

        boost::dynamic_bitset<> flags{index->getBaseLen(), 0};
        L = 0;
        for (unsigned i = 0; i < init_ids.size(); i++) {
            unsigned id = init_ids[i];
            if (id >= index->getBaseLen()) continue;
            float *x = (float *) (index->getBaseData() + index->getBaseDim() * id);
            float norm_x = *x;
            x++;
            float dist = index->getDist()->compare(x, index->getQueryData() + query * index->getQueryDim(),
                                                   (unsigned) index->getBaseDim());
            index->addDistCount();
            pool[i] = Index::Neighbor(id, dist, true);
            flags[id] = true;
            L++;
        }

        std::sort(pool.begin(), pool.begin() + L);
    }


    void ComponentSearchEntryKDT::SearchEntryInner(unsigned int query, std::vector<Index::Neighbor> &pool) {
        unsigned TreeNum = index->nTrees;
        const auto L = index->getParam().get<unsigned>("L_search");

        pool.clear();
        pool.resize(L+1);

        std::vector<char> flags(index->getBaseLen());
        std::memset(flags.data(), 0, index->getBaseLen() * sizeof(char));
        flags[query] = true;

        std::vector<unsigned> init_ids(L);

        unsigned lsize = L / (TreeNum * index->TNS) + 1;
        std::vector<std::vector<Index::Node*> > Vnl;
        Vnl.resize(TreeNum);
        for(unsigned i =0; i < TreeNum; i ++)
            getSearchNodeList(index->tree_roots_[i], index->getQueryData() + index->getQueryDim() * query, lsize, Vnl[i]);

        unsigned p = 0;
        for(unsigned ni = 0; ni < lsize; ni ++) {
            for(unsigned i = 0; i < Vnl.size(); i ++) {
                Index::Node *leafn = Vnl[i][ni];
                for(size_t j = leafn->StartIdx; j < leafn->EndIdx && p < L; j ++) {
                    size_t nn = index->LeafLists[i][j];
                    if(flags[nn])continue;
                    flags[nn] = 1;
                    init_ids[p++]=(nn);
                }
                if(p >= L) break;
            }
            if(p >= L) break;
        }

        while(p < L){
            unsigned int nn = rand() % index->getBaseLen();
            if(flags[nn])continue;
            flags[nn] = 1;
            init_ids[p++]=(nn);
        }

        memset(flags.data(), 0, index->getBaseLen() * sizeof(char));
        for(unsigned i=0; i<L; i++){
            unsigned id = init_ids[i];
            float dist = index->getDist()->compare(index->getBaseData() + index->getBaseDim() * id,
                                                  index->getQueryData() + index->getQueryDim() * query,
                                                  index->getBaseDim());
            index->addDistCount();
            pool[i]=Index::Neighbor(id, dist, true);
        }

        std::sort(pool.begin(), pool.begin()+L);
    }

    void ComponentSearchEntryKDT::getSearchNodeList(Index::Node* node, const float *q, unsigned int lsize, std::vector<Index::Node*>& vn){
        if(vn.size() >= lsize)
            return;

        if(node->Lchild != nullptr && node->Rchild != nullptr){
            if(q[node->DivDim] < node->DivVal){
                getSearchNodeList(node->Lchild, q, lsize,  vn );
                getSearchNodeList(node->Rchild, q, lsize, vn);
            }else{
                getSearchNodeList(node->Rchild, q, lsize, vn);
                getSearchNodeList(node->Lchild, q, lsize, vn);
            }
        }else
            vn.push_back(node);
    }


    void ComponentSearchEntryKDTSingle::SearchEntryInner(unsigned int query, std::vector<Index::Neighbor> &pool) {
        unsigned TreeNum = index->nTrees;
        const auto L = index->getParam().get<unsigned>("L_search");

        pool.clear();
        pool.resize(L+1);

        std::vector<char> flags(index->getBaseLen());
        std::memset(flags.data(), 0, index->getBaseLen() * sizeof(char));
        flags[query] = true;

        std::vector<unsigned> init_ids(L);

        unsigned lsize = L / (TreeNum * index->TNS) + 1;
        std::vector<std::vector<Index::Node*> > Vnl;
        Vnl.resize(TreeNum);
        for(unsigned i =0; i < TreeNum; i ++)
            getSearchNodeList(index->tree_roots_[i], index->getQueryData() + index->getQueryDim() * query, lsize, Vnl[i]);

        unsigned p = 0;
        for(unsigned ni = 0; ni < lsize; ni ++) {
            for(unsigned i = 0; i < Vnl.size(); i ++) {
                Index::Node *leafn = Vnl[i][ni];
                for(size_t j = leafn->StartIdx; j < leafn->EndIdx && p < L; j ++) {
                    size_t nn = index->LeafLists[i][j];
                    if(flags[nn])continue;
                    flags[nn] = 1;
                    init_ids[p++]=(nn);
                }
                if(p >= L) break;
            }
            if(p >= L) break;
        }

        while(p < L){
            unsigned int nn = rand() % index->getBaseLen();
            if(flags[nn])continue;
            flags[nn] = 1;
            init_ids[p++]=(nn);
        }

        memset(flags.data(), 0, index->getBaseLen() * sizeof(char));
        for(unsigned i=0; i<L; i++){
            unsigned id = init_ids[i];
            float dist = index->getDist()->compare(index->getBaseData() + index->getBaseDim() * id,
                                                   index->getQueryData() + index->getQueryDim() * query,
                                                   index->getBaseDim());
            index->addDistCount();
            pool[i]=Index::Neighbor(id, dist, true);
        }

        std::sort(pool.begin(), pool.begin()+L);
    }

    void ComponentSearchEntryKDTSingle::getSearchNodeList(Index::Node* node, const float *q, unsigned int lsize, std::vector<Index::Node*>& vn){
        if(vn.size() >= lsize)
            return;

        if(node->Lchild != nullptr && node->Rchild != nullptr){
            if(q[node->DivDim] < node->DivVal){
                getSearchNodeList(node->Lchild, q, lsize,  vn );
                getSearchNodeList(node->Rchild, q, lsize, vn);
            }else{
                getSearchNodeList(node->Rchild, q, lsize, vn);
                getSearchNodeList(node->Lchild, q, lsize, vn);
            }
        }else
            vn.push_back(node);
    }


    void ComponentSearchEntryHash::SearchEntryInner(unsigned int query, std::vector<Index::Neighbor> &pool) {
        unsigned int idx1 = index->querycode[query] >> index->LowerBits;
        unsigned int idx2 = index->querycode[query] - (idx1 << index->LowerBits);
        auto bucket = index->tb[idx1].find(idx2);
        //std::vector<int> canstmp;
        if (bucket != index->tb[idx1].end()) {
            std::vector<unsigned int> vp = bucket->second;
            //cout<<i<<":"<<vp.size()<<endl;
            for (size_t j = 0; j < vp.size() && pool.size() < INIT_NUM; j++) {
                pool.emplace_back(vp[j], -1, true);
            }
        }

        for (size_t j = 0; j < DEPTH; j++) {
            unsigned int searchcode = index->querycode[query] ^(1 << j);
            unsigned int idx1 = searchcode >> index->LowerBits;
            unsigned int idx2 = searchcode - (idx1 << index->LowerBits);
            auto bucket = index->tb[idx1].find(idx2);
            if (bucket != index->tb[idx1].end()) {
                std::vector<unsigned int> vp = bucket->second;
                for (size_t k = 0; k < vp.size() && pool.size() < INIT_NUM; k++) {
                    pool.push_back(Index::Neighbor(vp[k], -1, true));
                }
            }
        }
    }


    void ComponentSearchEntryNone::SearchEntryInner(unsigned int query, std::vector<Index::Neighbor> &pool) { }


    void ComponentSearchEntryVPT::SearchEntryInner(unsigned int query, std::vector<Index::Neighbor> &pool) {
        float cq = static_cast<float>(FLT_MAX);
        //float cq = 100;
        std::multimap<float, unsigned> result_found;
        Search(query, NGT_SEED_SIZE, result_found, index->vp_tree.get_root(), cq);

        for(auto it : result_found) {
            pool.emplace_back(it.second, it.first, true);
        }
    }


    void ComponentSearchEntryVPT::Search(const unsigned& query_value, const size_t count, std::multimap<float, unsigned> &pool,
                const Index::VPNodePtr& node, float& q)
    {
        assert(node.get());

        if(node->get_leaf_node())
        {
            for(size_t c_pos = 0; c_pos < node->m_objects_list.size(); ++c_pos)
            {
                //m_stat.distance_threshold_count++;
                float c_distance = index->getDist()->compare(index->getQueryData() + index->getQueryDim() * query_value,
                                                             index->getBaseData() + index->getBaseDim() * node->m_objects_list[c_pos],
                                                             index->getBaseDim());
                index->addDistCount();
                if( c_distance <= q)
                {
                    pool.insert(std::pair<float, unsigned>(c_distance, node->m_objects_list[c_pos]));

                    while(pool.size() > count)
                    {
                        typename std::multimap<float, unsigned>::iterator it_last = pool.end();

                        pool.erase(--it_last);
                    }

                    if(pool.size() == count)
                        q = (*pool.rbegin()).first;
                }
            }

        }else{
            float dist = 0; //m_get_distance(node->get_value(), query_value);

            // Search flag
            size_t c_mu_pos = index->vp_tree.m_non_leaf_branching_factor;
            if(node->m_mu_list.size() == 1)
            {
                c_mu_pos = 0;
                //m_stat.distance_threshold_count++;
                //dist = m_get_distance(node->get_value(), query_value, node->m_mu_list[c_mu_pos] + q);
                dist = index->getDist()->compare(index->getBaseData() + index->getBaseDim() * node->get_value(),
                                                 index->getQueryData() + index->getQueryDim() * query_value,
                                                 index->getBaseDim());
                index->addDistCount();
            }else{
                //m_stat.distance_count++;
                dist = index->getDist()->compare(index->getBaseData() + index->getBaseDim() * node->get_value(),
                                                 index->getQueryData() + index->getQueryDim() * query_value,
                                                 index->getBaseDim());
                index->addDistCount();
                //dist = m_get_distance(node->get_value(), query_value);
                for(size_t c_pos = 0; c_pos < node->m_mu_list.size() -1 ; ++c_pos)
                {
                    if(dist > node->m_mu_list[c_pos] && dist < node->m_mu_list[c_pos + 1] )
                    {
                        c_mu_pos = c_pos;
                        break;
                    }
                }
            }

            if(c_mu_pos != index->vp_tree.m_non_leaf_branching_factor)
            {
                float c_mu = node->m_mu_list[c_mu_pos];
                if(dist < c_mu)
                {
                    if(dist < c_mu + q)
                    {
                        //m_stat.search_jump++;
                        Search(query_value, count, pool, node->m_child_list[c_mu_pos], q);
                    }
                    if(dist >= c_mu - q)
                    {
                        //m_stat.search_jump++;
                        Search(query_value, count, pool, node->m_child_list[c_mu_pos + 1], q);
                    }
                }else
                {
                    if(dist >= c_mu - q)
                    {
                        //m_stat.search_jump++;
                        Search(query_value, count, pool, node->m_child_list[c_mu_pos + 1], q);
                    }
                    if(dist < c_mu + q)
                    {
                        //m_stat.search_jump++;
                        Search(query_value, count, pool, node->m_child_list[c_mu_pos], q);
                    }
                }
            }
        }

    }


}

//
// Created by MurphySL on 2020/10/23.
//

#include "weavess/component.h"

namespace weavess {




    /**
    * KDT获取最近入口点
    * @param query 查询点
    * @param pool 候选池
    */
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


    /**
     * Hash 获取入口点
     * @param query 查询点
     * @param pool 候选池
     */
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


    /**
     * 获取入口点空方法
     * @param query 查询点
     * @param pool 候选池
     */
    void ComponentSearchEntryNone::SearchEntryInner(unsigned int query, std::vector<Index::Neighbor> &pool) { }





}

//
// Created by Murph on 2020/8/18.
//
#include "weavess/index.h"
#include "weavess/index_component_load.h"
#include "weavess/index_component_init.h"
#include "weavess/index_component_coarse.h"
#include "weavess/index_component_prune.h"
#include "weavess/index_component_connect.h"

namespace weavess {
    inline void Index::IndexLoadData(FILE_TYPE type, char *&data_file) {
        IndexComponentLoad *loader = nullptr;
        switch (type) {
            case Index::VECS:
                std::cout << "__Loader I/FVECS__" << std::endl;
                loader = new IndexComponentLoadVECS(this);
                break;
            default:
                std::cerr << "load data wrong type" << std::endl;
        }

        loader->LoadDataInner(data_file, data_, dim_, n_);
        std::cout << "data dimension: " << dim_ << std::endl;
        std::cout << "data num: " << n_ << std::endl;
    }

    inline void Index::IndexInit(INIT_TYPE type) {
        IndexComponentInit *a = nullptr;
        switch (type) {
            case Index::Random:
                a = new IndexComponentInitRandom(this, param_);
                break;
            case Index::KDTree:
                a = new IndexComponentInitKDTree(this, param_);
            default:
                std::cerr << "init index wrong type" << std::endl;
        }
        a->InitInner();
    }

    inline void Index::IndexCoarseBuild(COARSE_TYPE type) {
        IndexComponentCoarse *a = nullptr;
        switch (type) {
            case Index::NN_Descent:
                a = new IndexComponentCoarseNNDescent(this, param_);
                break;
            default:
                std::cerr << "coarse KNN wrong type" << std::endl;
        }
        a->CoarseInner();
    }

    inline void Index::IndexPrune(PRUNE_TYPE type) {
        IndexComponentPrune *a = nullptr;
        switch (type) {
            case Index::PRUNE_NONE:
                a = new IndexComponentPruneNone(this, param_);
                break;
            case Index::NSG:
                a = new IndexComponentPruneNSG(this, param_);
                break;
            default:
                std::cerr << "prune wrong type" << std::endl;
        }
        a->PruneInner();
    }

    inline void Index::IndexConnect(CONN_TYPE type) {
        IndexComponentConnect *a = nullptr;

        switch (type) {
            case Index::CONN_NONE:
                a = new IndexComponentConnectNone(this, param_);
                break;
            case Index::DFS:
                a = new IndexComponentConnectDFS(this, param_);
                break;
            default:
                std::cerr << "connect wrong type" << std::endl;
        }
        a->ConnectInner();
    }

    inline void Index::Search(const float *query, const float *x, size_t K, const Parameters &parameter, unsigned int *indices) {
        const unsigned L = parameter.Get<unsigned>("L_search");

        std::vector<Neighbor> retset(L+1);
        std::vector<unsigned> init_ids(L);
        std::mt19937 rng(rand());
        GenRandom(rng, init_ids.data(), L, (unsigned)n_);

        std::vector<char> flags(n_);
        memset(flags.data(), 0, n_ * sizeof(char));
        for(unsigned i=0; i<L; i++){
            unsigned id = init_ids[i];
            float dist = distance_->compare(data_ + dim_*id, query, (unsigned)dim_);
            retset[i]=Neighbor(id, dist, true);
        }

        std::sort(retset.begin(), retset.begin()+L);
        int k=0;
        while(k < (int)L) {
            int nk = L;

            if (retset[k].flag) {
                retset[k].flag = false;
                unsigned n = retset[k].id;

                for (unsigned m = 0; m < final_graph_[n].size(); ++m) {
                    unsigned id = final_graph_[n][m];
                    if(flags[id])continue;
                    flags[id] = 1;
                    float dist = distance_->compare(query, data_ + dim_ * id, (unsigned)dim_);
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

    inline void Index::FreeGraphData() {}
}

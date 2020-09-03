//
// Created by Murph on 2020/8/17.
//

#ifndef WEAVESS_INDEX_COMPONENT_H
#define WEAVESS_INDEX_COMPONENT_H

#include <cassert>
#include <cstring>
#include <ctime>
#include <chrono>
#include <utility>
#include <mmcobj.h>
#include <queue>
#include <stack>
#include <boost/dynamic_bitset.hpp>
#include "index.h"
#include "parameters.h"

namespace weavess {

    class IndexComponent {
    public:
        explicit IndexComponent() = default;

        explicit IndexComponent(Index *index) : index_(index) {}

        virtual ~IndexComponent() = default;

    protected:
        Index *index_ = nullptr;
    };

    // load - 加载数据
    class IndexComponentLoad : public IndexComponent {
    public :
        virtual void
        LoadInner(Index *index, char *data_file, char *query_file, char *ground_file, Parameters &parameters);
    };

    // coarse — 初始图构建方法
    class IndexComponentCoarse : public IndexComponent {
    public:
        explicit IndexComponentCoarse(Index *index) : IndexComponent(index) {}

        virtual void CoarseInner() = 0;
    };

    // coarse NN
    class IndexComponentCoarseNNDescent : public IndexComponentCoarse {
    public:
        explicit IndexComponentCoarseNNDescent(Index *index) : IndexComponentCoarse(index) {}

        void CoarseInner() override;

    private:
        void init();

        void NNDescent();

        void join();

        void update(const Parameters &parameters);

        void generate_control_set(std::vector<unsigned> &c, std::vector<std::vector<unsigned> > &v, unsigned N);

        void eval_recall(std::vector<unsigned> &ctrl_points, std::vector<std::vector<unsigned> > &acc_eval_set);
    };

    // coarse KDT
    class IndexComponentCoarseKDT : public IndexComponentCoarse {
    public:
        explicit IndexComponentCoarseKDT(Index *index) : IndexComponentCoarse(index) {}

        void CoarseInner() override;

    private:
        void meanSplit(std::mt19937 &rng, unsigned *indices, unsigned count, unsigned &index, unsigned &cutdim,
                       float &cutval);

        void
        planeSplit(unsigned *indices, unsigned count, unsigned cutdim, float cutval, unsigned &lim1, unsigned &lim2);

        int selectDivision(std::mt19937 &rng, float *v);

        void DFSbuild(Index::Node *node, std::mt19937 &rng, unsigned *indices, unsigned count, unsigned offset);

        void DFStest(unsigned level, unsigned dim, Index::Node *node);

        Index::Node *SearchToLeaf(Index::Node *node, size_t id);

        void mergeSubGraphs(size_t treeid, Index::Node *node);

        void getMergeLevelNodeList(Index::Node *node, size_t treeid, unsigned deepth);
    };

    // coarse mst
    class IndexComponentCoarseMST : public IndexComponentCoarse {
    public:
        explicit IndexComponentCoarseMST(Index *index) : IndexComponentCoarse(index) {}

        void CoarseInner() override;

    private:
        int rand_int(const int & min, const int & max);

        std::vector<std::vector< Index::Edge > >  create_exact_mst(int *idx_points, int left, int right, int max_mst_degree);

        void create_clusters(int *idx_points, int left, int right, std::vector<std::vector< Index::Edge > > &graph, int minsize_cl, std::vector<omp_lock_t> &locks, int max_mst_degree);
    };


    // refine
    class IndexComponentPrune : public IndexComponent {
    public:
        explicit IndexComponentPrune(Index *index) : IndexComponent(index) {}

        virtual void PruneInner() = 0;
    };

    class IndexComponentPruneNSG : public IndexComponentPrune {
    public:
        explicit IndexComponentPruneNSG(Index *index) : IndexComponentPrune(index) {}

        void PruneInner() override;

    protected:
        virtual void Link(const Parameters &parameters, SimpleNeighbor *cut_graph_);

    private:
        void init_graph();

        void get_neighbors(const float *query, const Parameters &parameter,
                           std::vector<Neighbor> &retset,
                           std::vector<Neighbor> &fullset);

        void get_neighbors(const float *query, const Parameters &parameter,
                           boost::dynamic_bitset<> &flags,
                           std::vector<Neighbor> &retset,
                           std::vector<Neighbor> &fullset);

        void sync_prune(unsigned q, std::vector<Neighbor> &pool,
                        const Parameters &parameter,
                        boost::dynamic_bitset<> &flags,
                        SimpleNeighbor *cut_graph_);

        void InterInsert(unsigned n, unsigned range,
                         std::vector<std::mutex> &locks,
                         SimpleNeighbor *cut_graph_);
    };

    class IndexComponentPruneNSSG : public IndexComponentPruneNSG {
    public:
        explicit IndexComponentPruneNSSG(Index *index) : IndexComponentPruneNSG(index) {}

        void Link(const Parameters &parameters, SimpleNeighbor *cut_graph_) override;

    private:
        void InterInsert(unsigned n, unsigned range, float threshold,
                         std::vector<std::mutex> &locks,
                         SimpleNeighbor *cut_graph_);

        void sync_prune(unsigned q, std::vector<Neighbor> &pool,
                        const Parameters &parameters, float threshold,
                        SimpleNeighbor *cut_graph_);

        void get_neighbors(unsigned q, const Parameters &parameter, std::vector<Neighbor> &pool);
    };

    class IndexComponentPruneDPG : public IndexComponentPrune {
    public:
        explicit IndexComponentPruneDPG(Index *index) : IndexComponentPrune(index) {}

        void PruneInner() override;

    private:
        void sync_prune(unsigned q, SimpleNeighbor *cut_graph_);

        void Link(const Parameters &parameters, SimpleNeighbor *cut_graph_);

        void InterInsert(unsigned n, unsigned range, std::vector<std::mutex> &locks, SimpleNeighbor *cut_graph_);
    };

    class IndexComponentRefineNNDescent : public IndexComponentPrune {
    public:
        explicit IndexComponentRefineNNDescent(Index *index) : IndexComponentPrune(index) {}

        void PruneInner() override;
    };

    class IndexComponentRefineVAMANA : public IndexComponentPrune {
    public:
        explicit IndexComponentRefineVAMANA(Index *index) : IndexComponentPrune(index){}

        void PruneInner() override;

    private:
        void init_graph(std::vector<Neighbor> &pool);

        void get_neighbors(const float *query, std::vector<Neighbor> &retset, std::vector<Neighbor> &fullset);

        void greedySearch(const unsigned query_id, const std::vector<Neighbor> &pool, std::vector<Neighbor> &retset);

        void sync_prune(unsigned q, float alpha, std::vector<Neighbor> &pool, boost::dynamic_bitset<> &flags,
                        SimpleNeighbor *cut_graph_);
    };


    // connect
    class IndexComponentConn : public IndexComponent {
    public:
        explicit IndexComponentConn(Index *index) : IndexComponent(index) {}

        virtual void ConnInner() = 0;
    };

    class IndexComponentConnDFS : public IndexComponentConn {
    public:
        explicit IndexComponentConnDFS(Index *index) : IndexComponentConn(index) {}

        void ConnInner() override;

    private:

        void tree_grow(const Parameters &parameter);

        void DFS(boost::dynamic_bitset<> &flag, unsigned root, unsigned &cnt);

        void findroot(boost::dynamic_bitset<> &flag, unsigned &root,
                      const Parameters &parameter);

        void get_neighbors(const float *query, const Parameters &parameter,
                           std::vector<Neighbor> &retset,
                           std::vector<Neighbor> &fullset);
    };

    class IndexComponentConnDFS_EXPAND : public IndexComponentConn {
    public:
        explicit IndexComponentConnDFS_EXPAND(Index *index) : IndexComponentConn(index) {}

        void ConnInner() override;

    private:
        void DFS_expand(const Parameters &parameter);
    };

    class IndexComponentConnDPG : public IndexComponentConn {
    public:
        explicit IndexComponentConnDPG(Index *index) : IndexComponentConn(index) {}

        void ConnInner() override;
    };


    // entry
    class IndexComponentEntry : public IndexComponent {
    public:
        explicit IndexComponentEntry(Index *index) : IndexComponent(index) {}

        virtual void EntryInner(unsigned query_id) = 0;
    };

    class IndexComponentEntryRand : public IndexComponentEntry {
    public:
        explicit IndexComponentEntryRand(Index *index) : IndexComponentEntry(index) {}

        void EntryInner(unsigned query_id) override;
    };

    class IndexComponentEntryKDT : public IndexComponentEntry {
    public:
        explicit IndexComponentEntryKDT(Index *index) : IndexComponentEntry(index) {}

        void EntryInner(unsigned query_id) override;

    private:
        void
        SearchNearLeaf(Index::Node *node, std::stack<Index::Node *> &st, size_t tree_id, size_t query_id,
                       size_t Nnode, std::vector<Neighbor> &retset);
    };

    class IndexComponentEntryCentroid : public IndexComponentEntry {
    public:
        explicit IndexComponentEntryCentroid(Index *index) : IndexComponentEntry(index) {}

        void EntryInner(unsigned query_id) override;
    };


    // route
    class IndexComponentRoute : public IndexComponent {
    public :
        explicit IndexComponentRoute(Index *index) : IndexComponent(index) {}

        virtual void RouteInner(unsigned query_id, unsigned K, unsigned *indices) = 0;
    };

    class IndexComponentRouteGreedy : public IndexComponentRoute {
    public:
        explicit IndexComponentRouteGreedy(Index *index) : IndexComponentRoute(index) {}

        void RouteInner(unsigned query_id, unsigned K, unsigned *indices) override;
    };


    class IndexBuilder {
    public:
        explicit IndexBuilder() {
            index = new Index();
        }

        ~IndexBuilder() {
            delete index;
        }

        enum TYPE {
            COARSE_NN_DESCENT, COARSE_KDT, COARSE_HASH, COARSE_MST,
            PRUNE_NSG, PRUNE_NSSG, PRUNE_HNSW, PRUNE_DPG, REFINE_NN_DESCENT, REFINE_VAMANA,
            CONN_DFS, CONN_DFS_EXPAND,
            ENTRY_RAND, ENTRY_KDT, ENTRY_CEN,
            ROUTER_GREEDY
        };

        IndexBuilder *load(char *data_file, char *query_file, char *ground_file, Parameters &parameters) {
            std::cout << "__LOAD DATA__" << std::endl;

            auto *a = new IndexComponentLoad();
            a->LoadInner(index, data_file, query_file, ground_file, parameters);

            e = std::chrono::high_resolution_clock::now();

            return this;
        }

        IndexBuilder *coarse(TYPE type) {
            IndexComponentCoarse *a = nullptr;

            if(type == COARSE_NN_DESCENT){
                std::cout << "__COARSE KNN : NN_DESCENT__" << std::endl;
                a = new IndexComponentCoarseNNDescent(index);
            }else if(type == COARSE_KDT){
                std::cout << "__COARSE KNN : KDT__" << std::endl;
                a = new IndexComponentCoarseKDT(index);
            }else if(type == COARSE_MST){
                std::cout << "__COARSE KNN : MST__" << std::endl;
                a = new IndexComponentCoarseMST(index);
            }else{
                std::cerr << "coarse KNN wrong type" << std::endl;
            }

            a->CoarseInner();

            e = std::chrono::high_resolution_clock::now();

            return this;
        }

        IndexBuilder *refine(TYPE type) {
            IndexComponentPrune *a = nullptr;

            if (type == PRUNE_NSG) {
                std::cout << "__REFINE : NSG__" << std::endl;
                a = new IndexComponentPruneNSG(index);
            } else if (type == PRUNE_NSSG) {
                std::cout << "__REFINE : NSSG__" << std::endl;
                a = new IndexComponentPruneNSSG(index);
            } else if (type == PRUNE_DPG) {
                std::cout << "__REFINE : DPG__" << std::endl;
                a = new IndexComponentPruneDPG(index);
            } else if (type == REFINE_NN_DESCENT) {
                std::cout << "__REFINE : NN_DESCENT__" << std::endl;
                a = new IndexComponentRefineNNDescent(index);
            } else {
                std::cerr << "PRUNE wrong type" << std::endl;
            }

            a->PruneInner();

            e = std::chrono::high_resolution_clock::now();

            return this;
        }

        IndexBuilder *connect(TYPE type) {
            std::cout << "__CONNECT__" << std::endl;

            IndexComponentConn *a = nullptr;
            if (type == CONN_DFS) {
                a = new IndexComponentConnDFS(index);
            } else if (type == CONN_DFS_EXPAND) {
                a = new IndexComponentConnDFS_EXPAND(index);
            } else {
                std::cerr << "PRUNE wrong type" << std::endl;
            }

            a->ConnInner();

            e = std::chrono::high_resolution_clock::now();

            return this;
        }

        IndexBuilder *eva(TYPE entry_type, TYPE route_type) {
            std::cout << "__EVA__" << std::endl;

            unsigned K = 50;
            unsigned L_start = 50;
            unsigned L_end = 500;
            unsigned experiment_num = 15;
            unsigned LI = (L_end - L_start) / experiment_num;


            for (unsigned L = L_start; L <= L_end; L += LI) {
                std::cout << "__ENTRY & Search__" << std::endl;
                if (L < K) {
                    std::cout << "search_L cannot be smaller than search_K! " << std::endl;
                    exit(-1);
                }

                index->param_.set<unsigned>("L_search", L);
                auto s1 = std::chrono::high_resolution_clock::now();
                std::vector<std::vector<unsigned>> res;
                unsigned distcount = 0;

                for (unsigned i = 0; i < index->query_num_; i++) {
                    std::vector<unsigned> tmp(K);

                    IndexComponentEntry *a = nullptr;
                    if (entry_type == ENTRY_RAND) {
                        a = new IndexComponentEntryRand(index);
                    } else if (entry_type == ENTRY_KDT) {
                        a = new IndexComponentEntryKDT(index);
                    } else if (entry_type == ENTRY_CEN) {
                        a = new IndexComponentEntryCentroid(index);
                    }else {
                        std::cerr << "entry wrong type" << std::endl;
                        exit(-1);
                    }
                    a->EntryInner(i);

                    IndexComponentRoute *b = nullptr;

                    if (route_type == ROUTER_GREEDY) {
                        b = new IndexComponentRouteGreedy(index);
                    } else {
                        std::cerr << "route wrong type" << std::endl;
                        exit(-1);
                    }
                    b->RouteInner(i, K, tmp.data());

                    res.push_back(tmp);
                }

                auto e1 = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> diff = e1 - s1;
                std::cout << "search time: " << diff.count() << "\n";

                //float speedup = (float)(index_->n_ * query_num) / (float)distcount;
                std::cout << "DistCount: " << distcount << std::endl;
                //结果评估
                int cnt = 0;
                for (unsigned i = 0; i < index->ground_num_; i++) {
                    for (unsigned j = 0; j < K; j++) {
                        unsigned k = 0;
                        for (; k < K; k++) {
                            if (res[i][j] == index->ground_data_[i * index->ground_dim_ + k])
                                break;
                        }
                        if (k == K)
                            cnt++;
                    }
                }
                float acc = 1 - (float) cnt / (index->ground_num_ * K);
                std::cout << K << " NN accuracy: " << acc << std::endl;
            }

            e = std::chrono::high_resolution_clock::now();

            return this;
        }

        std::chrono::duration<double> GetBuildTime() { return e - s; }

    private:
        Index *index;

        std::chrono::high_resolution_clock::time_point s;
        std::chrono::high_resolution_clock::time_point e;
    };


}

#endif //WEAVESS_INDEX_COMPONENT_H

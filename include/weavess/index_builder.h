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
        virtual void LoadInitInner(Index *index, char *data_file, char *query_file, char *ground_file, Parameters &parameters);
    };

    // init - 初始图入口点策略, 初始化全局 Index
    class IndexComponentInit : public IndexComponent {
    public:
        explicit IndexComponentInit(Index *index) : IndexComponent(index) {}

        virtual void InitInner() = 0;
    };

    // init - RAND
    class IndexComponentInitRandom : public IndexComponentInit {
    public:
        explicit IndexComponentInitRandom(Index *index) : IndexComponentInit(index) {}

        void InitInner() override;
    };

    // init - Hash
    class IndexComponentInitHash : public IndexComponentInit {
    public:
        explicit IndexComponentInitHash(Index *index) : IndexComponentInit(index) {}

        void InitInner() override;

    private:
        typedef std::vector<unsigned int> Codes;
        typedef std::unordered_map<unsigned int, std::vector<unsigned int> > HashBucket;
        typedef std::vector<HashBucket> HashTable;

        typedef std::vector<unsigned long> Codes64;
        typedef std::unordered_map<unsigned long, std::vector<unsigned int> > HashBucket64;
        typedef std::vector<HashBucket64> HashTable64;

        void verify();

        void LoadCode32(char* filename, std::vector<Codes>& baseAll);

        void LoadCode64(char* filename, std::vector<Codes64>& baseAll);

        void buildIndexImpl();

        void generateMask32();

        void generateMask64();

        void BuildHashTable32(int upbits, int lowbits, std::vector<Codes>& baseAll ,std::vector<HashTable>& tbAll);

        void BuildHashTable64(int upbits, int lowbits, std::vector<Codes64>& baseAll ,std::vector<HashTable64>& tbAll);
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

        void CoarseInner();

    private:
        void join() ;

        void update(const Parameters &parameters) ;

        void generate_control_set(std::vector<unsigned> &c, std::vector<std::vector<unsigned> > &v, unsigned N) ;

        void eval_recall(std::vector<unsigned> &ctrl_points, std::vector<std::vector<unsigned> > &acc_eval_set) ;
    };

    // coarse KDT
    class IndexComponentCoarseKDT : public IndexComponentCoarse {
    public:
        explicit IndexComponentCoarseKDT(Index *index) : IndexComponentCoarse(index) {}

        void CoarseInner();

    private:
        void meanSplit(std::mt19937& rng, unsigned* indices, unsigned count, unsigned& index, unsigned& cutdim, float& cutval);

        void planeSplit(unsigned *indices, unsigned count, unsigned cutdim, float cutval, unsigned &lim1, unsigned &lim2);

        int selectDivision(std::mt19937 &rng, float *v);

        void DFSbuild(Index::Node *node, std::mt19937 &rng, unsigned *indices, unsigned count, unsigned offset);

        void DFStest(unsigned level, unsigned dim, Index::Node *node);

        Index::Node *SearchToLeaf(Index::Node *node, size_t id);

        void mergeSubGraphs(size_t treeid, Index::Node *node);

        void getMergeLevelNodeList(Index::Node *node, size_t treeid, int deepth);
    };


    // prune — 剪枝方法
    class IndexComponentPrune : public IndexComponent {
    public:
        explicit IndexComponentPrune(Index *index) : IndexComponent(index) {}

        virtual void PruneInner() = 0;
    };

    class IndexComponentPruneNSG : public IndexComponentPrune {
    public:
        explicit IndexComponentPruneNSG(Index *index) : IndexComponentPrune(index) {}

        void PruneInner() override ;

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

        void get_neighbors(const unsigned q, const Parameters &parameter, std::vector<Neighbor> &pool);
    };

    class IndexComponentPruneDPG : public IndexComponentPrune {
    public:
        explicit IndexComponentPruneDPG(Index *index) : IndexComponentPrune(index) {}

        void PruneInner() override;

    private:
        void sync_prune(unsigned q, std::vector<Neighbor> &pool,
                        SimpleNeighbor *cut_graph_);

        void Link(const Parameters &parameters, SimpleNeighbor *cut_graph_);

        void get_neighbors(const unsigned q, const Parameters &parameter,
                           std::vector<Neighbor> &pool);

        void diversify_by_cut();

        void add_backward_edges();
    };

    class IndexComponentRefineNNDescent : public IndexComponentPrune {
    public:
        explicit IndexComponentRefineNNDescent(Index *index) : IndexComponentPrune(index) {}

        void PruneInner() override;
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

    class IndexComponentConnNSSG : public IndexComponentConn {
    public:
        explicit IndexComponentConnNSSG(Index *index) : IndexComponentConn(index) {}

        void ConnInner() override;

    private:
        void DFS_expand(const Parameters &parameter);
    };

    // entry
    class IndexComponentEntry : public IndexComponent{
    public:
        explicit IndexComponentEntry(Index *index) : IndexComponent(index) {}

        virtual void EntryInner(unsigned query_id) = 0;
    };

    class IndexComponentEntryRand : public IndexComponentEntry{
    public:
        explicit IndexComponentEntryRand(Index *index) : IndexComponentEntry(index) {}

        void EntryInner(unsigned query_id) override;
    };

    class IndexComponentEntryKDT : public IndexComponentEntry{
    public:
        explicit IndexComponentEntryKDT(Index *index) : IndexComponentEntry(index) {}

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

    // search
    class IndexComponentSearch : public IndexComponent {
    public:
        explicit IndexComponentSearch(Index *index) : IndexComponent(index) {}

        void EvaInner();

        virtual void
        SearchInner(unsigned query_id, size_t K, unsigned *indices, unsigned &distcount) = 0;
    };

    // 随机入口点 + 贪心
    class IndexComponentSearchRandom : public IndexComponentSearch {
    public:
        explicit IndexComponentSearchRandom(Index *index) : IndexComponentSearch(index) {}

        void SearchInner(unsigned query_id, size_t K, unsigned *indices, unsigned &distcount) override;
    };

    // 中点入口点 + 贪心
    class IndexComponentSearchNSG : public IndexComponentSearch {
    public:
        explicit IndexComponentSearchNSG(Index *index) : IndexComponentSearch(index) {}

        void SearchInner(unsigned query_id, size_t K, unsigned *indices, unsigned &distcount) override;
    };

    // hash桶入口点 + NN-Expand
    class IndexComponentSearchIEH : public IndexComponentSearch {
    public :
        explicit IndexComponentSearchIEH(Index *index) : IndexComponentSearch(index) {}

        void SearchInner(unsigned query_id, size_t K, unsigned *indices, unsigned &distcount) override;
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
            INIT_RAND,
            COARSE_NN_DESCENT, COARSE_KDT, COARSE_HASH,
            PRUNE_NSG, PRUNE_NSSG, PRUNE_HNSW, PRUNE_DPG, REFINE_NN_DESCENT,
            CONN_DFS, CONN_NSSG,
            ENTRY_RAND, ENTRY_KDT,
            SEARCH_NSG, SEARCH_IEH,
            ROUTER_GREEDY
        };

        IndexBuilder *load(char *data_file, char *query_file, char *ground_file, Parameters &parameters) {
            std::cout << "__Load Data__" << std::endl;

            auto *a = new IndexComponentLoad();
            a->LoadInitInner(index, data_file, query_file, ground_file, parameters);

            e = std::chrono::high_resolution_clock::now();

            return this;
        }

        IndexBuilder *init(TYPE type) {
            std::cout << "__Init Graph__" << std::endl;

            s = std::chrono::high_resolution_clock::now();

            IndexComponentInit *a = nullptr;
            switch (type) {
                case INIT_RAND:
                    a = new IndexComponentInitRandom(index);
                    break;
                default:
                    std::cerr << "init index wrong type" << std::endl;
            }

            a->InitInner();

            e = std::chrono::high_resolution_clock::now();

            return this;
        }

        IndexBuilder *coarse(TYPE type) {
            std::cout << "__COARSE KNN__" << std::endl;

            IndexComponentCoarse *a = nullptr;
            switch (type) {
                case COARSE_NN_DESCENT:
                    a = new IndexComponentCoarseNNDescent(index);
                    break;
                case COARSE_KDT:
                    a = new IndexComponentCoarseKDT(index);
                default:
                    std::cerr << "coarse KNN wrong type" << std::endl;
            }

            a->CoarseInner();

            e = std::chrono::high_resolution_clock::now();

            return this;
        }

        IndexBuilder *refine(TYPE type) {
            std::cout << "__PRUNE__" << std::endl;

            IndexComponentPrune *a = nullptr;
            if (type == PRUNE_NSG) {
                a = new IndexComponentPruneNSG(index);
            } else if (type == PRUNE_NSSG) {
                a = new IndexComponentPruneNSSG(index);
            } else if (type == PRUNE_DPG){
                a = new IndexComponentPruneDPG(index);
            } else if (type == REFINE_NN_DESCENT) {
                a = new IndexComponentRefineNNDescent(index);
            }else{
                std::cerr << "PRUNE wrong type" << std::endl;
            }

            a->PruneInner();

            e = std::chrono::high_resolution_clock::now();

            return this;
        }

        IndexBuilder *connect(TYPE type) {
            std::cout << "__CONNECT__" << std::endl;

            IndexComponentConn *a = nullptr;
            if(type == CONN_DFS){
                a = new IndexComponentConnDFS(index);
            } else if (type == CONN_NSSG){
                a = new IndexComponentConnNSSG(index);
            }else{
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

                std::cout << "__SEARCH__" << std::endl;
                for (unsigned i = 0; i < index->query_num_; i++) {
                    std::vector<unsigned> tmp(K);

                    IndexComponentEntry *a = nullptr;
                    if(entry_type == ENTRY_RAND){
                        a = new IndexComponentEntryRand(index);
                    }else{
                        std::cerr << "entry wrong type" << std::endl;
                        exit(-1);
                    }
                    a->EntryInner(i);

                    IndexComponentRoute *b = nullptr;

                    if(route_type == ROUTER_GREEDY) {
                        b = new IndexComponentRouteGreedy(index);
                    }else{
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

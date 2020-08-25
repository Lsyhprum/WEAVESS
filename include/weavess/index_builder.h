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

    // load
    class IndexComponentLoad : public IndexComponent {
    public :
        virtual void LoadInner(Index *index, char *data_file, Parameters &parameters);
    };

    // init
    class IndexComponentInit : public IndexComponent {
    public:
        explicit IndexComponentInit(Index *index) : IndexComponent(index) {}

        virtual void InitInner() = 0;
    };

    // 随机入口点
    class IndexComponentInitRandom : public IndexComponentInit {
    public:
        explicit IndexComponentInitRandom(Index *index) : IndexComponentInit(index) {}

        void InitInner() override;
    };

    class IndexComponentInitKDTree : public IndexComponentInit {
    public:
        explicit IndexComponentInitKDTree(Index *index) : IndexComponentInit(index) {}

        void InitInner() override;

    private:
        void
        planeSplit(unsigned *indices, unsigned count, unsigned cutdim, float cutval, unsigned &lim1, unsigned &lim2);

        int selectDivision(std::mt19937 &rng, float *v);

        void DFSbuild(Node *node, std::mt19937 &rng, unsigned *indices, unsigned count, unsigned offset);

        void DFStest(unsigned level, unsigned dim, Node *node);

        Node *SearchToLeaf(Node *node, size_t id);

        void mergeSubGraphs(size_t treeid, Node *node);

        void getMergeLevelNodeList(Node *node, size_t treeid, int deepth);
    };

    // coarse
    class IndexComponentCoarse : public IndexComponent {
    public:
        explicit IndexComponentCoarse(Index *index) : IndexComponent(index) {}

        virtual void CoarseInner() = 0;
    };

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

    // prune
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

    // search
    class IndexComponentEva : public IndexComponent {
    public:
        explicit IndexComponentEva(Index *index) : IndexComponent(index) {}

        void EvaInner(char *query_file, char *ground_truth_file);

        virtual void
        SearchInner(const float *query, const float *x, size_t K, const Parameters &parameters, unsigned *indices,
                    unsigned &distcount) = 0;
    };

    // 随机入口点 + 贪心
    class IndexComponentEvaRandom : public IndexComponentEva {
    public:
        explicit IndexComponentEvaRandom(Index *index) : IndexComponentEva(index) {}

        void SearchInner(const float *query, const float *x, size_t K, const Parameters &parameters, unsigned *indices,
                         unsigned &distcount) override;
    };

    // 中点入口点 + 贪心
    class IndexComponentEvaNSG : public IndexComponentEva {
    public:
        explicit IndexComponentEvaNSG(Index *index) : IndexComponentEva(index) {}

        void SearchInner(const float *query, const float *x, size_t K, const Parameters &parameters, unsigned *indices,
                         unsigned &distcount) override;
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
            INIT_RAND, INIT_KDT, INIT_IEH,
            COARSE_NN_DESCENT,
            PRUNE_NSG, PRUNE_NSSG, PRUNE_HNSW,
            CONN_DFS, CONN_NSSG,
            SEARCH_RAND, SEARCH_NSG
        };

        IndexBuilder *load(char *data_file, Parameters &parameters) {
            std::cout << "__Load Data__" << std::endl;

            auto *a = new IndexComponentLoad();
            a->LoadInner(index, data_file, parameters);

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
                case INIT_KDT:
                    a = new IndexComponentInitKDTree(index);
                    break;
                default:
                    std::cerr << "init index wrong type" << std::endl;
            }

            a->InitInner();

            return this;
        }

        IndexBuilder *coarse(TYPE type) {
            std::cout << "__COARSE KNN__" << std::endl;

            IndexComponentCoarse *a = nullptr;
            switch (type) {
                case COARSE_NN_DESCENT:
                    a = new IndexComponentCoarseNNDescent(index);
                    break;
                default:
                    std::cerr << "coarse KNN wrong type" << std::endl;
            }

            a->CoarseInner();

            e = std::chrono::high_resolution_clock::now();

            return this;
        }

        IndexBuilder *prune(TYPE type) {
            std::cout << "__PRUNE__" << std::endl;

            IndexComponentPrune *a = nullptr;
            if(type == PRUNE_NSG){
                a = new IndexComponentPruneNSG(index);
            }else if (type == PRUNE_NSSG) {
                a = new IndexComponentPruneNSSG(index);
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

        IndexBuilder *eva(TYPE type, char *query_data, char *ground_data) {
            std::cout << "__search__" << std::endl;

            IndexComponentEva *a = nullptr;
            switch (type) {
                case SEARCH_RAND:
                    a = new IndexComponentEvaRandom(index);
                    break;
                case SEARCH_NSG:
                    a = new IndexComponentEvaNSG(index);
                    break;
                default:
                    std::cerr << "coarse KNN wrong type" << std::endl;
            }

            a->EvaInner(query_data, ground_data);

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

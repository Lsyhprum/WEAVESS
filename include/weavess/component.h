//
// Created by MurphySL on 2020/9/14.
//

#ifndef WEAVESS_COMPONENT_H
#define WEAVESS_COMPONENT_H

#include "index.h"

namespace weavess {
    class Component {
    public:
        //explicit Component() = default;

        explicit Component(Index *index) : index(index) {}

        virtual ~Component() { delete index; }

    protected:
        Index *index = nullptr;
    };

    // load data
    class ComponentLoad : public Component {
    public:
        explicit ComponentLoad(Index *index) : Component(index){}

        virtual void LoadInner(char *data_file, char *query_file, char *ground_file, Parameters &parameters);
    };


    // serialization graph
    class ComponentSerialization : public Component {
    public:
        explicit ComponentSerialization(Index *index) : Component(index) {}

        virtual void SaveGraphInner(const char *filename) = 0;
    };

    class ComponentSerializationNNDescent : public Component {
    public:
        explicit ComponentSerializationNNDescent(Index *index) : Component(index) {}

        void SaveGraphInner(const char *filename);
        void LoadGraphInner(const char *filename);
    };


    // initial graph
    class ComponentInit : public Component {
    public:
        explicit ComponentInit(Index *index) : Component(index) {}

        virtual void InitInner() = 0;
    };

    class ComponentInitNNDescent : public ComponentInit {
    public:
        explicit ComponentInitNNDescent(Index *index) : ComponentInit(index) {}

        void InitInner() override;

    private:
        void SetConfigs();

        void init();

        void NNDescent();

        void join();

        void update();

        void generate_control_set(std::vector<unsigned> &c, std::vector<std::vector<unsigned> > &v, unsigned N);

        void eval_recall(std::vector<unsigned> &ctrl_points, std::vector<std::vector<unsigned> > &acc_eval_set);
    };

    class ComponentInitRandom : public ComponentInit {
    public:
        explicit ComponentInitRandom(Index *index) : ComponentInit(index) {}

        void InitInner() override;
    };

    class ComponentInitKDT : public ComponentInit {
    public:
        explicit ComponentInitKDT(Index *index) : ComponentInit(index) {}

        void InitInner() override;

    private:
        void SetConfigs();

        void meanSplit(std::mt19937& rng, unsigned* indices, unsigned count, unsigned& index, unsigned& cutdim, float& cutval);

        void planeSplit(unsigned* indices, unsigned count, unsigned cutdim, float cutval, unsigned& lim1, unsigned& lim2);

        int selectDivision(std::mt19937& rng, float* v);

        void DFSbuild(Index::EFANNA::Node* node, std::mt19937& rng, unsigned* indices, unsigned count, unsigned offset);

        void DFStest(unsigned level, unsigned dim, Index::EFANNA::Node* node);

        void getMergeLevelNodeList(Index::EFANNA::Node* node, size_t treeid, unsigned deepth);

        Index::EFANNA::Node* SearchToLeaf(Index::EFANNA::Node* node, size_t id);

        void mergeSubGraphs(size_t treeid, Index::EFANNA::Node* node);
    };

    class ComponentInitHCNNG : public ComponentInit {
    public:
        explicit ComponentInitHCNNG(Index *index) : ComponentInit(index) {}

        void InitInner() override;

    private:
        void SetConfigs();

        int rand_int(const int & min, const int & max);

        std::vector<std::vector< Index::Edge > >  create_exact_mst(int *idx_points, int left, int right, int max_mst_degree);

        void create_clusters(int *idx_points, int left, int right, std::vector<std::vector< Index::Edge > > &graph,
                             int minsize_cl, std::vector<omp_lock_t> &locks, int max_mst_degree);

        void sort_edges(std::vector<std::vector< Index::Edge > > &G);

        void print_stats_graph(std::vector<std::vector< Index::Edge > > &G);
    };


    // refine graph
    class ComponentRefine : public Component {
    public:
        explicit ComponentRefine(Index *index) : Component(index) {}

        virtual void RefineInner() = 0;
    };

    class ComponentRefineNSG : public ComponentRefine {
    public:
        explicit ComponentRefineNSG(Index *index) : ComponentRefine(index) {}

        void RefineInner() override;

    private:
        void Link(Index::SimpleNeighbor *cut_graph_);

        void InterInsert(unsigned n, unsigned range, std::vector<std::mutex> &locks, Index::SimpleNeighbor *cut_graph_);

        void SetConfigs();
    };

    class ComponentRefineNSSG : public ComponentRefine {
    public:
        explicit ComponentRefineNSSG(Index *index) : ComponentRefine(index) {}

        void RefineInner() override;

    private:
        void SetConfigs();

        void Link(Index::SimpleNeighbor *cut_graph_);

        void InterInsert(unsigned n, unsigned range, float threshold, std::vector<std::mutex> &locks, Index::SimpleNeighbor *cut_graph_);
    };

    class ComponentRefineDPG : public ComponentRefine {
    public:
        explicit ComponentRefineDPG(Index *index) : ComponentRefine(index) {}

        void RefineInner() override;

    private:
        void SetConfigs();

        void Link(Index::SimpleNeighbor *cut_graph_);
    };

    class ComponentRefineEFANNA : public ComponentRefine {
    public:
        explicit ComponentRefineEFANNA(Index *index) : ComponentRefine(index) {}

        void RefineInner() override;

    private:
        void SetConfigs();

        void init();

        void NNDescent();

        void join();

        void update();

        void generate_control_set(std::vector<unsigned> &c, std::vector<std::vector<unsigned> > &v, unsigned N);

        void eval_recall(std::vector<unsigned> &ctrl_points, std::vector<std::vector<unsigned> > &acc_eval_set);
    };

    class ComponentRefineHNSW : public ComponentRefine {
    public:
        explicit ComponentRefineHNSW(Index *index) : ComponentRefine(index) {}

        void RefineInner() override;

    private:
        int SetConfigs();

        void Build(bool reverse);

        int GetRandomSeedPerThread();

        int GetRandomNodeLevel();

        void InsertNode(Index::HnswNode* qnode, Index::VisitedList* visited_list);

        void SearchAtLayer(Index::HnswNode* qnode, Index::HnswNode* enterpoint, int level,
                                                Index::VisitedList* visited_list, std::priority_queue<Index::FurtherFirst>& result);

        void Link(Index::HnswNode* source, Index::HnswNode* target, int level);
    };

    class ComponentRefineVAMANA : public ComponentRefine {
    public:
        explicit ComponentRefineVAMANA(Index *index) : ComponentRefine(index) {}

        void RefineInner() override;

    private:
        void SetConfigs();

        void Link(Index::SimpleNeighbor *cut_graph_);
    };

    // entry
    class ComponentEntry : public Component {
    public:
        explicit ComponentEntry(Index *index) : Component(index) {}

        virtual void EntryInner() = 0;
    };

    class ComponentEntryCentroidNSG : public ComponentEntry {
    public:
        explicit ComponentEntryCentroidNSG(Index *index) : ComponentEntry(index) {}

        void EntryInner() override;

    private:
        void get_neighbors(const float *query, std::vector<Index::Neighbor> &retset, std::vector<Index::Neighbor> &fullset);
    };

    class ComponentEntryCentroidNSSG : public ComponentEntry {
    public:
        explicit ComponentEntryCentroidNSSG(Index *index) : ComponentEntry(index) {}

        void EntryInner() override;

    private:
        void get_neighbors(const float *query, std::vector<Index::Neighbor> &retset, std::vector<Index::Neighbor> &fullset);
    };

    // select candidate
    class ComponentCandidate : public Component {
    public:
        explicit ComponentCandidate(Index *index) : Component(index) {}

        virtual void CandidateInner(const unsigned query, const unsigned enter, Index::VisitedList *visited_list,
                                    std::vector<Index::Neighbor> &result, int level) = 0;
    };

    // L
    class ComponentCandidateNSG : public ComponentCandidate {
    public:
        explicit ComponentCandidateNSG(Index *index) : ComponentCandidate(index) {}

        void CandidateInner(const unsigned query, const unsigned enter, Index::VisitedList *visited_list,
                                    std::vector<Index::Neighbor> &result, int level) override;
    };

    class ComponentCandidateNSSG : public ComponentCandidate {
    public:
        explicit ComponentCandidateNSSG(Index *index) : ComponentCandidate(index) {}

        void CandidateInner(const unsigned query, const unsigned enter, Index::VisitedList *visited_list,
                       std::vector<Index::Neighbor> &result, int level) override;
    };

    class ComponentCandidateNone : public ComponentCandidate {
    public:
        explicit ComponentCandidateNone(Index *index) : ComponentCandidate(index) {}

        void CandidateInner(const unsigned query, const unsigned enter, Index::VisitedList *visited_list,
                            std::vector<Index::Neighbor> &result, int level) override;
    };


    // graph prune
    class ComponentPrune : public Component {
    public:
        explicit ComponentPrune(Index *index) : Component(index) {}

        virtual void PruneInner(unsigned q, unsigned range, Index::VisitedList *visited_list,
                                std::vector<Index::Neighbor> &pool, Index::SimpleNeighbor *cut_graph_, unsigned level) = 0;

        void Hnsw2Neighbor(unsigned range, std::priority_queue<Index::FurtherFirst> result) {
            int n = result.size();
            std::vector<Index::Neighbor> pool(n);
            std::vector<Index::FurtherFirst> tmp;
            for(int i = n - 1; i >= 0; i --) {
                Index::FurtherFirst f = result.top();
                pool[i] = Index::Neighbor(f.GetNode()->GetId(), f.GetDistance(), true);
                tmp.push_back(result.top());
                result.pop();
            }

            PruneInner(0, range, nullptr, pool, nullptr, 0);

            int start = pool.size() - 1;
            for(int i = 0; i < tmp.size(); i ++){
                for(int j = start; j >= 0; j --){
                    if(pool[j].id == tmp[i].GetNode()->GetId()){
                        result.push(tmp[i]);
                        start -= 1;
                        break;
                    }else{
                        if(pool[j].distance < tmp[i].GetDistance()){
                            break;
                        }
                    }
                }
            }

            std::vector<Index::Neighbor>().swap(pool);
            std::vector<Index::FurtherFirst>().swap(tmp);
        }
    };

    class ComponentPruneNSG : public ComponentPrune {
    public:
        explicit ComponentPruneNSG(Index *index) : ComponentPrune(index) {}

        void PruneInner(unsigned q, unsigned range, Index::VisitedList *visited_list,
                        std::vector<Index::Neighbor> &pool, Index::SimpleNeighbor *cut_graph_, unsigned level) override;
    };

    class ComponentPruneNSSG : public ComponentPrune {
    public:
        explicit ComponentPruneNSSG(Index *index) : ComponentPrune(index) {}

        void PruneInner(unsigned q, unsigned range, Index::VisitedList *visited_list,
                        std::vector<Index::Neighbor> &pool, Index::SimpleNeighbor *cut_graph_, unsigned level) override;
    };

    class ComponentPruneDPG : public ComponentPrune {
    public:
        explicit ComponentPruneDPG(Index *index) : ComponentPrune(index) {}

        void PruneInner(unsigned q, unsigned range, Index::VisitedList *visited_list,
                        std::vector<Index::Neighbor> &pool, Index::SimpleNeighbor *cut_graph_, unsigned level) override;
    };

    class ComponentPruneNaive : public ComponentPrune {
    public:
        explicit ComponentPruneNaive(Index *index) : ComponentPrune(index) {}

        void PruneInner(unsigned q, unsigned range, Index::VisitedList *visited_list,
                        std::vector<Index::Neighbor> &pool, Index::SimpleNeighbor *cut_graph_, unsigned level) override;
    };

    class ComponentPruneVAMANA : public ComponentPrune {
    public:
        explicit ComponentPruneVAMANA(Index *index) : ComponentPrune(index) {}

        void PruneInner(unsigned q, unsigned range, Index::VisitedList *visited_list,
                        std::vector<Index::Neighbor> &pool, Index::SimpleNeighbor *cut_graph_, unsigned level) override;
    };

    class ComponentPruneHeuristic : public ComponentPrune {
    public:
        explicit ComponentPruneHeuristic(Index *index) : ComponentPrune(index) {}

        void PruneInner(unsigned q, unsigned range, Index::VisitedList *visited_list,
                        std::vector<Index::Neighbor> &pool, Index::SimpleNeighbor *cut_graph_, unsigned level) override;
    };


    // graph conn
    class ComponentConn : public Component {
    public:
        explicit ComponentConn(Index *index) : Component(index) {}

        virtual void ConnInner() = 0;
    };

    class ComponentConnDFS : ComponentConn {
    public:
        explicit ComponentConnDFS(Index *index) : ComponentConn(index) {}

        void ConnInner();
    };


}

#endif //WEAVESS_COMPONENT_H

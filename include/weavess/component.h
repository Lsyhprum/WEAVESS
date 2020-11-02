//
// Created by MurphySL on 2020/10/23.
//

#ifndef WEAVESS_COMPONENT_H
#define WEAVESS_COMPONENT_H

#include "index.h"

namespace weavess {
    class Component {
    public:
        explicit Component(Index *index) : index(index) {}

        virtual ~Component() { delete index; }

    protected:
        Index *index = nullptr;
    };

    // load data
    class ComponentLoad : public Component {
    public:
        explicit ComponentLoad(Index *index) : Component(index) {}

        virtual void LoadInner(char *data_file, char *query_file, char *ground_file, Parameters &parameters);
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
    };

    class ComponentInitRand : public ComponentInit {
    public:
        explicit ComponentInitRand(Index *index) : ComponentInit(index) {}

        void InitInner() override;

    private:
        void SetConfigs();
    };

    class ComponentInitKDT : public ComponentInit {
    public:
        explicit ComponentInitKDT(Index *index) : ComponentInit(index) {}

        void InitInner() override;

    private:
        void SetConfigs();

        void meanSplit(std::mt19937 &rng, unsigned *indices, unsigned count, unsigned &index, unsigned &cutdim,
                       float &cutval);

        void
        planeSplit(unsigned *indices, unsigned count, unsigned cutdim, float cutval, unsigned &lim1, unsigned &lim2);

        int selectDivision(std::mt19937 &rng, float *v);

        void DFSbuild(Index::EFANNA::Node *node, std::mt19937 &rng, unsigned *indices, unsigned count, unsigned offset);

        void DFStest(unsigned level, unsigned dim, Index::EFANNA::Node *node);

        void getMergeLevelNodeList(Index::EFANNA::Node *node, size_t treeid, unsigned deepth);

        Index::EFANNA::Node *SearchToLeaf(Index::EFANNA::Node *node, size_t id);

        void mergeSubGraphs(size_t treeid, Index::EFANNA::Node *node);
    };

    class ComponentInitIEH : public ComponentInit {
    public:
        explicit ComponentInitIEH(Index *index) : ComponentInit(index) {}

        void InitInner() override;

    private:
        void LoadData(char *filename, Index::Matrix &dataset);

        void LoadHashFunc(char *filename, Index::Matrix &func);

        void LoadBaseCode(char *filename, Index::Codes &base);

        void BuildHashTable(int upbits, int lowbits, Index::Codes base, Index::HashTable &tb);

        void LoadKnnTable(char *filename, std::vector<Index::CandidateHeap2> &tb);

        void QueryToCode(Index::Matrix query, Index::Matrix func, Index::Codes &querycode);
    };

    class ComponentInitNSW : public ComponentInit {
    public:
        explicit ComponentInitNSW(Index *index) : ComponentInit(index) {}

        void InitInner() override;

    private:
        void SetConfigs();

        int GetRandomSeedPerThread();

        int GetRandomNodeLevel();

        void InsertNode(Index::HnswNode *qnode, Index::VisitedList *visited_list);

        void SearchAtLayer(Index::HnswNode *qnode, Index::HnswNode *enterpoint, int level,
                           Index::VisitedList *visited_list, std::priority_queue<Index::FurtherFirst> &result);

        void Link(Index::HnswNode *source, Index::HnswNode *target, int level);
    };

    class ComponentInitHNSW : public ComponentInit {
    public:
        explicit ComponentInitHNSW(Index *index) : ComponentInit(index) {}

        void InitInner() override;

    private:
        void SetConfigs();

        void Build(bool reverse);

        static int GetRandomSeedPerThread();

        int GetRandomNodeLevel();

        void InsertNode(Index::HnswNode *qnode, Index::VisitedList *visited_list);

        void SearchAtLayer(Index::HnswNode *qnode, Index::HnswNode *enterpoint, int level,
                           Index::VisitedList *visited_list, std::priority_queue<Index::FurtherFirst> &result);

        void Link(Index::HnswNode *source, Index::HnswNode *target, int level);
    };

    class ComponentInitANNG : public ComponentInit {
    public:
        explicit ComponentInitANNG(Index *index) : ComponentInit(index) {}

        void InitInner() override;

    private:
        void SetConfigs();

        void Build();

        void InsertNode(unsigned id);

        bool addEdge(unsigned target, unsigned addID, float dist);

        void truncateEdgesOptimally(unsigned id, size_t truncationSize);

        void Search(unsigned startId, unsigned query, std::vector<Index::SimpleNeighbor> &pool);
    };

    class ComponentInitSPTAG : public ComponentInit {
    public:
        explicit ComponentInitSPTAG(Index *index) : ComponentInit(index) {}

    protected:
        virtual void BuildTrees() = 0;

        void BuildGraph();

        unsigned rand(unsigned high, unsigned low = 0);

        void PartitionByTptree(std::vector<unsigned> &indices, const unsigned first, const unsigned last,
                               std::vector<std::pair<unsigned, unsigned>> &leaves);

        void AddNeighbor(unsigned idx, float dist, unsigned origin, unsigned size);

        static inline bool Compare(const Index::SimpleNeighbor &lhs, const Index::SimpleNeighbor &rhs);
    };

    class ComponentInitSPTAG_KDT : public ComponentInitSPTAG {
    public:
        explicit ComponentInitSPTAG_KDT(Index *index) : ComponentInitSPTAG(index) {}

        void InitInner() override;

    private:
        void SetConfigs();

        void BuildTrees() override;

        //void BuildGraph();

        void
        DivideTree(std::vector<unsigned> &indices, unsigned first, unsigned last, unsigned index, unsigned &iTreeSize);

        void ChooseDivision(Index::KDTNode &node, const std::vector<unsigned> &indices, const unsigned first,
                            const unsigned last);

        unsigned SelectDivisionDimension(const std::vector<float> &varianceValues);

        unsigned Subdivide(const Index::KDTNode &node, std::vector<unsigned> &indices, const unsigned first,
                           const unsigned last);
    };

    class ComponentInitSPTAG_BKT : public ComponentInitSPTAG {
    public:
        explicit ComponentInitSPTAG_BKT(Index *index) : ComponentInitSPTAG(index) {}

        void InitInner() override;

    protected:
        void BuildTrees() override;

    private:
        void SetConfigs();

        int KmeansClustering(std::vector<unsigned> &indices, const unsigned first, const unsigned last,
                             Index::KmeansArgs<float> &args, int samples = 1000);

        inline void InitCenters(std::vector<unsigned> &indices, const unsigned first, const unsigned last,
                                Index::KmeansArgs<float> &args, int samples, int tryIters);

        inline float KmeansAssign(std::vector<unsigned> &indices,
                                  const unsigned first, const unsigned last, Index::KmeansArgs<float> &args,
                                  const bool updateCenters, float lambda);

        float RefineCenters(Index::KmeansArgs<float> &args);
    };

    class ComponentInitFANNG : public ComponentInit {
    public:
        explicit ComponentInitFANNG(Index *index) : ComponentInit(index) {}

        void InitInner() override;

    private:
        void SetConfigs();

        void init();
    };


    // refine graph
    class ComponentRefine : public Component {
    public:
        explicit ComponentRefine(Index *index) : Component(index) {}

        virtual void RefineInner() = 0;
    };

    class ComponentRefineNNDescent : public ComponentRefine {
    public:
        explicit ComponentRefineNNDescent(Index *index) : ComponentRefine(index) {}

        void RefineInner() override;

    private:
        void SetConfigs();
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

    class ComponentRefineSSG : public ComponentRefine {
    public:
        explicit ComponentRefineSSG(Index *index) : ComponentRefine(index) {}

        void RefineInner() override;

    private:
        void SetConfigs();

        void Link(Index::SimpleNeighbor *cut_graph_);

        void InterInsert(unsigned n, unsigned range, float threshold, std::vector<std::mutex> &locks,
                         Index::SimpleNeighbor *cut_graph_);
    };

    class ComponentRefineDPG : public ComponentRefine {
    public:
        explicit ComponentRefineDPG(Index *index) : ComponentRefine(index) {}

        void RefineInner() override;

    private:
        void SetConfigs();

        void Link(Index::SimpleNeighbor *cut_graph_);

        void InterInsert(unsigned n, unsigned range, std::vector<std::mutex> &locks, Index::SimpleNeighbor *cut_graph_);
    };

    class ComponentRefineVAMANA : public ComponentRefine {
    public:
        explicit ComponentRefineVAMANA(Index *index) : ComponentRefine(index) {}

        void RefineInner() override;

    private:
        void SetConfigs();

        void Link(Index::SimpleNeighbor *cut_graph_);

        void InterInsert(unsigned n, unsigned range, std::vector<std::mutex> &locks,
                         Index::SimpleNeighbor *cut_graph_);
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

    class ComponentRefineONNG : public ComponentRefine {
    public:
        explicit ComponentRefineONNG(Index *index) : ComponentRefine(index) {}

        void RefineInner() override;
    };

    class ComponentRefineSPTAG_BKT : public ComponentRefine {
    public:
        explicit ComponentRefineSPTAG_BKT(Index *index) : ComponentRefine(index) {}

        void RefineInner() override;

    private:
        void Link(Index::SimpleNeighbor *cut_graph_);
    };

    class ComponentRefineSPTAG_KDT : public ComponentRefine {
    public:
        explicit ComponentRefineSPTAG_KDT(Index *index) : ComponentRefine(index) {}

        void RefineInner() override;

    private:
        void Link(Index::SimpleNeighbor *cut_graph_);
    };

    class ComponentRefineFANNG : public ComponentRefine {
    public:
        explicit ComponentRefineFANNG(Index *index) : ComponentRefine(index) {}

        void RefineInner() override;

    private:
        void Link(Index::SimpleNeighbor *cut_graph_);

        void SetConfigs();
    };


    // entry
    class ComponentRefineEntry : public Component {
    public:
        explicit ComponentRefineEntry(Index *index) : Component(index) {}

        virtual void EntryInner() = 0;
    };

    class ComponentRefineEntryCentroid : public ComponentRefineEntry {
    public:
        explicit ComponentRefineEntryCentroid(Index *index) : ComponentRefineEntry(index) {}

        void EntryInner() override;

    private:
        void
        get_neighbors(const float *query, std::vector<Index::Neighbor> &retSet, std::vector<Index::Neighbor> &fullset);
    };


    // select candidate
    class ComponentCandidate : public Component {
    public:
        explicit ComponentCandidate(Index *index) : Component(index) {}

        virtual void CandidateInner(unsigned query, unsigned enter, boost::dynamic_bitset<> flags,
                                    std::vector<Index::SimpleNeighbor> &pool) = 0;
    };

    class ComponentCandidateNSG : public ComponentCandidate {
    public:
        explicit ComponentCandidateNSG(Index *index) : ComponentCandidate(index) {}

        void CandidateInner(unsigned query, unsigned enter, boost::dynamic_bitset<> flags,
                            std::vector<Index::SimpleNeighbor> &result) override;
    };

    class ComponentCandidatePropagation2 : public ComponentCandidate {
    public:
        explicit ComponentCandidatePropagation2(Index *index) : ComponentCandidate(index) {}

        void CandidateInner(const unsigned query, const unsigned enter, boost::dynamic_bitset<> flags,
                            std::vector<Index::SimpleNeighbor> &pool) override;
    };

    class ComponentCandidateSPTAG_BKT : public ComponentCandidate {
    public:
        explicit ComponentCandidateSPTAG_BKT(Index *index) : ComponentCandidate(index) {}

        void CandidateInner(unsigned query, unsigned enter, boost::dynamic_bitset<> flags,
                            std::vector<Index::SimpleNeighbor> &result) override;
    };

    class ComponentCandidateSPTAG_KDT : public ComponentCandidate {
    public:
        explicit ComponentCandidateSPTAG_KDT(Index *index) : ComponentCandidate(index) {}

        void CandidateInner(unsigned query, unsigned enter, boost::dynamic_bitset<> flags,
                            std::vector<Index::SimpleNeighbor> &result) override;

    private:
        void KDTSearch(unsigned query, unsigned node, Index::Heap &m_NGQueue, Index::Heap &m_SPTQueue,
                       Index::OptHashPosVector &nodeCheckStatus,
                       unsigned &m_iNumberOfCheckedLeaves, unsigned &m_iNumberOfTreeCheckedLeaves);
    };


    // graph prune
    class ComponentPrune : public Component {
    public:
        explicit ComponentPrune(Index *index) : Component(index) {}

        virtual void PruneInner(unsigned query, unsigned range, boost::dynamic_bitset<> flags,
                                std::vector<Index::SimpleNeighbor> &pool,
                                Index::SimpleNeighbor *cut_graph_) = 0;

        void Hnsw2Neighbor(unsigned query, unsigned range, std::priority_queue<Index::FurtherFirst> &result) {
            int n = result.size();
            std::vector<Index::SimpleNeighbor> pool(n);
            std::unordered_map<int, Index::HnswNode *> tmp;

            for (int i = n - 1; i >= 0; i--) {
                Index::FurtherFirst f = result.top();
                pool[i] = Index::SimpleNeighbor(f.GetNode()->GetId(), f.GetDistance());
                tmp[f.GetNode()->GetId()] = f.GetNode();
                result.pop();
            }

            boost::dynamic_bitset<> flags;

            auto *cut_graph_ = new Index::SimpleNeighbor[range];

            PruneInner(query, range, flags, pool, cut_graph_);

            for (unsigned j = 0; j < range; j++) {
                if (cut_graph_[j].distance == -1) break;

                result.push(Index::FurtherFirst(tmp[cut_graph_[j].id], cut_graph_[j].distance));
            }

            std::vector<Index::SimpleNeighbor>().swap(pool);
            std::unordered_map<int, Index::HnswNode *>().swap(tmp);
        }
    };

    class ComponentPruneNaive : public ComponentPrune {
    public:
        explicit ComponentPruneNaive(Index *index) : ComponentPrune(index) {}

        void PruneInner(unsigned query, unsigned range, boost::dynamic_bitset<> flags,
                        std::vector<Index::SimpleNeighbor> &pool, Index::SimpleNeighbor *cut_graph_) override;
    };

    class ComponentPruneNSG : public ComponentPrune {
    public:
        explicit ComponentPruneNSG(Index *index) : ComponentPrune(index) {}

        void PruneInner(unsigned q, unsigned range, boost::dynamic_bitset<> flags,
                        std::vector<Index::SimpleNeighbor> &pool, Index::SimpleNeighbor *cut_graph_) override;
    };

    class ComponentPruneSSG : public ComponentPrune {
    public:
        explicit ComponentPruneSSG(Index *index) : ComponentPrune(index) {}

        void PruneInner(unsigned q, unsigned range, boost::dynamic_bitset<> flags,
                        std::vector<Index::SimpleNeighbor> &pool, Index::SimpleNeighbor *cut_graph_) override;
    };

    class ComponentPruneDPG : public ComponentPrune {
    public:
        explicit ComponentPruneDPG(Index *index) : ComponentPrune(index) {}

        void PruneInner(unsigned q, unsigned range, boost::dynamic_bitset<> flags,
                        std::vector<Index::SimpleNeighbor> &pool, Index::SimpleNeighbor *cut_graph_) override;
    };

    class ComponentPruneVAMANA : public ComponentPrune {
    public:
        explicit ComponentPruneVAMANA(Index *index) : ComponentPrune(index) {}

        void PruneInner(unsigned q, unsigned range, boost::dynamic_bitset<> flags,
                        std::vector<Index::SimpleNeighbor> &pool, Index::SimpleNeighbor *cut_graph_) override;
    };

    class ComponentPruneHeuristic : public ComponentPrune {
    public:
        explicit ComponentPruneHeuristic(Index *index) : ComponentPrune(index) {}

        void PruneInner(unsigned q, unsigned range, boost::dynamic_bitset<> flags,
                        std::vector<Index::SimpleNeighbor> &pool, Index::SimpleNeighbor *cut_graph_) override;
    };

    class ComponentPruneRNG : public ComponentPrune {
    public:
        explicit ComponentPruneRNG(Index *index) : ComponentPrune(index) {}

        void PruneInner(unsigned q, unsigned range, boost::dynamic_bitset<> flags,
                        std::vector<Index::SimpleNeighbor> &pool, Index::SimpleNeighbor *cut_graph_) override;
    };


    // graph conn
    class ComponentConn : public Component {
    public:
        explicit ComponentConn(Index *index) : Component(index) {}

        virtual void ConnInner() = 0;
    };

    class ComponentConnNSGDFS : ComponentConn {
    public:
        explicit ComponentConnNSGDFS(Index *index) : ComponentConn(index) {}

        void ConnInner();

    private:
        void tree_grow();

        void DFS(boost::dynamic_bitset<> &flag, unsigned root, unsigned &cnt);

        void findroot(boost::dynamic_bitset<> &flag, unsigned &root);

        void
        get_neighbors(const float *query, std::vector<Index::Neighbor> &retset, std::vector<Index::Neighbor> &fullset);
    };

    class ComponentConnSSGDFS : ComponentConn {
    public:
        explicit ComponentConnSSGDFS(Index *index) : ComponentConn(index) {}

        void ConnInner();
    };

    class ComponentConnReverse : ComponentConn {
    public:
        explicit ComponentConnReverse(Index *index) : ComponentConn(index) {}

        void ConnInner();
    };


    // search entry
    class ComponentSearchEntry : public Component {
    public:
        explicit ComponentSearchEntry(Index *index) : Component(index) {}

        virtual void SearchEntryInner(unsigned query, std::vector<Index::Neighbor> &pool) = 0;
    };

    class ComponentSearchEntryRand : public ComponentSearchEntry {
    public:
        explicit ComponentSearchEntryRand(Index *index) : ComponentSearchEntry(index) {}

        void SearchEntryInner(unsigned query, std::vector<Index::Neighbor> &pool) override;
    };

    class ComponentSearchEntryCentroid : public ComponentSearchEntry {
    public:
        explicit ComponentSearchEntryCentroid(Index *index) : ComponentSearchEntry(index) {}

        void SearchEntryInner(unsigned query, std::vector<Index::Neighbor> &pool) override;
    };

    class ComponentSearchEntrySubCentroid : public ComponentSearchEntry {
    public:
        explicit ComponentSearchEntrySubCentroid(Index *index) : ComponentSearchEntry(index) {}

        void SearchEntryInner(unsigned query, std::vector<Index::Neighbor> &pool) override;
    };

    class ComponentSearchEntryKDT : public ComponentSearchEntry {
    public:
        explicit ComponentSearchEntryKDT(Index *index) : ComponentSearchEntry(index) {}

        void SearchEntryInner(unsigned query, std::vector<Index::Neighbor> &pool) override;

    private:
        void getSearchNodeList(Index::Node *node, const float *q, unsigned int lsize, std::vector<Index::Node *> &vn);
    };

    class ComponentSearchEntryNone : public ComponentSearchEntry {
    public:
        explicit ComponentSearchEntryNone(Index *index) : ComponentSearchEntry(index) {}

        void SearchEntryInner(unsigned query, std::vector<Index::Neighbor> &pool) override;
    };

    class ComponentSearchEntryHash : public ComponentSearchEntry {
    public:
        explicit ComponentSearchEntryHash(Index *index) : ComponentSearchEntry(index) {}

        void SearchEntryInner(unsigned query, std::vector<Index::Neighbor> &pool) override;
    };


    // search route
    class ComponentSearchRoute : public Component {
    public:
        explicit ComponentSearchRoute(Index *index) : Component(index) {}

        virtual void RouteInner(unsigned query, std::vector<Index::Neighbor> &pool, std::vector<unsigned> &res) = 0;
    };

    class ComponentSearchRouteGreedy : public ComponentSearchRoute {
    public:
        explicit ComponentSearchRouteGreedy(Index *index) : ComponentSearchRoute(index) {}

        void RouteInner(unsigned query, std::vector<Index::Neighbor> &pool, std::vector<unsigned> &res) override;
    };

    class ComponentSearchRouteNSW : public ComponentSearchRoute {
    public:
        explicit ComponentSearchRouteNSW(Index *index) : ComponentSearchRoute(index) {}

        void RouteInner(unsigned query, std::vector<Index::Neighbor> &pool, std::vector<unsigned> &res) override;

    private:
        void SearchAtLayer(unsigned qnode, Index::HnswNode *enterpoint, int level,
                           Index::VisitedList *visited_list,
                           std::priority_queue<Index::FurtherFirst> &result);
    };

    class ComponentSearchRouteHNSW : public ComponentSearchRoute {
    public:
        explicit ComponentSearchRouteHNSW(Index *index) : ComponentSearchRoute(index) {}

        void RouteInner(unsigned query, std::vector<Index::Neighbor> &pool, std::vector<unsigned> &res) override;

    private:
        void SearchById_(unsigned query, Index::HnswNode* cur_node, float cur_dist, size_t k,
                         size_t ef_search, std::vector<std::pair<Index::HnswNode*, float>> &result);
    };

    class ComponentSearchRouteIEH : public ComponentSearchRoute {
    public:
        explicit ComponentSearchRouteIEH(Index *index) : ComponentSearchRoute(index) {}

        void RouteInner(unsigned query, std::vector<Index::Neighbor> &pool, std::vector<unsigned> &res) override;

    private:
        void HashTest(int upbits, int lowbits, Index::Codes querycode, Index::HashTable tb,
                      std::vector<std::vector<int> > &cands);
    };

    class ComponentSearchRouteBacktrack : public ComponentSearchRoute {
    public:
        explicit ComponentSearchRouteBacktrack(Index *index) : ComponentSearchRoute(index) {}

        void RouteInner(unsigned query, std::vector<Index::Neighbor> &pool, std::vector<unsigned> &res) override;
    };

}

#endif //WEAVESS_COMPONENT_H

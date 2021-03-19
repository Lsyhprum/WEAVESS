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

    class ComponentInitRandom : public ComponentInit {
    public:
        explicit ComponentInitRandom(Index *index) : ComponentInit(index) {}

        void InitInner() override;

    private:
        void SetConfigs();

        void GenRandom(std::mt19937 &rng, unsigned *addr, unsigned size);
    };

    class ComponentInitKNNG : public ComponentInit {
    public:
        explicit ComponentInitKNNG(Index *index) : ComponentInit(index) {}

        void InitInner() override;

    private:
        void SetConfigs();
    };


    class ComponentInitKDTree : public ComponentInit {
    public:
        explicit ComponentInitKDTree(Index *index) : ComponentInit(index) {}

        void InitInner() override;

    private:
        void SetConfigs();

        void meanSplit(std::mt19937 &rng, unsigned *indices, unsigned count, unsigned &index, unsigned &cutdim, float &cutval);

        void
        planeSplit(unsigned *indices, unsigned count, unsigned cutdim, float cutval, unsigned &lim1, unsigned &lim2);

        int selectDivision(std::mt19937 &rng, float *v);

        void DFSbuild(Index::EFANNA::Node *node, std::mt19937 &rng, unsigned *indices, unsigned count, unsigned offset);

        void getMergeLevelNodeList(Index::EFANNA::Node *node, size_t treeid, unsigned deepth);

        Index::EFANNA::Node *SearchToLeaf(Index::EFANNA::Node *node, size_t id);

        void mergeSubGraphs(size_t treeid, Index::EFANNA::Node *node);
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

        void InsertNode(Index::HnswNode *qnode, Index::VisitedList *visited_list);

        void Link(Index::HnswNode *source, Index::HnswNode *target, int level);

        void SearchAtLayer(Index::HnswNode *qnode, Index::HnswNode *enterpoint, int level,
                                                      Index::VisitedList *visited_list,
                                                      std::priority_queue<Index::FurtherFirst> &result);

        // VP-tree
        void Insert(const unsigned& new_value);

        void Insert(const unsigned& new_value, Index::VPNodePtr& root);

        void MakeVPTree(const std::vector<unsigned>& objects);

        void SplitLeafNode(const Index::VPNodePtr& parent_node, const size_t child_id, const unsigned& new_value);

        void InsertSplitLeafRoot(Index::VPNodePtr& root, const unsigned& new_value);

        void CollectObjects(const Index::VPNodePtr& node, std::vector<unsigned>& S);

        void RedistributeAmongLeafNodes(const Index::VPNodePtr& parent_node, const unsigned& new_value);

        const float MedianSumm(const std::vector<unsigned>& SS1, const std::vector<unsigned>& SS2, const unsigned& v) const;

        float Median(const unsigned& value, const std::vector<unsigned>::const_iterator it_begin,
                     const std::vector<unsigned>::const_iterator it_end);

        void RedistributeAmongNonLeafNodes(const Index::VPNodePtr& parent_node, const size_t k_id,
                                           const size_t k1_id, const unsigned& new_value);

        void Remove(const unsigned& query_value, const Index::VPNodePtr& node);

        void InsertSplitRoot(Index::VPNodePtr& root, const unsigned& new_value);

        void SplitNonLeafNode(const Index::VPNodePtr& parent_node, const size_t child_id, const unsigned& new_value);

        Index::VPNodePtr MakeVPTree(const std::vector<unsigned>& objects, const Index::VPNodePtr& parent);

        const unsigned& SelectVP(const std::vector<unsigned>& objects);
    };

    class ComponentInitSPTAG_KDT : public ComponentInit {
    public:
        explicit ComponentInitSPTAG_KDT(Index *index) : ComponentInit(index) {}

        void InitInner() override;

    private:
        void SetConfigs();

        void BuildTrees();

        void BuildGraph();

        void BuildInitKNNGraph();

        void PartitionByTptree(std::vector<int> &indices, const int first, const int last,
                               std::vector<std::pair<int, int>> &leaves);

        void AddNeighbor(int idx, float dist, int origin);

        static inline bool Compare(const Index::SimpleNeighbor &lhs, const Index::SimpleNeighbor &rhs);

        int rand(int high, int low = 0);

        void
        DivideTree(std::vector<int> &indices, int first, int last, int index, int &iTreeSize);

        void ChooseDivision(Index::KDTNode &node, const std::vector<int> &indices, const int first,
                            const int last);

        int SelectDivisionDimension(const std::vector<float> &varianceValues);

        int Subdivide(const Index::KDTNode &node, std::vector<int> &indices, const int first,
                           const int last);
    };

    class ComponentInitSPTAG_BKT : public ComponentInit {
    public:
        explicit ComponentInitSPTAG_BKT(Index *index) : ComponentInit(index) {}

        void InitInner() override;

    protected:
        void BuildTrees();

        void BuildGraph();

    private:
        void SetConfigs();

        int rand(int high, int low = 0);

        void PartitionByTptree(std::vector<int> &indices, const int first, const int last,
                               std::vector<std::pair<int, int>> &leaves);

        static inline bool Compare(const Index::SimpleNeighbor &lhs, const Index::SimpleNeighbor &rhs);

        void AddNeighbor(int idx, float dist, int origin);

        int KmeansClustering(std::vector<int> &indices, const int first, const int last,
                             Index::KmeansArgs<float> &args, int samples = 1000);

        inline void InitCenters(std::vector<int> &indices, const int first, const int last,
                                Index::KmeansArgs<float> &args, int samples, int tryIters);

        inline float KmeansAssign(std::vector<int> &indices,
                                  const int first, const int last, Index::KmeansArgs<float> &args,
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

    class ComponentInitHCNNG : public ComponentInit {
    public:
        explicit ComponentInitHCNNG(Index *index) : ComponentInit(index) {}

        void InitInner() override;

    private:
        void SetConfigs();

        void build_tree();

        int rand_int(const int & min, const int & max);

        std::vector<std::vector< Index::Edge > >  create_exact_mst(int *idx_points, int left, int right, int max_mst_degree);

        void create_clusters(int *idx_points, int left, int right, std::vector<std::vector< Index::Edge > > &graph,
                             int minsize_cl, std::vector<omp_lock_t> &locks, int max_mst_degree);

        void sort_edges(std::vector<std::vector< Index::Edge > > &G);

        void print_stats_graph(std::vector<std::vector< Index::Edge > > &G);

        void meanSplit(std::mt19937 &rng, unsigned *indices, unsigned count, unsigned &index1,
                                           unsigned &cutdim, float &cutval);

        void
        planeSplit(unsigned *indices, unsigned count, unsigned cutdim, float cutval, unsigned &lim1, unsigned &lim2);

        int selectDivision(std::mt19937 &rng, float *v);

        void DFSbuild(Index::EFANNA::Node *node, std::mt19937 &rng, unsigned *indices, unsigned count, unsigned offset);

        void getMergeLevelNodeList(Index::EFANNA::Node *node, size_t treeid, unsigned deepth);
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

        void init();

        void NNDescent();

        void join();

        void update();
        
        void generate_control_set(std::vector<unsigned> &c, std::vector<std::vector<unsigned> > &v, unsigned N);

        void eval_recall(std::vector<unsigned> &ctrl_points, std::vector<std::vector<unsigned> > &acc_eval_set);
    };

    class ComponentRefineKDRG : public ComponentRefine {
    public:
        explicit ComponentRefineKDRG(Index *index) : ComponentRefine(index) {}

        void RefineInner() override;

    private:
        bool GreedyReachabilityChecking(const unsigned x, const unsigned y, std::vector<std::vector<Index::SimpleNeighbor>>& cut_graph_, double& dist);
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

    class ComponentRefinePANNG : public ComponentRefine {
    public:
        explicit ComponentRefinePANNG(Index *index) : ComponentRefine(index) {}

        void RefineInner() override;

    private:
        bool hasEdge(size_t srcNodeID, size_t dstNodeID);

        void insert(std::vector<Index::SimpleNeighbor> &node, size_t edgeID, float edgeDistance);
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

    class ComponentRefineONNG : public ComponentRefine {
    public:
        explicit ComponentRefineONNG(Index *index) : ComponentRefine(index) {}

        void RefineInner() override;

    private:
        void Link(Index::SimpleNeighbor *cut_graph_);

        void SetConfigs();

        void extractGraph(std::vector<std::vector<Index::SimpleNeighbor>> &outGraph);

        void reconstructGraph(std::vector<std::vector<Index::SimpleNeighbor>> &outGraph);

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

    class ComponentCandidateSPTAG_KDT : public ComponentCandidate {
    public:
        explicit ComponentCandidateSPTAG_KDT(Index *index) : ComponentCandidate(index) {}

        void CandidateInner(unsigned query, unsigned enter, boost::dynamic_bitset<> flags,
                            std::vector<Index::SimpleNeighbor> &result) override;

    private:
        void KDTSearch(unsigned query, int node, Index::Heap &m_NGQueue, Index::Heap &m_SPTQueue,
                       Index::OptHashPosVector &nodeCheckStatus,
                       unsigned &m_iNumberOfCheckedLeaves, unsigned &m_iNumberOfTreeCheckedLeaves);
    };

    class ComponentCandidateSPTAG_BKT : public ComponentCandidate {
    public:
        explicit ComponentCandidateSPTAG_BKT(Index *index) : ComponentCandidate(index) {}

        void CandidateInner(unsigned query, unsigned enter, boost::dynamic_bitset<> flags,
                            std::vector<Index::SimpleNeighbor> &result) override;

    private:
        void BKTSearch(unsigned query, Index::Heap &m_NGQueue, Index::Heap &m_SPTQueue,
                       Index::OptHashPosVector &nodeCheckStatus,
                       unsigned &m_iNumberOfCheckedLeaves, unsigned &m_iNumberOfTreeCheckedLeaves, int p_limits);
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

            auto *cut_graph_ = new Index::SimpleNeighbor[index->getBaseLen() * range];

            PruneInner(query, range, flags, pool, cut_graph_);

            for (unsigned j = 0; j < range; j++) {
                if (cut_graph_[range * query + j].distance == -1) break;

                result.push(Index::FurtherFirst(tmp[cut_graph_[range * query + j].id], cut_graph_[range * query + j].distance));
            }

            delete[] cut_graph_;

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

    class ComponentSearchEntryKDTSingle : public ComponentSearchEntry {
    public:
        explicit ComponentSearchEntryKDTSingle(Index *index) : ComponentSearchEntry(index) {}

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

    class ComponentSearchEntryVPT : public ComponentSearchEntry {
    public:
        explicit ComponentSearchEntryVPT(Index *index) : ComponentSearchEntry(index) {}

        void SearchEntryInner(unsigned query, std::vector<Index::Neighbor> &pool) override;

    private:
        void Search(const unsigned& query_value, const size_t count, std::multimap<float, unsigned> &pool,
                                             const Index::VPNodePtr& node, float& q);
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
        // void SearchById_(unsigned query, Index::HnswNode* cur_node, float cur_dist, size_t k,
        //                  size_t ef_search, std::vector<std::pair<Index::HnswNode*, float>> &result);
        void SearchAtLayer(unsigned qnode, Index::HnswNode *enterpoint, int level,
                           Index::VisitedList *visited_list,
                           std::priority_queue<Index::FurtherFirst> &result);
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

    class ComponentSearchRouteSPTAG_KDT : public ComponentSearchRoute {
    public:
        explicit ComponentSearchRouteSPTAG_KDT(Index *index) : ComponentSearchRoute(index) {}

        void RouteInner(unsigned query, std::vector<Index::Neighbor> &pool, std::vector<unsigned> &res) override;

    private:
        void KDTSearch(unsigned query, int node, Index::Heap &m_NGQueue, Index::Heap &m_SPTQueue,
                       Index::OptHashPosVector &nodeCheckStatus,
                       unsigned &m_iNumberOfCheckedLeaves, unsigned &m_iNumberOfTreeCheckedLeaves);
    };

    class ComponentSearchRouteSPTAG_BKT : public ComponentSearchRoute {
    public:
        explicit ComponentSearchRouteSPTAG_BKT(Index *index) : ComponentSearchRoute(index) {}

        void RouteInner(unsigned query, std::vector<Index::Neighbor> &pool, std::vector<unsigned> &res) override;

    private:
        void BKTSearch(unsigned int query, Index::Heap &m_NGQueue,
                  Index::Heap &m_SPTQueue, Index::OptHashPosVector &nodeCheckStatus,
                  unsigned int &m_iNumberOfCheckedLeaves,
                  unsigned int &m_iNumberOfTreeCheckedLeaves,
                  int p_limits);
    };

    class ComponentSearchRouteGuided : public ComponentSearchRoute {
    public:
        explicit ComponentSearchRouteGuided(Index *index) : ComponentSearchRoute(index) {}

        void RouteInner(unsigned query, std::vector<Index::Neighbor> &pool, std::vector<unsigned> &res) override;
    };

    class ComponentSearchRouteNGT : public ComponentSearchRoute {
    public:
        explicit ComponentSearchRouteNGT(Index *index) : ComponentSearchRoute(index) {}

        void RouteInner(unsigned query, std::vector<Index::Neighbor> &pool, std::vector<unsigned> &res) override;
    };
}

#endif //WEAVESS_COMPONENT_H

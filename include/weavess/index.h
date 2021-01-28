//
// Created by MurphySL on 2020/10/23.
//

#ifndef WEAVESS_INDEX_H
#define WEAVESS_INDEX_H

#define PARALLEL

#define FLT_EPSILON 1.19209290E-07F

// NN-Descent
#define CONTROL_NUM 100

// IEH
#define MAX_ROWSIZE 1024
#define HASH_RADIUS 1
#define DEPTH 16 //smaller than code length
#define INIT_NUM 5500

// SPTAG
#define ALIGN 32
#define _rotl(value, bits) ((value << bits) | (value >> (sizeof(value)*8 - bits)))

// HCNNG
#define not_in_set(_elto, _set) (_set.find(_elto)==_set.end())

// NGT
#define NGT_SEED_SIZE 5

#include <omp.h>
#include <mutex>
#include <queue>
#include <stack>
#include <thread>
#include <vector>
#include <chrono>
#include <cstring>
#include <cfloat>
#include <fstream>
#include <cassert>
#include <iostream>
#include <algorithm>
#include <unordered_set>
#include <boost/dynamic_bitset.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/heap/d_ary_heap.hpp>
#include "util.h"
#include "policy.h"
#include "distance.h"
#include "parameters.h"
#include "CommonDataStructure.h"
#include <mm_malloc.h>
#include <stdlib.h>

namespace weavess {

    class NNDescent {
    public:
        unsigned K;
        unsigned S;
        unsigned R;
        unsigned L;
        unsigned ITER;

        struct Neighbor {
            unsigned id;
            float distance;
            bool flag;

            Neighbor() = default;

            Neighbor(unsigned id, float distance, bool f) : id{id}, distance{distance}, flag(f) {}

            inline bool operator<(const Neighbor &other) const {
                return distance < other.distance;
            }

            inline bool operator>(const Neighbor &other) const {
                return distance > other.distance;
            }
        };

        typedef std::lock_guard<std::mutex> LockGuard;

        struct nhood {
            std::mutex lock;
            std::vector<Neighbor> pool;
            unsigned M;

            std::vector<unsigned> nn_old;
            std::vector<unsigned> nn_new;
            std::vector<unsigned> rnn_old;
            std::vector<unsigned> rnn_new;

            nhood() {}

            nhood(unsigned l, unsigned s) {
                M = s;
                nn_new.resize(s * 2);
                nn_new.reserve(s * 2);
                pool.reserve(l);
            }

            nhood(unsigned l, unsigned s, std::mt19937 &rng, unsigned N) {
                M = s;
                nn_new.resize(s * 2);
                GenRandom(rng, &nn_new[0], (unsigned) nn_new.size(), N);
                nn_new.reserve(s * 2);
                pool.reserve(l);
            }

            nhood(const nhood &other) {
                M = other.M;
                std::copy(other.nn_new.begin(), other.nn_new.end(), std::back_inserter(nn_new));
                nn_new.reserve(other.nn_new.capacity());
                pool.reserve(other.pool.capacity());
            }

            void insert(unsigned id, float dist) {
                LockGuard guard(lock);
                if (dist > pool.front().distance) return;
                for (unsigned i = 0; i < pool.size(); i++) {
                    if (id == pool[i].id)return;
                }
                if (pool.size() < pool.capacity()) {
                    pool.push_back(Neighbor(id, dist, true));
                    std::push_heap(pool.begin(), pool.end());
                } else {
                    std::pop_heap(pool.begin(), pool.end());
                    pool[pool.size() - 1] = Neighbor(id, dist, true);
                    std::push_heap(pool.begin(), pool.end());
                }

            }

            template<typename C>
            void join(C callback) const {
                for (unsigned const i: nn_new) {
                    for (unsigned const j: nn_new) {
                        if (i < j) {
                            callback(i, j);
                        }
                    }
                    for (unsigned j: nn_old) {
                        callback(i, j);
                    }
                }
            }
        };

        static inline int InsertIntoPool(Neighbor *addr, unsigned K, Neighbor nn) {
            // find the location to insert
            int left = 0, right = K - 1;
            if (addr[left].distance > nn.distance) {
                memmove((char *) &addr[left + 1], &addr[left], K * sizeof(Neighbor));
                addr[left] = nn;
                return left;
            }
            if (addr[right].distance < nn.distance) {
                addr[K] = nn;
                return K;
            }
            while (left < right - 1) {
                int mid = (left + right) / 2;
                if (addr[mid].distance > nn.distance)right = mid;
                else left = mid;
            }
            //check equal ID

            while (left > 0) {
                if (addr[left].distance < nn.distance) break;
                if (addr[left].id == nn.id) return K + 1;
                left--;
            }
            if (addr[left].id == nn.id || addr[right].id == nn.id)return K + 1;
            memmove((char *) &addr[right + 1], &addr[right], (K - right) * sizeof(Neighbor));
            addr[right] = nn;
            return right;
        }

        typedef std::vector<nhood> KNNGraph;
        KNNGraph graph_;
    };

    class NSG {
    public:
        unsigned R_refine;
        unsigned L_refine;
        unsigned C_refine;

        unsigned ep_;
        unsigned width;
    };

    class SSG {
    public:
        float A;
        unsigned n_try;
        //unsigned width;

        std::vector<unsigned> eps_;
        unsigned test_min = INT_MAX;
        unsigned test_max = 0;
        long long test_sum = 0;
    };

    class DPG {
    public:
        unsigned L_dpg;
    };

    class VAMANA {
    public:
        float alpha = 1;
    };

    class EFANNA {
    public:
        struct Node {
            int DivDim;
            float DivVal;
            size_t StartIdx, EndIdx;
            unsigned treeid;
            EFANNA::Node *Lchild, *Rchild;
            bool visit = false;

            ~Node() {
                if (Lchild != nullptr) Lchild->~Node();
                if (Rchild != nullptr) Rchild->~Node();
            }
        };

        struct Candidate {
            size_t row_id;
            float distance;

            Candidate(const size_t row_id, const float distance) : row_id(row_id), distance(distance) {}

            bool operator>(const Candidate &rhs) const {
                if (this->distance == rhs.distance) {
                    return this->row_id > rhs.row_id;
                }
                return this->distance > rhs.distance;
            }

            bool operator<(const Candidate &rhs) const {
                if (this->distance == rhs.distance) {
                    return this->row_id < rhs.row_id;
                }
                return this->distance < rhs.distance;
            }
        };

        typedef std::set<Candidate, std::greater<Candidate> > CandidateHeap;
        std::vector<CandidateHeap> knn_graph;
        size_t TNS = 10; //tree node size

        enum {
            /**
             * To improve efficiency, only SAMPLE_NUM random values are used to
             * compute the mean and variance at each level when building a tree.
             * A value of 100 seems to perform as well as using all values.
             */
            SAMPLE_NUM = 100,
            /**
             * Top random dimensions to consider
             *
             * When creating random trees, the dimension on which to subdivide is
             * selected at random from among the top RAND_DIM dimensions with the
             * highest variance.  A value of 5 works well.
             */
            RAND_DIM = 5
        };

        std::vector<Node *> tree_roots_;
        std::vector<std::pair<Node *, size_t> > mlNodeList;
        std::vector<std::vector<unsigned> > LeafLists;
        omp_lock_t rootlock;
        bool error_flag = false;
        int max_deepth = 0x0fffffff;

        unsigned mLevel;
        unsigned nTrees;
    };

    class IEH {
    public:
        template<typename T>
        struct Candidate2 {
            size_t row_id;
            T distance;
            Candidate2(const size_t row_id, const T distance): row_id(row_id), distance(distance) { }

            bool operator >(const Candidate2& rhs) const {
                if (this->distance == rhs.distance) {
                    return this->row_id > rhs.row_id;
                }
                return this->distance > rhs.distance;
            }
        };

        typedef std::vector<std::vector<float> > Matrix;
        typedef std::vector<unsigned int> Codes;
        typedef std::unordered_map<unsigned int, std::vector<unsigned int> > HashBucket;
        typedef std::vector<HashBucket> HashTable;
        typedef std::set<Candidate2<float>, std::greater<Candidate2<float> > > CandidateHeap2;

        Matrix func;
        Matrix train;
        Matrix test;
        Codes querycode;
        int UpperBits = 8;
        int LowerBits = 8; //change with code length:code length = up + low;
        HashTable tb;

        std::vector<CandidateHeap2> knntable;
    };

    class NSW {
    public:
        unsigned NN_ ;
        unsigned ef_construction_ = 150;    //l
        unsigned n_threads_ = 32;
    };

    class HNSW {
    public:
        unsigned m_ = 12;       // k
        unsigned max_m_ = 12;
        unsigned max_m0_ = 24;
        int mult;
        float level_mult_ = 1 / log(1.0*m_);

        int max_level_ = 0;
        mutable std::mutex max_level_guard_;

        template <typename KeyType, typename DataType>
        class MinHeap {
        public:
            class Item {
            public:
                KeyType key;
                DataType data;
                Item() {}
                Item(const KeyType& key) :key(key) {}
                Item(const KeyType& key, const DataType& data) :key(key), data(data) {}
                bool operator<(const Item& i2) const {
                    return key > i2.key;
                }
            };

            MinHeap() {
            }

            const KeyType top_key() {
                if (v_.size() <= 0) return 0.0;
                return v_[0].key;
            }

            Item top() {
                if (v_.size() <= 0) throw std::runtime_error("[Error] Called top() operation with empty heap");
                return v_[0];
            }

            void pop() {
                std::pop_heap(v_.begin(), v_.end());
                v_.pop_back();
            }

            void push(const KeyType& key, const DataType& data) {
                v_.emplace_back(Item(key, data));
                std::push_heap(v_.begin(), v_.end());
            }

            size_t size() {
                return v_.size();
            }

        private:
            std::vector<Item> v_;
        };

        class HnswNode {
        public:
            explicit HnswNode(int id, int level, size_t max_m, size_t max_m0)
                    : id_(id), level_(level), max_m_(max_m), max_m0_(max_m0), friends_at_layer_(level+1) {
                for (int i = 1; i <= level; ++i)
                    friends_at_layer_[i].reserve(max_m_ + 1);

                friends_at_layer_[0].reserve(max_m0_ + 1);
            }

            inline int GetId() const { return id_; }
            inline void SetId(int id) {id_ = id; }
            inline int GetLevel() const { return level_; }
            inline void SetLevel(int level) {level_ = level; }
            inline size_t GetMaxM() const { return max_m_; }
            inline size_t GetMaxM0() const { return max_m0_; }

            inline std::vector<HnswNode*>& GetFriends(int level) { return friends_at_layer_[level]; }
            inline void SetFriends(int level, std::vector<HnswNode*>& new_friends) {
                if (level >= friends_at_layer_.size())
                friends_at_layer_.resize(level + 1);

                friends_at_layer_[level].swap(new_friends);
            }
            inline std::mutex& GetAccessGuard() { return access_guard_; }

            // 1. The list of friends is sorted
            // 2. bCheckForDup == true addFriend checks for duplicates using binary searching
            inline void AddFriends(HnswNode* element, bool bCheckForDup) {
                std::unique_lock<std::mutex> lock(access_guard_);

                if(bCheckForDup) {
                    auto it = std::lower_bound(friends_at_layer_[0].begin(), friends_at_layer_[0].end(), element);
                    if(it == friends_at_layer_[0].end() || (*it) != element) {
                        friends_at_layer_[0].insert(it, element);
                    }
                }else{
                    friends_at_layer_[0].push_back(element);
                }
            }

        private:
            int id_;
            int level_;
            size_t max_m_;
            size_t max_m0_;

            std::vector<std::vector<HnswNode*> > friends_at_layer_;
            std::mutex access_guard_;
        };

        class FurtherFirst {
        public:
            FurtherFirst(HnswNode* node, float distance) : node_(node), distance_(distance) {}
            inline float GetDistance() const { return distance_; }
            inline HnswNode* GetNode() const { return node_; }
            bool operator< (const FurtherFirst& n) const {
                return (distance_ < n.GetDistance());
            }
        private:
            HnswNode* node_;
            float distance_;
        };

        class CloserFirst {
        public:
            CloserFirst(HnswNode* node, float distance) : node_(node), distance_(distance) {}
            inline float GetDistance() const { return distance_; }
            inline HnswNode* GetNode() const { return node_; }
            bool operator< (const CloserFirst& n) const {
                return (distance_ > n.GetDistance());
            }
        private:
            HnswNode* node_;
            float distance_;
        };

        class VisitedList {
        public:
            VisitedList(unsigned size) : size_(size), mark_(1) {
                visited_ = new unsigned int[size_];
                memset(visited_, 0, sizeof(unsigned int) * size_);
            }

            ~VisitedList() { delete[] visited_; }

            inline bool Visited(unsigned int index) const { return visited_[index] == mark_; }

            inline bool NotVisited(unsigned int index) const { return visited_[index] != mark_; }

            inline void MarkAsVisited(unsigned int index) { visited_[index] = mark_; }

            inline void Reset() {
                if (++mark_ == 0) {
                    mark_ = 1;
                    memset(visited_, 0, sizeof(unsigned int) * size_);
                }
            }

            inline unsigned int *GetVisited() { return visited_; }

            inline unsigned int GetVisitMark() { return mark_; }

        private:
            unsigned int *visited_;
            unsigned int size_;
            unsigned int mark_;
        };

        typedef typename std::pair<HnswNode*, float> IdDistancePair;
        struct IdDistancePairMinHeapComparer {
            bool operator()(const IdDistancePair& p1, const IdDistancePair& p2) const {
                return p1.second > p2.second;
            }
        };
        typedef typename boost::heap::d_ary_heap<IdDistancePair, boost::heap::arity<4>, boost::heap::compare<IdDistancePairMinHeapComparer> > IdDistancePairMinHeap;

        HnswNode* enterpoint_ = nullptr;
        std::vector<HnswNode*> nodes_;
    };

    class NGT {
    public :
        class JobQueue : public std::deque<unsigned> {
        public:
            JobQueue() {
            }
            ~JobQueue() {
            }
            bool isDeficient() { return std::deque<unsigned>::size() <= requestSize; }
            bool isEmpty() { return std::deque<unsigned>::size() == 0; }
            bool isFull() { return std::deque<unsigned>::size() >= maxSize; }
            void setRequestSize(int s) { requestSize = s; }
            void setMaxSize(int s) { maxSize = s; }
            unsigned int	requestSize;
            unsigned int	maxSize;
        };

        class InputJobQueue : public JobQueue {
        public:
            InputJobQueue() {
                isTerminate = false;
                underPushing = false;
                pushedSize = 0;
            }

            void popFront(unsigned &d) {
                while (JobQueue::isEmpty()) {
                    if (isTerminate) {
                        std::cerr << "Thread::termination" << std::endl;
                    }
                }
                d = std::deque<unsigned>::front();
                std::deque<unsigned>::pop_front();
                return;
            }

            void popFront(std::deque<unsigned> &d, size_t s) {
                while (JobQueue::isEmpty()) {
                    if (isTerminate) {
                        std::cerr << "Thread::termination" << std::endl;
                    }
                }
                for (size_t i = 0; i < s; i++) {
                    d.push_back(std::deque<unsigned>::front());
                    std::deque<unsigned>::pop_front();
                    if (JobQueue::isEmpty()) {
                        break;
                    }
                }
                return;
            }

            void pushBack(unsigned &data) {
                if (!underPushing) {
                    underPushing = true;
                    pushedSize = 0;
                }
                pushedSize++;
                std::deque<unsigned>::push_back(data);
            }

            void pushBackEnd() {
                underPushing = false;
            }

            void terminate() {
                if (underPushing || !JobQueue::isEmpty()) {
                    std::cerr << "Thread::teminate:Under pushing!" << std::endl;
                }
                isTerminate = true;
            }

            bool		isTerminate;
            bool		underPushing;
            size_t		pushedSize;
        };

        class OutputJobQueue : public JobQueue {
        public:
            void pushBack(unsigned &data) {
                std::deque<unsigned>::push_back(data);
                if (!JobQueue::isFull()) {
                    return;
                }
            }
        };

        class VPNode {
        public:
            typedef boost::shared_ptr<VPNode> VPNodePtr;
            typedef std::vector<VPNodePtr> ChildList;
            typedef std::vector<float> DistanceList;
            typedef std::vector<unsigned> ObjectsList; // Objects, if this is the leaf node

        public:
            VPNode(): m_leaf_node(false)
                    , m_branches_count(0)
            {
            }


            VPNode(const VPNode& orig): m_leaf_node(false)
                    , m_branches_count(0)
            {
            }

            VPNode(const unsigned& value): m_leaf_node(true)
                    , m_branches_count(0)
                    , m_value(value)
            {
                //m_objects_list.push_back(value);
            }

            virtual ~VPNode(){
            }

            bool get_leaf_node() const{
                return m_leaf_node;
            }

            void set_leaf_node(const bool leaf_node){
                m_leaf_node = leaf_node;
            }

            size_t get_branches_count() const{
                return m_branches_count;
            }

            void set_branches_count(size_t new_size){
                m_branches_count = new_size;
            }

            size_t get_objects_count() const{
                return m_objects_list.size();
            }

            const VPNodePtr& get_parent() const{
                return m_parent;
            }
            void set_parent(const VPNodePtr& parent){
                m_parent = parent;
            }

            void AddChild(const size_t child_pos, const VPNodePtr& child){
                /*
                if(m_child_list.size() <= child_pos)
                {
                    m_child_list.reserve(child_pos + 1);
                    m_mu_list.reserve(child_pos + 1);
                }
                */
                //m_child_list[child_pos] = child;
                m_child_list.push_back(child);

                //(*m_child_list.rbegin())->set_parent(shared_from_this());

                if(get_branches_count() != 1)
                    m_mu_list.push_back(static_cast<float>(0));

                ++m_branches_count;
            }

            void AddObject(const unsigned& new_object){
                m_leaf_node = true;
                m_objects_list.push_back(new_object);
            }
            bool DeleteObject(const unsigned& object){
                typename ObjectsList::iterator it_find = std::find(m_objects_list.begin(),
                                                                   m_objects_list.end(),
                                                                   object);
                if(it_find != m_objects_list.end())
                {
                    m_objects_list.erase(it_find);
                    return true;
                }

                return false;
            }

            const unsigned& get_value() const{
                return m_value;
            }
            void set_value(const unsigned& new_value){
                m_value = new_value;
            }


        public:
            DistanceList m_mu_list;		// "mu"  and child lists for vp-tree like
            ChildList m_child_list;		// |chld| MU |chld| MU |chld|
            VPNodePtr m_parent;

            ObjectsList m_objects_list;		// Objects, if this is the leaf node

        private:
            bool m_leaf_node;	// Is leaf node
            size_t m_branches_count;	// Number of internal branches
            unsigned m_value;	// Storage value
        };

        struct TreeStatistic{
            uint64_t distance_count;
            uint64_t distance_threshold_count;
            uint64_t search_jump;

            void clear()
            {
                distance_count = 0;
                distance_threshold_count = 0;
                search_jump = 0;
            }

            TreeStatistic()
                    : distance_count(0)
                    , distance_threshold_count(0)
                    , search_jump(0)
            {
            }
        };

        template <typename ValType, typename DistanceObtainer>
        class ValueSorter
        {
        public:
            ValueSorter(const ValType& main_val, const DistanceObtainer& distancer)
                    : m_main_val(main_val)
                    , m_get_distance(distancer)
            {
            }

            bool operator()(const ValType& val1, const ValType& val2)
            {
                return m_get_distance(m_main_val, val1) < m_get_distance(m_main_val, val2);
            }
        private:
            const ValType& m_main_val;
            const DistanceObtainer& m_get_distance;
        };

        class VPTree
        {
        public:
            typedef VPNode VPNodeType;
            typedef boost::shared_ptr<VPNodeType> VPNodePtr;

            typedef std::vector<unsigned> ObjectsList;	// Objects, at leaf node

            typedef std::pair<float, unsigned> SearchResult;
            typedef std::multimap<float, unsigned> SearchResultMap;
            typedef std::vector<float> DistanceObtainer;

            typedef ValueSorter<unsigned, DistanceObtainer> ValueSorterType;
        public:

            VPTree(const size_t non_leaf_branching_factor = 2, const size_t leaf_branching_factor = 10)
                    : m_root(new VPNodeType())
                    , m_non_leaf_branching_factor(non_leaf_branching_factor)
                    , m_leaf_branching_factor(leaf_branching_factor)
                    , m_out(NULL)
            {
            }


            VPTree(const VPTree& orig)
                    : m_non_leaf_branching_factor(3)
                    , m_leaf_branching_factor(3)
                    , m_out(NULL)
            {
                if(&orig == this)
                    return;

                m_root = VPNodePtr(new VPNodeType());
            }

            virtual ~VPTree()
            {
            }

            const VPNodePtr& get_root() const
            {
                return m_root;
            }

            void set_info_output(std::ostream* stream)
            {
                m_out = stream;
            }

            size_t get_object_count() const
            {
                return get_object_count(m_root);
            }

            void ClearStatistic()
            {
                m_stat.clear();
            }

            const TreeStatistic& statistic() const
            {
                return m_stat;
            }

            size_t get_object_count(const VPNodePtr& node) const
            {
                size_t c_count = 0;
                get_object_count(c_count, node);
                return c_count;
            }

            void get_object_count(size_t& obj_count, const VPNodePtr& node) const
            {
                if(node->get_leaf_node())
                    obj_count += node->get_objects_count();
                else
                {
                    for (size_t c_pos = 0; c_pos < node->get_branches_count(); ++c_pos)
                        get_object_count(obj_count, node->m_child_list[c_pos]);
                }
            }

        public:
            VPNodePtr m_root;
            DistanceObtainer m_get_distance;

            size_t m_non_leaf_branching_factor;
            size_t m_leaf_branching_factor;

            std::ostream* m_out;

            // Statistic
            TreeStatistic m_stat;
        };

        class DistanceCheckedSet : public std::unordered_set<unsigned> {
        public:
            bool operator[](unsigned id) { return find(id) != end(); }
        };

        unsigned edgeSizeForCreation;
        unsigned batchSizeForCreation;

        unsigned truncationThreshold;

        unsigned edgeSizeForSearch;
        unsigned size;
        float explorationCoefficient = 1.1;

        int numOfOutgoingEdges ;
        int numOfIncomingEdges ;

        typedef VPNode VPNodeType;
        typedef boost::shared_ptr<VPNodeType> VPNodePtr;

        VPTree vp_tree;
    };

    class SPTAG {
    public:
        struct KDTNode
        {
            int left;
            int right;
            int split_dim;
            float split_value;
        };

        // node type for storing BKT
        struct BKTNode
        {
            int centerid;
            int childStart;
            int childEnd;

            BKTNode(int cid = -1) : centerid(cid), childStart(-1), childEnd(-1) {}
        };

        template <typename T>
        struct KmeansArgs {
            int _K;
            int _DK;
            int _D;
            int _T;
            T* centers;
            T* newTCenters;
            int* counts;
            float* newCenters;
            int* newCounts;
            int* label;
            int* clusterIdx;
            float* clusterDist;
            float* weightedCounts;
            float* newWeightedCounts;

            KmeansArgs(int k, int dim, int datasize, int threadnum) : _K(k), _DK(k), _D(dim), _T(threadnum) {
                centers = (T*)_mm_malloc(sizeof(T) * k * dim, ALIGN);
                newTCenters = (T*)_mm_malloc(sizeof(T) * k * dim, ALIGN);
                counts = new int[k];
                newCenters = new float[threadnum * k * dim];
                newCounts = new int[threadnum * k];
                label = new int[datasize];
                clusterIdx = new int[threadnum * k];
                clusterDist = new float[threadnum * k];
                weightedCounts = new float[k];
                newWeightedCounts = new float[threadnum * k];
            }

            ~KmeansArgs() {
                _mm_free(centers);
                _mm_free(newTCenters);
                delete[] counts;
                delete[] newCenters;
                delete[] newCounts;
                delete[] label;
                delete[] clusterIdx;
                delete[] clusterDist;
                delete[] weightedCounts;
                delete[] newWeightedCounts;
            }

            inline void ClearCounts() {
                memset(newCounts, 0, sizeof(int) * _T * _K);
                memset(newWeightedCounts, 0, sizeof(float) * _T * _K);
            }

            inline void ClearCenters() {
                memset(newCenters, 0, sizeof(float) * _T * _K * _D);
            }

            inline void ClearDists(float dist) {
                for (int i = 0; i < _T * _K; i++) {
                    clusterIdx[i] = -1;
                    clusterDist[i] = dist;
                }
            }

            void Shuffle(std::vector<int>& indices, int first, int last) {
                int* pos = new int[_K];
                pos[0] = first;
                for (int k = 1; k < _K; k++) pos[k] = pos[k - 1] + newCounts[k - 1];

                for (int k = 0; k < _K; k++) {
                    if (newCounts[k] == 0) continue;
                    int i = pos[k];
                    while (newCounts[k] > 0) {
                        int swapid = pos[label[i]] + newCounts[label[i]] - 1;
                        newCounts[label[i]]--;
                        std::swap(indices[i], indices[swapid]);
                        std::swap(label[i], label[swapid]);
                    }
                    while (indices[i] != clusterIdx[k]) i++;
                    std::swap(indices[i], indices[pos[k] + counts[k] - 1]);
                }
                delete[] pos;
            }
        };

        struct HeapCell
        {
            int node;
            float distance;

            HeapCell(int _node = -1, float _distance = (std::numeric_limits<float>::max)()) : node(_node), distance(_distance) {}

            inline bool operator < (const HeapCell& rhs) const
            {
                return distance < rhs.distance;
            }

            inline bool operator > (const HeapCell& rhs) const
            {
                return distance > rhs.distance;
            }
        };

        class Heap {
        public:
            Heap() : heap(nullptr), length(0), count(0) {}

            Heap(int size) { Resize(size); }

            void Resize(int size)
            {
                length = size;
                heap.reset(new HeapCell[length + 1]);  // heap uses 1-based indexing
                count = 0;
                lastlevel = int(pow(2.0, floor(log2((float)size))));
            }
            ~Heap() {}
            inline int size() { return count; }
            inline bool empty() { return count == 0; }
            inline void clear() { count = 0; }
            inline HeapCell& Top() { if (count == 0) return heap[0]; else return heap[1]; }

            // Insert a new element in the heap.
            void insert(const HeapCell& value)
            {
                /* If heap is full, then return without adding this element. */
                int loc;
                if (count == length) {
                    int maxi = lastlevel;
                    for (int i = lastlevel + 1; i <= length; i++)
                        if (heap[maxi] < heap[i]) maxi = i;
                    if (value > heap[maxi]) return;
                    loc = maxi;
                }
                else {
                    loc = ++(count);   /* Remember 1-based indexing. */
                }
                /* Keep moving parents down until a place is found for this node. */
                int par = (loc >> 1);                 /* Location of parent. */
                while (par > 0 && value < heap[par]) {
                    heap[loc] = heap[par];     /* Move parent down to loc. */
                    loc = par;
                    par >>= 1;
                }
                /* Insert the element at the determined location. */
                heap[loc] = value;
            }
            // Returns the node of minimum value from the heap (top of the heap).
            bool pop(HeapCell& value)
            {
                if (count == 0) return false;
                /* Switch first node with last. */
                value = heap[1];
                std::swap(heap[1], heap[count]);
                count--;
                heapify();      /* Move new node 1 to right position. */
                return true;  /* Return old last node. */
            }
            HeapCell& pop()
            {
                if (count == 0) return heap[0];
                /* Switch first node with last. */
                std::swap(heap[1], heap[count]);
                count--;
                heapify();      /* Move new node 1 to right position. */
                return heap[count + 1];  /* Return old last node. */
            }
            void heapify()
            {
                int parent = 1, next = 2;
                while (next < count) {
                    if (heap[next] > heap[next + 1]) next++;
                    if (heap[next] < heap[parent]) {
                        std::swap(heap[parent], heap[next]);
                        parent = next;
                        next <<= 1;
                    }
                    else break;
                }
                if (next == count && heap[next] < heap[parent]) std::swap(heap[parent], heap[next]);
            }
        private:
            // Storage array for the heap.
            // Type T must be comparable.
            std::unique_ptr<HeapCell[]> heap;
            int length;
            int count; // Number of element in the heap
            int lastlevel;
            // Reorganizes the heap (a parent is smaller than its children) starting with a node.
        };

        class OptHashPosVector
        {
        protected:
            // Max loop number in one hash block.
            static const int m_maxLoop = 8;

            // Could we use the second hash block.
            bool m_secondHash;

            // Max pool size.
            int m_poolSize;

            // Record 2 hash tables.
            // [0~m_poolSize + 1) is the first block.
            // [m_poolSize + 1, 2*(m_poolSize + 1)) is the second block;
            std::unique_ptr<int[]> m_hashTable;


            inline int hash_func2(int idx, int loop)
            {
                return (idx + loop) & m_poolSize;
            }


            inline int hash_func(int idx)
            {
                return ((int)(idx * 99991) + _rotl(idx, 2) + 101) & m_poolSize;
            }

        public:
            OptHashPosVector() {}

            ~OptHashPosVector() {}


            void Init(int size, int exp)
            {
                int ex = 0;
                while (size != 0) {
                    ex++;
                    size >>= 1;
                }
                m_secondHash = true;
                m_poolSize = (1 << (ex + exp)) - 1;
                m_hashTable.reset(new int[(m_poolSize + 1) * 2]);
                clear();
            }

            void clear()
            {
                if (!m_secondHash)
                {
                    // Clear first block.
                    memset(m_hashTable.get(), 0, sizeof(int) * (m_poolSize + 1));
                }
                else
                {
                    // Clear all blocks.
                    m_secondHash = false;
                    memset(m_hashTable.get(), 0, 2 * sizeof(int) * (m_poolSize + 1));
                }
            }


            inline bool CheckAndSet(int idx)
            {
                // Inner Index is begin from 1
                return _CheckAndSet(m_hashTable.get(), idx + 1) == 0;
            }


            inline int _CheckAndSet(int* hashTable, int idx)
            {
                int index = hash_func((int)idx);
                for (int loop = 0; loop < m_maxLoop; ++loop)
                {
                    if (!hashTable[index])
                    {
                        // index first match and record it.
                        hashTable[index] = idx;
                        return 1;
                    }
                    if (hashTable[index] == idx)
                    {
                        // Hit this item in hash table.
                        return 0;
                    }
                    // Get next hash position.
                    index = hash_func2(index, loop);
                }

                if (hashTable == m_hashTable.get())
                {
                    // Use second hash block.
                    m_secondHash = true;
                    return _CheckAndSet(m_hashTable.get() + m_poolSize + 1, idx);
                }

                // Do not include this item.
                std::cout << "Hash table is full!" << std::endl;
                return -1;
            }
        };

        class SPTAGFurtherFirst {
        public:
            SPTAGFurtherFirst(int node, float distance) : node_(node), distance_(distance) {}
            inline float GetDistance() const { return distance_; }
            inline int GetNode() const { return node_; }
            bool operator< (const SPTAGFurtherFirst& n) const {
                return (distance_ < n.GetDistance());
            }
        private:
            int node_;
            float distance_;
        };

        class SPTAGCloserFirst {
        public:
            SPTAGCloserFirst(int node, float distance) : node_(node), distance_(distance) {}
            inline float GetDistance() const { return distance_; }
            inline int GetNode() const { return node_; }
            bool operator< (const SPTAGCloserFirst& n) const {
                return (distance_ > n.GetDistance());
            }
        private:
            int node_;
            float distance_;
        };

        struct BasicResult
        {
            int VID;
            float Dist;

            BasicResult() : VID(-1), Dist((std::numeric_limits<float>::max)()) {}

            BasicResult(int p_vid, float p_dist) : VID(p_vid), Dist(p_dist) {}
        };

        class QueryResult
        {
        public:
            typedef BasicResult* iterator;
            typedef const BasicResult* const_iterator;

            QueryResult()
                    : m_resultNum(0)
            {
            }


            QueryResult(int p_resultNum)
            {
                Init(p_resultNum);
            }


            QueryResult(int p_resultNum, BasicResult* p_results)
                    : m_resultNum(p_resultNum)

            {
                m_results.Set(p_results, p_resultNum, false);
            }


//            QueryResult(const QueryResult& p_other)
//            {
//                Init(p_other.m_resultNum);
//                if (m_resultNum > 0)
//                {
//                    std::copy(p_other.m_results.Data(), p_other.m_results.Data() + m_resultNum, m_results.Data());
//                }
//            }


            QueryResult& operator=(const QueryResult& p_other)
            {
                Init(p_other.m_resultNum);
                if (m_resultNum > 0)
                {
                    std::copy(p_other.m_results.Data(), p_other.m_results.Data() + m_resultNum, m_results.Data());
                }

                return *this;
            }


            ~QueryResult()
            {
            }


            inline void Init(int p_resultNum)
            {
                m_resultNum = p_resultNum;

                m_results = Array<BasicResult>::Alloc(p_resultNum);
            }


            inline int GetResultNum() const
            {
                return m_resultNum;
            }


            inline BasicResult* GetResult(int i) const
            {
                return i < m_resultNum ? m_results.Data() + i : nullptr;
            }


            inline void SetResult(int p_index, int p_VID, float p_dist)
            {
                if (p_index < m_resultNum)
                {
                    m_results[p_index].VID = p_VID;
                    m_results[p_index].Dist = p_dist;
                }
            }


            inline BasicResult* GetResults() const
            {
                return m_results.Data();
            }


            inline void Reset()
            {
                for (int i = 0; i < m_resultNum; i++)
                {
                    m_results[i].VID = -1;
                    const float MaxDist = (std::numeric_limits<float>::max)();
                    m_results[i].Dist = MaxDist;
                }
            }


            iterator begin()
            {
                return m_results.Data();
            }


            iterator end()
            {
                return m_results.Data() + m_resultNum;
            }


            const_iterator begin() const
            {
                return m_results.Data();
            }


            const_iterator end() const
            {
                return m_results.Data() + m_resultNum;
            }


        protected:
            int m_resultNum;

            Array<BasicResult> m_results;
        };

        // Space to save temporary answer, similar with TopKCache
        class QueryResultSet : public QueryResult
        {
        public:
            QueryResultSet(int K) : QueryResult(K)
            {
            }

//            QueryResultSet(const QueryResultSet& other) : QueryResult(other)
//            {
//            }

            inline float worstDist() const
            {
                return m_results[0].Dist;
            }

            bool AddPoint(const int index, float dist)
            {
                if (dist < m_results[0].Dist || (dist == m_results[0].Dist && index < m_results[0].VID))
                {
                    m_results[0].VID = index;
                    m_results[0].Dist = dist;
                    Heapify(m_resultNum);
                    return true;
                }
                return false;
            }

            inline void SortResult()
            {
                for (int i = m_resultNum - 1; i >= 0; i--)
                {
                    std::swap(m_results[0], m_results[i]);
                    Heapify(i);
                }
            }

            void Reverse()
            {
                std::reverse(m_results.Data(), m_results.Data() + m_resultNum);
            }

        private:
            void Heapify(int count)
            {
                int parent = 0, next = 1, maxidx = count - 1;
                while (next < maxidx)
                {
                    if (m_results[next].Dist < m_results[next + 1].Dist) next++;
                    if (m_results[parent].Dist < m_results[next].Dist)
                    {
                        std::swap(m_results[next], m_results[parent]);
                        parent = next;
                        next = (parent << 1) + 1;
                    }
                    else break;
                }
                if (next == maxidx && m_results[parent].Dist < m_results[next].Dist) std::swap(m_results[parent], m_results[next]);
            }
        };

        std::unordered_map<int, int> m_pSampleCenterMap;

        std::vector<int> m_pTreeStart;
        std::vector<KDTNode> m_pKDTreeRoots;
        std::vector<BKTNode> m_pBKTreeRoots;

        unsigned numOfThreads;

        unsigned m_iTreeNumber;
        unsigned m_numTopDimensionKDTSplit = 5;
        unsigned m_iTPTNumber = 32;
        unsigned m_iTPTLeafSize = 2000;
        unsigned m_numTopDimensionTPTSplit = 5;
        // K
        unsigned m_iNeighborhoodSize = 32;
        // K2 = K * m_iNeighborhoodScale
        unsigned m_iNeighborhoodScale = 2;
        // L
        unsigned m_iCEF = 1000;
        unsigned m_iHashTableExp = 4;

        int m_iSamples = 1000;

        unsigned m_iCEFScale = 2;
        unsigned m_iBKTKmeansK;
        unsigned m_iBKTLeafSize = 8;
        unsigned m_iNumberOfInitialDynamicPivots = 50;
        unsigned m_iNumberOfOtherDynamicPivots = 4;
        unsigned m_iMaxCheckForRefineGraph = 10000;
        unsigned m_iMaxCheck = 8192L;
    };

    class FANNG {
    public:
        unsigned M;

        class FANNGCloserFirst {
        public:
            FANNGCloserFirst(unsigned node, float distance) : node_(node), distance_(distance) {}
            inline float GetDistance() const { return distance_; }
            inline unsigned GetNode() const { return node_; }
            bool operator< (const FANNGCloserFirst& n) const {
                return (distance_ > n.GetDistance());
            }
        private:
            unsigned node_;
            float distance_;
        };
    };

    class HCNNG {
    public:
        struct Edge{
            int v1, v2;
            float weight;
            Edge(){
                v1 = -1;
                v2 = -1;
                weight = -1;
            }
            Edge(int _v1, int _v2, float _weight){
                v1 = _v1;
                v2 = _v2;
                weight = _weight;
            }
            bool operator<(const Edge& e) const {
                return weight < e.weight;
            }
            ~Edge() { }
        };

        struct DisjointSet{
            int * parent;
            int * rank;
            DisjointSet(int N){
                parent = new int[N];
                rank = new int[N];
                for(int i=0; i<N; i++){
                    parent[i] = i;
                    rank[i] = 0;
                }
            }
            void _union(int x, int y){
                int xroot = parent[x];
                int yroot = parent[y];
                int xrank = rank[x];
                int yrank = rank[y];
                if(xroot == yroot)
                    return;
                else if(xrank < yrank)
                    parent[xroot] = yroot;
                else{
                    parent[yroot] = xroot;
                    if(xrank == yrank)
                        rank[xroot] = rank[xroot] + 1;
                }
            }
            int find(int x){
                if(parent[x] != x)
                    parent[x] = find(parent[x]);
                return parent[x];
            }

            ~DisjointSet() {
                delete[] parent;
                delete[] rank;
            }
        };

        struct Tnode {
            unsigned div_dim;
            std::vector <unsigned> left;
            std::vector <unsigned> right;
        };
        std::vector <Tnode> Tn;

        int xxx = 0;

        unsigned minsize_cl = 0;
        unsigned num_cl = 0;
    };

    class Index : public NNDescent, public NSG, public SSG, public DPG, public VAMANA, public EFANNA, public IEH,
            public NSW, public HNSW, public NGT, public SPTAG, public FANNG, public HCNNG {
    public:
        explicit Index() {
            dist_ = new Distance();
        }

        ~Index() {
            delete dist_;
        }

        struct SimpleNeighbor{
            unsigned id;
            float distance;

            SimpleNeighbor() = default;
            SimpleNeighbor(unsigned id, float distance) : id{id}, distance{distance}{}

            inline bool operator<(const SimpleNeighbor &other) const {
                return distance < other.distance;
            }
        };

        float *getBaseData() const {
            return base_data_;
        }

        void setBaseData(float *baseData) {
            base_data_ = baseData;
        }

        float *getQueryData() const {
            return query_data_;
        }

        void setQueryData(float *queryData) {
            query_data_ = queryData;
        }

        unsigned int *getGroundData() const {
            return ground_data_;
        }

        void setGroundData(unsigned int *groundData) {
            ground_data_ = groundData;
        }

        unsigned int getBaseLen() const {
            return base_len_;
        }

        void setBaseLen(unsigned int baseLen) {
            base_len_ = baseLen;
        }

        unsigned int getQueryLen() const {
            return query_len_;
        }

        void setQueryLen(unsigned int queryLen) {
            query_len_ = queryLen;
        }

        unsigned int getGroundLen() const {
            return ground_len_;
        }

        void setGroundLen(unsigned int groundLen) {
            ground_len_ = groundLen;
        }

        unsigned int getBaseDim() const {
            return base_dim_;
        }

        void setBaseDim(unsigned int baseDim) {
            base_dim_ = baseDim;
        }

        unsigned int getQueryDim() const {
            return query_dim_;
        }

        void setQueryDim(unsigned int queryDim) {
            query_dim_ = queryDim;
        }

        unsigned int getGroundDim() const {
            return ground_dim_;
        }

        void setGroundDim(unsigned int groundDim) {
            ground_dim_ = groundDim;
        }

        Parameters &getParam() {
            return param_;
        }

        void setParam(const Parameters &param) {
            param_ = param;
        }

        unsigned int getInitEdgesNum() const {
            return init_edges_num;
        }

        void setInitEdgesNum(unsigned int initEdgesNum) {
            init_edges_num = initEdgesNum;
        }

        unsigned int getCandidatesEdgesNum() const {
            return candidates_edges_num;
        }

        void setCandidatesEdgesNum(unsigned int candidatesEdgesNum) {
            candidates_edges_num = candidatesEdgesNum;
        }

        unsigned int getResultEdgesNum() const {
            return result_edges_num;
        }

        void setResultEdgesNum(unsigned int resultEdgesNum) {
            result_edges_num = resultEdgesNum;
        }

        Distance *getDist() const {
            return dist_;
        }

        void setDist(Distance *dist) {
            dist_ = dist;
        }

        // sorted
        typedef std::vector<std::vector<SimpleNeighbor> > FinalGraph;
        typedef std::vector<std::vector<unsigned> > LoadGraph;

        FinalGraph &getFinalGraph() {
            return final_graph_;
        }

        LoadGraph &getLoadGraph() {
            return load_graph_;
        }

        LoadGraph &getExactGraph() {
            return exact_graph_;
        }

        TYPE getCandidateType() const {
            return candidate_type;
        }

        void setCandidateType(TYPE candidateType) {
            candidate_type = candidateType;
        }

        TYPE getPruneType() const {
            return prune_type;
        }

        void setPruneType(TYPE pruneType) {
            prune_type = pruneType;
        }

        TYPE getEntryType() const {
            return entry_type;
        }

        void setEntryType(TYPE entryType) {
            entry_type = entryType;
        }

        void setConnType(TYPE connType) {
            conn_type = connType;
        }

        TYPE getConnType() const {
            return conn_type;
        }

        unsigned int getDistCount() const {
            return dist_count;
        }

        void resetDistCount() {
            dist_count = 0;
        }

        void addDistCount() {
            dist_count += 1;
        }

        unsigned int getHopCount() const {
            return hop_count;
        }

        void resetHopCount() {
            hop_count = 0;
        }

        void addHopCount() {
            hop_count += 1;
        }
        
        void setNumThreads(const unsigned numthreads) {
            omp_set_num_threads(numthreads);
        }

        int i = 0;
        bool debug = false;

    private:
        float *base_data_, *query_data_;
        unsigned *ground_data_;
        unsigned base_len_, query_len_, ground_len_;
        unsigned base_dim_, query_dim_, ground_dim_;

        Parameters param_;
        unsigned init_edges_num; // S
        unsigned candidates_edges_num; // L
        unsigned result_edges_num; // K

        Distance *dist_;

        FinalGraph final_graph_;
        LoadGraph load_graph_;
        LoadGraph exact_graph_;


        TYPE entry_type;
        TYPE candidate_type;
        TYPE prune_type;
        TYPE conn_type;

        unsigned dist_count = 0;
        unsigned hop_count = 0;
    };
}

#endif //WEAVESS_INDEX_H

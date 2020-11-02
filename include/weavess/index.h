//
// Created by MurphySL on 2020/10/23.
//

#ifndef WEAVESS_INDEX_H
#define WEAVESS_INDEX_H

#define PARALLEL
#define FLT_EPSILON 1.19209290E-07F

// IEH
#define MAX_ROWSIZE 1024
#define HASH_RADIUS 1
#define DEPTH 16 //smaller than code length
#define INIT_NUM 5500

// SPTAG
#define ALIGN 32
#define aligned_malloc(a, b) _mm_malloc(a, b)

#include <omp.h>
#include <mutex>
#include <queue>
#include <stack>
#include <thread>
#include <vector>
#include <chrono>
#include <cstring>
#include <fstream>
#include <cassert>
#include <iostream>
#include <windows.h>
#include <algorithm>
#include <unordered_set>
#include <boost/dynamic_bitset.hpp>
#include <boost/heap/d_ary_heap.hpp>
#include "util.h"
#include "policy.h"
#include "distance.h"
#include "parameters.h"

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

            // 插入大顶堆
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
        float alpha;
    };

    class EFANNA {
    public:
        // 节点不保存数据，只维护一个 LeafLists 中对应的数据编号
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

        std::vector<Node *> tree_roots_;                    // 存储树根
        std::vector<std::pair<Node *, size_t> > mlNodeList;  //  ml 层 节点 和对应树根编号
        std::vector<std::vector<unsigned>> LeafLists;       // 存储每个随机截断树的对应节点
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
        unsigned n_threads_ = 1;
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
            inline int GetLevel() const { return level_; }
            inline size_t GetMaxM() const { return max_m_; }
            inline size_t GetMaxM0() const { return max_m0_; }

            inline std::vector<HnswNode*>& GetFriends(int level) { return friends_at_layer_[level]; }
            inline void SetFriends(int level, std::vector<HnswNode*>& new_friends) {
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

            std::vector<std::vector<HnswNode*>> friends_at_layer_;
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
        typedef typename boost::heap::d_ary_heap<IdDistancePair, boost::heap::arity<4>, boost::heap::compare<IdDistancePairMinHeapComparer>> IdDistancePairMinHeap;

        HnswNode* enterpoint_ = nullptr;
        std::vector<HnswNode*> nodes_;
    };

    class NGT {
    public :
        unsigned truncationThreshold;
        unsigned edgeSizeForCreation;
        unsigned edgeSizeForSearch;
        unsigned size;
        float explorationCoefficient = 1.1;
    };

    class SPTAG {
    public:
        struct KDTNode
        {
            unsigned left;
            unsigned right;
            unsigned split_dim;
            float split_value;
        };

        // node type for storing BKT
        struct BKTNode
        {
            unsigned centerid;
            unsigned childStart;
            unsigned childEnd;

            BKTNode(unsigned cid = -1) : centerid(cid), childStart(-1), childEnd(-1) {}
        };

        template <typename T>
        struct KmeansArgs {
            int _K;
            int _DK;
            unsigned _D;
            int _T;
            T* centers;
            T* newTCenters;
            unsigned* counts;
            float* newCenters;
            unsigned* newCounts;
            int* label;
            unsigned* clusterIdx;
            float* clusterDist;
            float* weightedCounts;
            float* newWeightedCounts;

            KmeansArgs(int k, unsigned dim, unsigned datasize, int threadnum) : _K(k), _DK(k), _D(dim), _T(threadnum) {
                centers = (T*)aligned_malloc(sizeof(T) * k * dim, ALIGN);
                newTCenters = (T*)aligned_malloc(sizeof(T) * k * dim, ALIGN);
                counts = new unsigned[k];
                newCenters = new float[threadnum * k * dim];
                newCounts = new unsigned[threadnum * k];
                label = new int[datasize];
                clusterIdx = new unsigned[threadnum * k];
                clusterDist = new float[threadnum * k];
                weightedCounts = new float[k];
                newWeightedCounts = new float[threadnum * k];
            }

            ~KmeansArgs() {
                delete[] centers;
                delete[] newTCenters;
//                aligned_free(centers);
//                aligned_free(newTCenters);
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
                memset(newCounts, 0, sizeof(unsigned) * _T * _K);
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

            void Shuffle(std::vector<unsigned>& indices, unsigned first, unsigned last) {
                unsigned* pos = new unsigned[_K];
                pos[0] = first;
                for (int k = 1; k < _K; k++) pos[k] = pos[k - 1] + newCounts[k - 1];

                for (int k = 0; k < _K; k++) {
                    if (newCounts[k] == 0) continue;
                    unsigned i = pos[k];
                    while (newCounts[k] > 0) {
                        unsigned swapid = pos[label[i]] + newCounts[label[i]] - 1;
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
            unsigned node;
            float distance;

            HeapCell(unsigned _node = -1, float _distance = (std::numeric_limits<float>::max)()) : node(_node), distance(_distance) {}

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
        private:
            // Storage array for the heap.
            // Type T must be comparable.
            std::unique_ptr<HeapCell[]> heap;
            int length;
            int count; // Number of element in the heap
            int lastlevel;
            // Reorganizes the heap (a parent is smaller than its children) starting with a node.

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
            std::unique_ptr<unsigned[]> m_hashTable;


            inline unsigned hash_func2(unsigned idx, int loop)
            {
                return (idx + loop) & m_poolSize;
            }


            inline unsigned hash_func(unsigned idx)
            {
                return ((unsigned)(idx * 99991) + _rotl(idx, 2) + 101) & m_poolSize;
            }

        public:
            OptHashPosVector() {}

            ~OptHashPosVector() {}


            void Init(unsigned size, int exp)
            {
                int ex = 0;
                while (size != 0) {
                    ex++;
                    size >>= 1;
                }
                m_secondHash = true;
                m_poolSize = (1 << (ex + exp)) - 1;
                m_hashTable.reset(new unsigned[(m_poolSize + 1) * 2]);
                clear();
            }

            void clear()
            {
                if (!m_secondHash)
                {
                    // Clear first block.
                    memset(m_hashTable.get(), 0, sizeof(unsigned) * (m_poolSize + 1));
                }
                else
                {
                    // Clear all blocks.
                    m_secondHash = false;
                    memset(m_hashTable.get(), 0, 2 * sizeof(unsigned) * (m_poolSize + 1));
                }
            }


            inline bool CheckAndSet(unsigned idx)
            {
                // Inner Index is begin from 1
                return _CheckAndSet(m_hashTable.get(), idx + 1) == 0;
            }


            inline int _CheckAndSet(unsigned* hashTable, unsigned idx)
            {
                unsigned index = hash_func((unsigned)idx);
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
            SPTAGFurtherFirst(unsigned node, float distance) : node_(node), distance_(distance) {}
            inline float GetDistance() const { return distance_; }
            inline unsigned GetNode() const { return node_; }
            bool operator< (const SPTAGFurtherFirst& n) const {
                return (distance_ < n.GetDistance());
            }
        private:
            unsigned node_;
            float distance_;
        };

        class SPTAGCloserFirst {
        public:
            SPTAGCloserFirst(unsigned node, float distance) : node_(node), distance_(distance) {}
            inline float GetDistance() const { return distance_; }
            inline unsigned GetNode() const { return node_; }
            bool operator< (const SPTAGCloserFirst& n) const {
                return (distance_ > n.GetDistance());
            }
        private:
            unsigned node_;
            float distance_;
        };

        std::unordered_map<unsigned, unsigned> m_pSampleCenterMap;

        std::vector<unsigned> m_pTreeStart;
        std::vector<KDTNode> m_pKDTreeRoots;
        std::vector<BKTNode> m_pBKTreeRoots;

        unsigned numOfThreads;
        unsigned m_iTreeNumber;

        // 抽样选取数量
        unsigned m_iSamples = 1000;
        unsigned m_iTPTNumber = 32;
        unsigned m_iNeighborhoodSize = 32;
        unsigned m_iNeighborhoodScale = 2;
        unsigned m_iCEF = 1000;
        unsigned m_iCEFScale = 2;
        unsigned m_iBKTKmeansK = 32;
        unsigned m_iNumberOfInitialDynamicPivots = 50;
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

    class Index : public NNDescent, public NSG, public SSG, public DPG, public VAMANA, public EFANNA, public IEH,
            public NSW, public HNSW, public NGT, public SPTAG, public FANNG {
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

        Distance *getDist() const {
            return dist_;
        }

        void setDist(Distance *dist) {
            dist_ = dist;
        }

        // sorted
        typedef std::vector<std::vector<SimpleNeighbor>> FinalGraph;

        FinalGraph &getFinalGraph() {
            return final_graph_;
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

        void addDistCount() {
            dist_count += 1;
        }

        int i = 0;

    private:
        float *base_data_, *query_data_;
        unsigned *ground_data_;
        unsigned base_len_, query_len_, ground_len_;
        unsigned base_dim_, query_dim_, ground_dim_;

        Parameters param_;

        Distance *dist_;

        // 迭代式
        FinalGraph final_graph_;

        TYPE entry_type;
        TYPE candidate_type;
        TYPE prune_type;
        TYPE conn_type;

        unsigned dist_count = 0;
    };
}

#endif //WEAVESS_INDEX_H

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

// HCNNG
#define not_in_set(_elto, _set) (_set.find(_elto)==_set.end())

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
#include <windows.h>
#include <algorithm>
#include <unordered_set>
#include <boost/dynamic_bitset.hpp>
#include <boost/heap/d_ary_heap.hpp>
#include "util.h"
#include "policy.h"
#include "distance.h"
#include "parameters.h"
#include "CommonDataStructure.h"

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
        float alpha = 1;
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

        template <class TYPE>
        class Repository : public std::vector<TYPE *>
        {
        public:
            static TYPE *allocate() { return new TYPE; }

            size_t push(TYPE *n)
            {
                if (std::vector<TYPE *>::size() == 0)
                {
                    std::vector<TYPE *>::push_back(0);
                }
                std::vector<TYPE *>::push_back(n);
                return std::vector<TYPE *>::size() - 1;
            }

            size_t insert(TYPE *n)
            {
#ifdef ADVANCED_USE_REMOVED_LIST
                if (!removedList.empty())
      {
        size_t idx = removedList.top();
        removedList.pop();
        put(idx, n);
        return idx;
      }
#endif
                return push(n);
            }

            bool isEmpty(size_t idx)
            {
                if (idx < std::vector<TYPE *>::size())
                {
                    return (*this)[idx] == 0;
                }
                else
                {
                    return true;
                }
            }

            void put(size_t idx, TYPE *n)
            {
                if (std::vector<TYPE *>::size() <= idx)
                {
                    std::vector<TYPE *>::resize(idx + 1, 0);
                }
                if ((*this)[idx] != 0)
                {
                    std::cout << "put: Not empty" << std::endl;
                }
                (*this)[idx] = n;
            }

            void erase(size_t idx)
            {
                if (isEmpty(idx))
                {
                    std::cout << "erase: Not in-memory or invalid id" << std::endl;
                }
                delete (*this)[idx];
                (*this)[idx] = 0;
            }

            void remove(size_t idx)
            {
                erase(idx);
#ifdef ADVANCED_USE_REMOVED_LIST
                removedList.push(idx);
#endif
            }

            TYPE **getPtr() { return &(*this)[0]; }

            inline TYPE *get(size_t idx)
            {
                if (isEmpty(idx))
                {
                    std::stringstream msg;
                    msg << "get: Not in-memory or invalid offset of node. idx=" << idx << " size=" << this->size();
                    std::cout << msg.str() << std::endl;
                }
                return (*this)[idx];
            }

            inline TYPE *getWithoutCheck(size_t idx) { return (*this)[idx]; }

            void deleteAll()
            {
                for (size_t i = 0; i < this->size(); i++)
                {
                    if ((*this)[i] != 0)
                    {
                        delete (*this)[i];
                        (*this)[i] = 0;
                    }
                }
                this->clear();
#ifdef ADVANCED_USE_REMOVED_LIST
                while (!removedList.empty())
      {
        removedList.pop();
      };
#endif
            }

            void set(size_t idx, TYPE *n)
            {
                (*this)[idx] = n;
            }

#ifdef ADVANCED_USE_REMOVED_LIST
            size_t count()
    {
      return std::vector<TYPE *>::size() == 0 ? 0 : std::vector<TYPE *>::size() - removedList.size() - 1;
    }

  protected:
    std::priority_queue<size_t, std::vector<size_t>, std::greater<size_t>> removedList;
#endif
        };

        typedef unsigned int ObjectID;
        typedef unsigned int NodeID;
        class ID {
        public:
            enum Type {
                Leaf		= 1,
                Internal	= 0
            };
            ID():id(0) {}
            ID &operator=(const ID &n) {
                id = n.id;
                return *this;
            }
            ID &operator=(int i) {
                setID(i);
                return *this;
            }
            bool operator==(ID &n) { return id == n.id; }
            bool operator<(ID &n) { return id < n.id; }
            Type getType() { return (Type)((0x80000000 & id) >> 31); }
            NodeID getID() { return 0x7fffffff & id; }
            NodeID get() { return id; }
            void setID(NodeID i) { id = (0x80000000 & id) | i; }
            void setType(Type t) { id = (t << 31) | getID(); }
            void setRaw(NodeID i) { id = i; }
            void setNull() { id = 0; }
        protected:
            NodeID id;
        };

        class Node {
        public:

            class Object {
            public:
                Object():object(0) {}
                bool operator<(const Object &o) const { return distance < o.distance; }
                static const double	Pivot;
                ObjectID		id;
                float	*object;
                float		distance;
                float		leafDistance;
                int		clusterID;
            };

            typedef std::vector<Object>	Objects;

            Node() {
                parent.setNull();
                id.setNull();
            }

            virtual ~Node() {}

            Node &operator=(const Node &n) {
                id = n.id;
                parent = n.parent;
                return *this;
            }

            void setPivot(float	*f) {
                pivot = f;
            }
            float* getPivot() { return pivot; }
            void deletePivot() {
                delete pivot;
            }

            bool pivotIsEmpty() {
                return pivot == 0;
            }

            ID		id;
            ID		parent;

            float		*pivot;

        };

        class InternalNode : public Node {
        public:
            InternalNode(size_t csize) : childrenSize(csize) { initialize(); }
            InternalNode(NGT::ObjectSpace *os = 0) : childrenSize(5) { initialize(); }

            ~InternalNode() {
            }

            void initialize() {
                id = 0;
                id.setType(ID::Internal);
                pivot = 0;
                children = new ID[childrenSize];
                for (size_t i = 0; i < childrenSize; i++) {
                    getChildren()[i] = 0;
                }
                borders = new Distance[childrenSize - 1];
                for (size_t i = 0; i < childrenSize - 1; i++) {
                    getBorders()[i] = 0;
                }
            }

            void updateChild(DVPTree &dvptree, ID src, ID dst);

//#ifdef NGT_SHARED_MEMORY_ALLOCATOR
//            ID *getChildren(SharedMemoryAllocator &allocator) { return (ID*)allocator.getAddr(children); }
//    Distance *getBorders(SharedMemoryAllocator &allocator) { return (Distance*)allocator.getAddr(borders); }
//#else // NGT_SHARED_MEMORY_ALLOCATOR
//            ID *getChildren() { return children; }
//            Distance *getBorders() { return borders; }
//#endif // NGT_SHARED_MEMORY_ALLOCATOR
//
//#if defined(NGT_SHARED_MEMORY_ALLOCATOR)
//            void serialize(std::ofstream &os, SharedMemoryAllocator &allocator, ObjectSpace *objectspace = 0) {
//#else
//            void serialize(std::ofstream &os, ObjectSpace *objectspace = 0) {
//#endif
//                Node::serialize(os);
//                if (pivot == 0) {
//                    NGTThrowException("Node::write: pivot is null!");
//                }
//                assert(objectspace != 0);
//#if defined(NGT_SHARED_MEMORY_ALLOCATOR)
//                getPivot(*objectspace).serialize(os, allocator, objectspace);
//#else
//                getPivot().serialize(os, objectspace);
//#endif
//                NGT::Serializer::write(os, childrenSize);
//                for (size_t i = 0; i < childrenSize; i++) {
//#if defined(NGT_SHARED_MEMORY_ALLOCATOR)
//                    getChildren(allocator)[i].serialize(os);
//#else
//                    getChildren()[i].serialize(os);
//#endif
//                }
//                for (size_t i = 0; i < childrenSize - 1; i++) {
//#if defined(NGT_SHARED_MEMORY_ALLOCATOR)
//                    NGT::Serializer::write(os, getBorders(allocator)[i]);
//#else
//                    NGT::Serializer::write(os, getBorders()[i]);
//#endif
//                }
//            }
//            void deserialize(std::ifstream &is, ObjectSpace *objectspace = 0) {
//                Node::deserialize(is);
//                if (pivot == 0) {
//#ifdef NGT_SHARED_MEMORY_ALLOCATOR
//                    pivot = PersistentObject::allocate(*objectspace);
//#else
//                    pivot = PersistentObject::allocate(*objectspace);
//#endif
//                }
//                assert(objectspace != 0);
//#if defined(NGT_SHARED_MEMORY_ALLOCATOR)
//                std::cerr << "not implemented" << std::endl;
//      assert(0);
//#else
//                getPivot().deserialize(is, objectspace);
//#endif
//                NGT::Serializer::read(is, childrenSize);
//                assert(children != 0);
//                for (size_t i = 0; i < childrenSize; i++) {
//#if defined(NGT_SHARED_MEMORY_ALLOCATOR)
//                    assert(0);
//#else
//                    getChildren()[i].deserialize(is);
//#endif
//                }
//                assert(borders != 0);
//                for (size_t i = 0; i < childrenSize - 1; i++) {
//#if defined(NGT_SHARED_MEMORY_ALLOCATOR)
//                    assert(0);
//#else
//                    NGT::Serializer::read(is, getBorders()[i]);
//#endif
//                }
//            }
//
//#if defined(NGT_SHARED_MEMORY_ALLOCATOR)
//            void serializeAsText(std::ofstream &os, SharedMemoryAllocator &allocator, ObjectSpace *objectspace = 0) {
//#else
//            void serializeAsText(std::ofstream &os, ObjectSpace *objectspace = 0) {
//#endif
//                Node::serializeAsText(os);
//                if (pivot == 0) {
//                    NGTThrowException("Node::write: pivot is null!");
//                }
//                os << " ";
//                assert(objectspace != 0);
//#if defined(NGT_SHARED_MEMORY_ALLOCATOR)
//                getPivot(*objectspace).serializeAsText(os, objectspace);
//#else
//                getPivot().serializeAsText(os, objectspace);
//#endif
//                os << " ";
//                NGT::Serializer::writeAsText(os, childrenSize);
//                os << " ";
//                for (size_t i = 0; i < childrenSize; i++) {
//#if defined(NGT_SHARED_MEMORY_ALLOCATOR)
//                    getChildren(allocator)[i].serializeAsText(os);
//#else
//                    getChildren()[i].serializeAsText(os);
//#endif
//                    os << " ";
//                }
//                for (size_t i = 0; i < childrenSize - 1; i++) {
//#if defined(NGT_SHARED_MEMORY_ALLOCATOR)
//                    NGT::Serializer::writeAsText(os, getBorders(allocator)[i]);
//#else
//                    NGT::Serializer::writeAsText(os, getBorders()[i]);
//#endif
//                    os << " ";
//                }
//            }
//#if defined(NGT_SHARED_MEMORY_ALLOCATOR)
//            void deserializeAsText(std::ifstream &is, SharedMemoryAllocator &allocator, ObjectSpace *objectspace = 0) {
//#else
//            void deserializeAsText(std::ifstream &is, ObjectSpace *objectspace = 0) {
//#endif
//                Node::deserializeAsText(is);
//                if (pivot == 0) {
//#ifdef NGT_SHARED_MEMORY_ALLOCATOR
//                    pivot = PersistentObject::allocate(*objectspace);
//#else
//                    pivot = PersistentObject::allocate(*objectspace);
//#endif
//                }
//                assert(objectspace != 0);
//#if defined(NGT_SHARED_MEMORY_ALLOCATOR)
//                getPivot(*objectspace).deserializeAsText(is, objectspace);
//#else
//                getPivot().deserializeAsText(is, objectspace);
//#endif
//                size_t csize;
//                NGT::Serializer::readAsText(is, csize);
//                assert(children != 0);
//                assert(childrenSize == csize);
//                for (size_t i = 0; i < childrenSize; i++) {
//#if defined(NGT_SHARED_MEMORY_ALLOCATOR)
//                    getChildren(allocator)[i].deserializeAsText(is);
//#else
//                    getChildren()[i].deserializeAsText(is);
//#endif
//                }
//                assert(borders != 0);
//                for (size_t i = 0; i < childrenSize - 1; i++) {
//#if defined(NGT_SHARED_MEMORY_ALLOCATOR)
//                    NGT::Serializer::readAsText(is, getBorders(allocator)[i]);
//#else
//                    NGT::Serializer::readAsText(is, getBorders()[i]);
//#endif
//                }
//            }
//
//            void show() {
//                std::cout << "Show internal node " << childrenSize << ":";
//                for (size_t i = 0; i < childrenSize; i++) {
//#if defined(NGT_SHARED_MEMORY_ALLOCATOR)
//                    assert(0);
//#else
//                    std::cout << getChildren()[i].getID() << " ";
//#endif
//                }
//                std::cout << std::endl;
//            }
//
//#if defined(NGT_SHARED_MEMORY_ALLOCATOR)
//            bool verify(PersistentRepository<InternalNode> &internalNodes, PersistentRepository<LeafNode> &leafNodes,
//		SharedMemoryAllocator &allocator);
//#else
//            bool verify(Repository<InternalNode> &internalNodes, Repository<LeafNode> &leafNodes);
//#endif

            static const int InternalChildrenSizeMax	= 5;
            const size_t	childrenSize;
            ID			*children;
            Distance		*borders;
        };


        class LeafNode : public Node {
        public:
            LeafNode(NGT::ObjectSpace *os = 0) {
                id = 0;
                id.setType(ID::Leaf);
                pivot = 0;
#ifdef NGT_NODE_USE_VECTOR
                objectIDs.reserve(LeafObjectsSizeMax);
#else
                objectSize = 0;
                objectIDs = new NGT::ObjectDistance[LeafObjectsSizeMax];
#endif
            }

//            ~LeafNode() {
//#ifndef NGT_SHARED_MEMORY_ALLOCATOR
//#ifndef NGT_NODE_USE_VECTOR
//                if (objectIDs != 0) {
//                    delete[] objectIDs;
//                }
//#endif
//#endif
//            }
//
//            static int
//            selectPivotByMaxDistance(Container &iobj, Node::Objects &fs);
//
//            static int
//            selectPivotByMaxVariance(Container &iobj, Node::Objects &fs);
//
//            static void
//            splitObjects(Container &insertedObject, Objects &splitObjectSet, int pivot);
//
//#if defined(NGT_SHARED_MEMORY_ALLOCATOR)
//            void removeObject(size_t id, size_t replaceId, SharedMemoryAllocator &allocator);
//#else
//            void removeObject(size_t id, size_t replaceId);
//#endif
//
//#ifdef NGT_SHARED_MEMORY_ALLOCATOR
//            #ifndef NGT_NODE_USE_VECTOR
//    NGT::ObjectDistance *getObjectIDs(SharedMemoryAllocator &allocator) {
//      return (NGT::ObjectDistance *)allocator.getAddr(objectIDs);
//    }
//#endif
//#else // NGT_SHARED_MEMORY_ALLOCATOR
//            NGT::ObjectDistance *getObjectIDs() { return objectIDs; }
//#endif // NGT_SHARED_MEMORY_ALLOCATOR
//
//            void serialize(std::ofstream &os, ObjectSpace *objectspace = 0) {
//                Node::serialize(os);
//#ifdef NGT_NODE_USE_VECTOR
//                NGT::Serializer::write(os, objectIDs);
//#else
//                NGT::Serializer::write(os, objectSize);
//                for (int i = 0; i < objectSize; i++) {
//#ifdef NGT_SHARED_MEMORY_ALLOCATOR
//                    std::cerr << "not implemented" << std::endl;
//	assert(0);
//#else
//                    objectIDs[i].serialize(os);
//#endif
//                }
//#endif // NGT_NODE_USE_VECTOR
//                if (pivot == 0) {
//                    // Before insertion, parent ID == 0 and object size == 0, that indicates an empty index
//                    if (parent.getID() != 0 || objectSize != 0) {
//                        NGTThrowException("Node::write: pivot is null!");
//                    }
//                } else {
//#ifdef NGT_SHARED_MEMORY_ALLOCATOR
//                    std::cerr << "not implemented" << std::endl;
//	assert(0);
//#else
//                    assert(objectspace != 0);
//                    pivot->serialize(os, objectspace);
//#endif
//                }
//            }
//            void deserialize(std::ifstream &is, ObjectSpace *objectspace = 0) {
//                Node::deserialize(is);
//
//#ifdef NGT_NODE_USE_VECTOR
//                objectIDs.clear();
//      NGT::Serializer::read(is, objectIDs);
//#else
//                assert(objectIDs != 0);
//                NGT::Serializer::read(is, objectSize);
//                for (int i = 0; i < objectSize; i++) {
//#if defined(NGT_SHARED_MEMORY_ALLOCATOR)
//                    std::cerr << "not implemented" << std::endl;
//	assert(0);
//#else
//                    getObjectIDs()[i].deserialize(is);
//#endif
//                }
//#endif
//                if (parent.getID() == 0 && objectSize == 0) {
//                    // The index is empty
//                    return;
//                }
//                if (pivot == 0) {
//#ifdef NGT_SHARED_MEMORY_ALLOCATOR
//                    pivot = PersistentObject::allocate(*objectspace);
//#else
//                    pivot = PersistentObject::allocate(*objectspace);
//                    assert(pivot != 0);
//#endif
//                }
//                assert(objectspace != 0);
//#if defined(NGT_SHARED_MEMORY_ALLOCATOR)
//                std::cerr << "not implemented" << std::endl;
//      assert(0);
//#else
//                getPivot().deserialize(is, objectspace);
//#endif
//            }
//
//#if defined(NGT_SHARED_MEMORY_ALLOCATOR)
//            void serializeAsText(std::ofstream &os, SharedMemoryAllocator &allocator, ObjectSpace *objectspace = 0) {
//#else
//            void serializeAsText(std::ofstream &os, ObjectSpace *objectspace = 0) {
//#endif
//                Node::serializeAsText(os);
//                os << " ";
//                if (pivot == 0) {
//                    NGTThrowException("Node::write: pivot is null!");
//                }
//#ifdef NGT_SHARED_MEMORY_ALLOCATOR
//                    getPivot(*objectspace).serializeAsText(os, objectspace);
//#else
//                assert(pivot != 0);
//                assert(objectspace != 0);
//#if defined(NGT_SHARED_MEMORY_ALLOCATOR)
//                pivot->serializeAsText(os, allocator, objectspace);
//#else
//                pivot->serializeAsText(os, objectspace);
//#endif
//#endif
//                os << " ";
//#ifdef NGT_NODE_USE_VECTOR
//                NGT::Serializer::writeAsText(os, objectIDs);
//#else
//                NGT::Serializer::writeAsText(os, objectSize);
//                for (int i = 0; i < objectSize; i++) {
//                    os << " ";
//#ifdef NGT_SHARED_MEMORY_ALLOCATOR
//                    getObjectIDs(allocator)[i].serializeAsText(os);
//#else
//                    objectIDs[i].serializeAsText(os);
//#endif
//                }
//#endif
//            }
//
//#if defined(NGT_SHARED_MEMORY_ALLOCATOR)
//            void deserializeAsText(std::ifstream &is, SharedMemoryAllocator &allocator, ObjectSpace *objectspace = 0) {
//#else
//            void deserializeAsText(std::ifstream &is, ObjectSpace *objectspace = 0) {
//#endif
//                Node::deserializeAsText(is);
//                if (pivot == 0) {
//#ifdef NGT_SHARED_MEMORY_ALLOCATOR
//                    pivot = PersistentObject::allocate(*objectspace);
//#else
//                    pivot = PersistentObject::allocate(*objectspace);
//#endif
//                }
//                assert(objectspace != 0);
//#if defined(NGT_SHARED_MEMORY_ALLOCATOR)
//                getPivot(*objectspace).deserializeAsText(is, objectspace);
//#else
//                getPivot().deserializeAsText(is, objectspace);
//#endif
//#ifdef NGT_NODE_USE_VECTOR
//                objectIDs.clear();
//      NGT::Serializer::readAsText(is, objectIDs);
//#else
//                assert(objectIDs != 0);
//                NGT::Serializer::readAsText(is, objectSize);
//                for (int i = 0; i < objectSize; i++) {
//#if defined(NGT_SHARED_MEMORY_ALLOCATOR)
//                    getObjectIDs(allocator)[i].deserializeAsText(is);
//#else
//                    getObjectIDs()[i].deserializeAsText(is);
//#endif
//                }
//#endif
//            }
//
//            void show() {
//                std::cout << "Show leaf node " << objectSize << ":";
//                for (int i = 0; i < objectSize; i++) {
//#if defined(NGT_SHARED_MEMORY_ALLOCATOR)
//                    std::cerr << "not implemented" << std::endl;
//	assert(0);
//#else
//                    std::cout << getObjectIDs()[i].id << "," << getObjectIDs()[i].distance << " ";
//#endif
//                }
//                std::cout << std::endl;
//            }
//
//#if defined(NGT_SHARED_MEMORY_ALLOCATOR)
//            bool verify(size_t nobjs, std::vector<uint8_t> &status, SharedMemoryAllocator &allocator);
//#else
//            bool verify(size_t nobjs, std::vector<uint8_t> &status);
//#endif
//
//
//#ifdef NGT_NODE_USE_VECTOR
//            size_t getObjectSize() { return objectIDs.size(); }
//#else
//            size_t getObjectSize() { return objectSize; }
//#endif
//
//            static const size_t LeafObjectsSizeMax		= 100;
//
//#ifdef NGT_NODE_USE_VECTOR
//            std::vector<Object>	objectIDs;
//#else
//            unsigned short	objectSize;
//#ifdef NGT_SHARED_MEMORY_ALLOCATOR
//            off_t		objectIDs;
//#else
//            ObjectDistance	*objectIDs;
//#endif
//#endif
        };


    class ObjectDistance
        {
        public:
            ObjectDistance() : id(0), distance(0.0) {}
            ObjectDistance(unsigned int i, float d) : id(i), distance(d) {}
            inline bool operator==(const ObjectDistance &o) const
            {
                return (distance == o.distance) && (id == o.id);
            }
            inline void set(unsigned int i, float d)
            {
                id = i;
                distance = d;
            }
            inline bool operator<(const ObjectDistance &o) const
            {
                if (distance == o.distance)
                {
                    return id < o.id;
                }
                else
                {
                    return distance < o.distance;
                }
            }
            inline bool operator>(const ObjectDistance &o) const
            {
                if (distance == o.distance)
                {
                    return id > o.id;
                }
                else
                {
                    return distance > o.distance;
                }
            }

            friend std::ostream &operator<<(std::ostream &os, const ObjectDistance &o)
            {
                os << o.id << " " << o.distance;
                return os;
            }
            friend std::istream &operator>>(std::istream &is, ObjectDistance &o)
            {
                is >> o.id;
                is >> o.distance;
                return is;
            }
            uint32_t id;
            float distance;
        };

        class ObjectDistances : public std::vector<ObjectDistance> {
        public:
            ObjectDistances() {}

            void moveFrom(std::priority_queue<ObjectDistance, std::vector<ObjectDistance>, std::less<ObjectDistance> > &pq) {
                this->clear();
                this->resize(pq.size());
                for (int i = pq.size() - 1; i >= 0; i--) {
                    (*this)[i] = pq.top();
                    pq.pop();
                }
                assert(pq.size() == 0);
            }

            void moveFrom(std::priority_queue<ObjectDistance, std::vector<ObjectDistance>, std::less<ObjectDistance> > &pq, double (&f)(double)) {
                this->clear();
                this->resize(pq.size());
                for (int i = pq.size() - 1; i >= 0; i--) {
                    (*this)[i] = pq.top();
                    (*this)[i].distance = f((*this)[i].distance);
                    pq.pop();
                }
                assert(pq.size() == 0);
            }

            void moveFrom(std::priority_queue<ObjectDistance, std::vector<ObjectDistance>, std::less<ObjectDistance> > &pq, unsigned int id) {
                this->clear();
                if (pq.size() == 0) {
                    return;
                }
                this->resize(id == 0 ? pq.size() : pq.size() - 1);
                int i = this->size() - 1;
                while (pq.size() != 0 && i >= 0) {
                    if (pq.top().id != id) {
                        (*this)[i] = pq.top();
                        i--;
                    }
                    pq.pop();
                }
                if (pq.size() != 0 && pq.top().id != id) {
                    std::cerr << "moveFrom: Fatal error: somethig wrong! " << pq.size() << ":" << this->size() << ":" << id << ":" << pq.top().id << std::endl;
                    assert(pq.size() == 0 || pq.top().id == id);
                }
            }

            ObjectDistances &operator=(ObjectDistances &objs);
        };

        class Container
        {
        public:
            Container(float *o, unsigned i) : object(o), id(i) {}
            Container(Container &c) : object(c.object), id(c.id) {}
            float *object;
            unsigned id;
        };

        typedef std::priority_queue<ObjectDistance, std::vector<ObjectDistance>, std::less<ObjectDistance>> ResultPriorityQueue;

        class SearchContainer : public Container
        {
        public:
            SearchContainer(float *f, unsigned i) : Container(f, i) { initialize(); }
            SearchContainer(float *f) : Container(f, 0) { initialize(); }
            SearchContainer(SearchContainer &sc, float *f) : Container(f, sc.id) { *this = sc; }

            SearchContainer &operator=(SearchContainer &sc)
            {
                size = sc.size;
                radius = sc.radius;
                explorationCoefficient = sc.explorationCoefficient;
                result = sc.result;
                distanceComputationCount = sc.distanceComputationCount;
                edgeSize = sc.edgeSize;
                workingResult = sc.workingResult;
                useAllNodesInLeaf = sc.useAllNodesInLeaf;
                expectedAccuracy = sc.expectedAccuracy;
                visitCount = sc.visitCount;
                return *this;
            }
            virtual ~SearchContainer() {}
            virtual void initialize()
            {
                size = 10;
                radius = FLT_MAX;
                explorationCoefficient = 1.1;
                result = 0;
                edgeSize = -1; // dynamically prune the edges during search. -1 means following the index property. 0 means using all edges.
                useAllNodesInLeaf = false;
                expectedAccuracy = -1.0;
            }
            void setSize(size_t s) { size = s; };
            void setResults(ObjectDistances *r) { result = r; }
            void setRadius(float r) { radius = r; }
            void setEpsilon(float e) { explorationCoefficient = e + 1.0; }
            void setEdgeSize(int e) { edgeSize = e; }
            void setExpectedAccuracy(float a) { expectedAccuracy = a; }

            inline bool resultIsAvailable() { return result != 0; }
            ObjectDistances &getResult()
            {
                if (result == 0)
                {
                    std::cout << "Inner error: results is not set" << std::endl;
                }
                return *result;
            }

            ResultPriorityQueue &getWorkingResult() { return workingResult; }

            size_t size;
            float radius;
            float explorationCoefficient;
            int edgeSize;
            size_t distanceComputationCount;
            ResultPriorityQueue workingResult;
            bool useAllNodesInLeaf;
            size_t visitCount;
            float expectedAccuracy;

        private:
            ObjectDistances *result;
        };

        class DVPTree {
        public:
            enum SplitMode {
                MaxDistance	= 0,
                MaxVariance	= 1
            };

            class Container : public NGT::Container {
            public:
                Container(float *f, unsigned i):NGT::Container(f, i) {}
                DVPTree			*vptree;
            };

            class SearchContainer : public NGT::SearchContainer {
            public:
                enum Mode {
                    SearchLeaf	= 0,
                    SearchObject	= 1
                };

                SearchContainer(float *f, ObjectID i):NGT::SearchContainer(f, i) {}
                SearchContainer(float *f):NGT::SearchContainer(f, 0) {}

                DVPTree			*vptree;

                Mode		mode;
                ID	nodeID;
            };
            class InsertContainer : public Container {
            public:
                InsertContainer(float *f, ObjectID i):Container(f, i) {}
            };

            DVPTree() {
                leafObjectsSize = LeafNode::LeafObjectsSizeMax;
                internalChildrenSize = InternalNode::InternalChildrenSizeMax;
                splitMode = MaxVariance;
            }

            Node *getNode(ID &id) {
                Node *n = 0;
                NodeID idx = id.getID();
                if (id.getType() == ID::Leaf) {
                    n = leafNodes.get(idx);
                } else {
                    n = internalNodes.get(idx);
                }
                return n;
            }

            void search(SearchContainer &sc) {
                ((SearchContainer&)sc).vptree = this;
                Node *root = getRootNode();
                assert(root != 0);
                if (sc.mode == DVPTree::SearchContainer::SearchLeaf) {
                    if (root->id.getType() == ID::Leaf) {
                        sc.nodeID.setRaw(root->id.get());
                        return;
                    }
                }

                UncheckedNode uncheckedNode;
                uncheckedNode.push(root->id);

                while (!uncheckedNode.empty()) {
                    ID nodeid = uncheckedNode.top();
                    uncheckedNode.pop();
                    Node *cnode = getNode(nodeid);
                    if (cnode == 0) {
                        std::cerr << "Error! child node is null. but continue." << std::endl;
                        continue;
                    }
                    if (cnode->id.getType() == ID::Internal) {
                        search(sc, (InternalNode&)*cnode, uncheckedNode);
                    } else if (cnode->id.getType() == ID::Leaf) {
                        search(sc, (LeafNode&)*cnode, uncheckedNode);
                    } else {
                        std::cerr << "Tree: Inner fatal error!: Node type error!" << std::endl;
                        abort();
                    }
                }
            }

            void insert(InsertContainer &iobj) {
                SearchContainer q(iobj.object);
                q.mode = SearchContainer::SearchLeaf;
                q.vptree = this;
                q.radius = 0.0;

                search(q);

                iobj.vptree = this;

                assert(q.nodeID.getType() == ID::Leaf);
                LeafNode *ln = (LeafNode*)getNode(q.nodeID);
                insert(iobj, ln);

                return;
            }

        public:
            int		internalChildrenSize;
            int		leafObjectsSize;

            SplitMode		splitMode;

            std::string		name;

            Repository<LeafNode>	leafNodes;
            Repository<InternalNode>	internalNodes;
        };

        unsigned edgeSizeForCreation;
        unsigned batchSizeForCreation;

        unsigned truncationThreshold;

        unsigned edgeSizeForSearch;
        unsigned size;
        float explorationCoefficient = 1.1;

        DVPTree dvp;
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

        // KDT/BKT 个数
        unsigned m_iTreeNumber;
        unsigned m_numTopDimensionKDTSplit = 5;
        // TPT 个数
        unsigned m_iTPTNumber = 32;
        // TPT 叶子个数
        unsigned m_iTPTLeafSize = 2000;
        unsigned m_numTopDimensionTPTSplit = 5;
        // K
        unsigned m_iNeighborhoodSize = 32;
        // K2 = K * m_iNeighborhoodScale
        unsigned m_iNeighborhoodScale = 2;
        // L
        unsigned m_iCEF = 1000;
        unsigned m_iHashTableExp = 4;

        // 抽样选取数量
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

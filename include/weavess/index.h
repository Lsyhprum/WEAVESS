//
// Created by MurphySL on 2020/9/14.
//

#ifndef WEAVESS_INDEX_H
#define WEAVESS_INDEX_H

#include <set>
#include <omp.h>
#include <mutex>
#include <queue>
#include <stack>
#include <thread>
#include <vector>
#include <chrono>
#include <fstream>
#include <cassert>
#include <iostream>
#include <unordered_set>
#include <boost/dynamic_bitset.hpp>
#include "util.h"
#include "policy.h"
#include "distance.h"
#include "parameters.h"

namespace weavess {

    class NNDescent {
    public:
        unsigned S;
        unsigned R;
        unsigned L;
        unsigned ITER;
        unsigned K;
        unsigned delta;

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

        struct SimpleNeighbor{
            unsigned id;
            float distance;

            SimpleNeighbor() = default;
            SimpleNeighbor(unsigned id, float distance) : id{id}, distance{distance}{}

            inline bool operator<(const SimpleNeighbor &other) const {
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
        unsigned R_nsg;     // nsg 中可变
        unsigned L_nsg;
        unsigned C_nsg;

        unsigned ep_;
    };

    class NSSG {
    public:
        float A;
        unsigned n_try;
        unsigned width;

        std::vector<unsigned> eps_;
        unsigned test_min = INT_MAX;
        unsigned test_max = 0;
        long long test_sum = 0;
    };

    class DPG {
    public:
        unsigned L_dpg;
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

    class NSW {
    public:
        unsigned NN_ ;
    };

    class HNSW {
    public:
        unsigned m_ = 12;
        unsigned max_m_ = 12;
        unsigned max_m0_ = 24;
        unsigned ef_construction_ = 150;
        unsigned n_threads_ = 1;
        unsigned mult;
        float level_mult_ = 1 / log(1.0*m_);

        class HnswNode {
        public:
            explicit HnswNode(int id, int level, size_t max_m, size_t max_m0)
                : id_(id), level_(level), max_m_(max_m), max_m0_(max_m0), friends_at_layer_(level+1) {
                for (int i = 1; i <= level; ++i) {
                    friends_at_layer_[i].reserve(max_m_ + 1);
                }
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
            void CopyLinksToOptIndex(char* mem_offset, int level) const;

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

        int max_level_ = 0;
        HnswNode* enterpoint_ = nullptr;
        std::vector<HnswNode*> nodes_;

        mutable std::mutex max_level_guard_;
    };

    class VAMANA {
    public:
        float alpha;
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
            Tnode *left;
            Tnode *right;
            bool isLeaf;

            std::vector<unsigned> val;
        };
        std::vector <Tnode> Tn;

        int xxx = 0;

        float S_hcnng = 0.0;
        unsigned N = 0;
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
    };

    class ANNG {
    public :
        unsigned truncationThreshold;
        unsigned edgeSizeForCreation;
        unsigned edgeSizeForSearch;
        unsigned size;
        float explorationCoefficient = 1.1;
    };

    class Index : public NNDescent, public NSG, public NSSG, public DPG, public EFANNA, public HNSW, public VAMANA, public HCNNG, public NSW, public IEH, public ANNG {
    public:

        explicit Index() {
            dist_ = new Distance();
        }

        ~Index() {
            delete dist_;
        }

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

        typedef std::vector<std::vector<std::vector<unsigned>>> CompactGraph;

        CompactGraph &getFinalGraph() {
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

        CompactGraph final_graph_;

        TYPE entry_type;
        TYPE candidate_type;
        TYPE prune_type;
        TYPE conn_type;

        unsigned dist_count = 0;

    };

}

#endif //WEAVESS_INDEX_H

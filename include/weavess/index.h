//
// Created by Murph on 2020/8/14.
//

#ifndef WEAVESS_INDEX_H
#define WEAVESS_INDEX_H

#define _CONTROL_NUM 100

//#include <utility>
#include <vector>
//#include <cstring>
#include <set>
#include <omp.h>
#include <mutex>
//#include "util.h"
#include "distance.h"
#include "neighbor.h"
#include "parameters.h"

namespace weavess {

    class IndexNSW {
    public:
        class MSWNode{
        public:
            // int -> IntType
            MSWNode(const float *data, size_t id) {
                nodeObj_ = data;
                id_ = id;
            }
            ~MSWNode(){};
            void removeAllFriends(){
                friends_.clear();
            }

            static void link(MSWNode* first, MSWNode* second){
                // addFriend checks for duplicates if the second argument is true
                first->addFriend(second, true);
                second->addFriend(first, true);
            }

            // Removes only friends from a given set
            void removeGivenFriends(const std::vector<bool>& delNodes) {
                size_t newQty = 0;
                /*
                 * This in-place one-iteration deletion of elements in delNodes
                 * Invariant in the beginning of each loop iteration:
                 * i >= newQty
                 * Furthermore:
                 * i - newQty == the number of entries deleted in previous iterations
                 */
                for (size_t i = 0; i < friends_.size(); ++i) {
                    int id = friends_[i]->getId();
                    if (!delNodes.at(id)) {
                        friends_[newQty] = friends_[i];
                        ++newQty;
                    }
                }
                friends_.resize(newQty);
            }

            // Removes friends from a given set and attempts to replace them with friends' closest neighbors
            // cacheDelNode should be thread-specific (or else calling this function isn't thread-safe)
//            template <class dist_t>
//            void removeGivenFriendsPatchWithClosestNeighbor(const Space<dist_t>& space, bool use_proxy_dist,
//                                                            const std::vector<bool>& delNodes, std::vector<MSWNode*>& cacheDelNode) {
//                /*
//                 * This in-place one-iteration deletion of elements in delNodes
//                 * Loop invariants:
//                 * 1) i >= newQty
//                 * 2) i - newQty = delQty
//                 * Hence, when the loop terminates delQty + newQty == friends_.size()
//                 */
//                size_t newQty = 0, delQty = 0;
//
//                for (size_t i = 0; i < friends_.size(); ++i) {
//                    MSWNode* oneFriend = friends_[i];
//                    int id = oneFriend->getId();
//                    if (!delNodes.at(id)) {
//                        friends_[newQty] = friends_[i];
//                        ++newQty;
//                    } else {
//                        if (cacheDelNode.size() <= delQty) cacheDelNode.resize(2*delQty + 1);
//                        cacheDelNode[delQty] = oneFriend;
//                        ++delQty;
//                    }
//                }
////                CHECK_MSG((delQty + newQty) == friends_.size(),
////                          "Seems like a bug, delQty:" + ConvertToString(delQty) +
////                          " newQty: " + ConvertToString(newQty) +
////                          " friends_.size()=" + ConvertToString(friends_.size()));
//                friends_.resize(newQty);
//                // When patching use the function link()
//                for (size_t i = 0; i < delQty; ++i) {
//                    MSWNode *toDelFriend = cacheDelNode[i];
//                    MSWNode *friendReplacement = nullptr;
//                    dist_t  dmin = numeric_limits<dist_t>::max();
//                    const Object* queryObj = this->getData();
//                    for (MSWNode* neighb : toDelFriend->getAllFriends()) {
//                        int neighbId = neighb->getId();
//                        if (!delNodes.at(neighbId)) {
//                            const MSWNode* provider = neighb;
//                            dist_t d = use_proxy_dist ?  space.ProxyDistance(provider->getData(), queryObj) :
//                                       space.IndexTimeDistance(provider->getData(), queryObj);
//                            if (d < dmin) {
//                                dmin = d;
//                                friendReplacement = neighb;
//                            }
//                        }
//                    }
//                    if (friendReplacement != nullptr) {
//                        link(this, friendReplacement);
//                    }
//                }
//            }
            /*
             * 1. The list of friend pointers is sorted.
             * 2. If bCheckForDup == true addFriend checks for
             *    duplicates using binary searching (via pointer comparison).
             */
            void addFriend(MSWNode* element, bool bCheckForDup) {
                std::unique_lock<std::mutex> lock(accessGuard_);

                if (bCheckForDup) {
                    auto it = lower_bound(friends_.begin(), friends_.end(), element);
                    if (it == friends_.end() || (*it) != element) {
                        friends_.insert(it, element);
                    }
                } else {
                    friends_.push_back(element);
                }
            }
            const float * getData() const {
                return nodeObj_;
            }
            size_t getId() const { return id_; }
            void setId(int id) { id_ = id; }
            /*
             * THIS NOTE APPLIES ONLY TO THE INDEXING PHASE:
             *
             * Before getting access to the friends,
             * one needs to lock the mutex accessGuard_
             * The mutex can be released ONLY when
             * we exit the scope that has access to
             * the reference returned by getAllFriends()
             */
            const std::vector<MSWNode*>& getAllFriends() const {
                return friends_;
            }

            std::mutex accessGuard_;

        private:
            const float*       nodeObj_;
            size_t              id_;
            std::vector<MSWNode*>    friends_;
        };
//----------------------------------

        class EvaluatedMSWNodeReverse{
        public:
            EvaluatedMSWNodeReverse() {
                distance = 0;
                element = NULL;
            }
            EvaluatedMSWNodeReverse(float di, MSWNode* node) {
                distance = di;
                element = node;
            }
            ~EvaluatedMSWNodeReverse(){}
            float getDistance() const {return distance;}
            MSWNode* getMSWNode() const {return element;}
            bool operator< (const EvaluatedMSWNodeReverse &obj1) const {
                return (distance > obj1.getDistance());
            }

        private:
            float distance;
            MSWNode* element;
        };

        class EvaluatedMSWNodeDirect{
        public:
            EvaluatedMSWNodeDirect() {
                distance = 0;
                element = NULL;
            }
            EvaluatedMSWNodeDirect(float di, MSWNode* node) {
                distance = di;
                element = node;
            }
            ~EvaluatedMSWNodeDirect(){}
            float getDistance() const {return distance;}
            MSWNode* getMSWNode() const {return element;}
            bool operator< (const EvaluatedMSWNodeDirect &obj1) const {
                return (distance < obj1.getDistance());
            }

        private:
            float distance;
            MSWNode* element;
        };

        mutable std::mutex   ElListGuard_;
        std::unordered_map<int, MSWNode*>      ElList_;
        int          NextNodeId_ = 0; // This is internal node id
        bool            changedAfterCreateIndex_ = false;
        MSWNode*        pEntryPoint_ = nullptr;
    };

    class IndexMST {
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
    };

    class IndexKDTree {
    public:
        // 节点不保存数据，只维护一个 LeafLists 中对应的数据编号
        struct Node {
            int DivDim;
            float DivVal;
            size_t StartIdx, EndIdx;
            unsigned treeid;
            Node *Lchild, *Rchild;
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
    };

    class IndexHash {
    public:
        int tablenum = 0;
        int upbits = 0;
        int codelength = 0;
        int codelengthshift = 0;
        int radius = 0;

        typedef std::vector<unsigned int> Codes;
        typedef std::unordered_map<unsigned int, std::vector<unsigned int> > HashBucket;
        typedef std::vector<HashBucket> HashTable;

        typedef std::vector<unsigned long> Codes64;
        typedef std::unordered_map<unsigned long, std::vector<unsigned int> > HashBucket64;
        typedef std::vector<HashBucket64> HashTable64;

        std::vector<HashTable> htb;
        std::vector<Codes> BaseCode;
        std::vector<Codes> QueryCode;
        std::vector<unsigned int> HammingBallMask;

        std::vector<HashTable64> htb64;
        std::vector<Codes64> BaseCode64;
        std::vector<Codes64> QueryCode64;
        std::vector<unsigned long> HammingBallMask64;

        std::vector<unsigned int> HammingRadius;

        // for statistic info
        std::vector<unsigned int> VisitBucketNum;

    };

    class IndexNSG {
    public:
        unsigned width; // 待删除
        unsigned ep_;
        std::vector<std::mutex> locks;
        char* opt_graph_;   // 删除
        size_t node_size;
        size_t data_len;    // 删除
        size_t neighbor_len;
        std::vector<nhood> nnd_graph;
    };

    class IndexNSSG {
    public:
        std::vector<unsigned> eps_;
    };

    class Index : public IndexNSG, public IndexNSSG, public IndexKDTree, public IndexHash, public IndexMST, public IndexNSW {
    public:
        float *data_ = nullptr;
        float *query_data_ = nullptr;
        unsigned *ground_data_ = nullptr;
        unsigned n_{};
        unsigned dim_{};
        unsigned query_num_{};
        unsigned query_dim_{};
        unsigned ground_num_{};
        unsigned ground_dim_{};

        Parameters param_;
        Distance *distance_ = nullptr;

        // init
        std::vector<nhood> graph_;
        // coarse
        std::vector<std::vector<unsigned> > final_graph_;

        // search_init
        //std::vector<std::vector<Neighbor> > entry_graph_;
        //std::vector<nhood> entry_graph_;
        std::vector<Neighbor> retset;


        explicit Index() {
            distance_ = new Distance();
        }

        explicit Index(Parameters &param, float *data, unsigned n, unsigned dim) : param_(param), data_(data), n_(n), dim_(dim) {
            std::cout << param_.ToString() << std::endl;
            std::cout << "data num : " << n_ << std::endl;
            std::cout << "data dim : " << dim_ << std::endl;
        }

        ~Index() {
            delete distance_;
        }
    };

}

#endif //WEAVESS_INDEX_H

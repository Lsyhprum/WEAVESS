//
// Created by Murph on 2020/8/14.
//

#ifndef WEAVESS_INDEX_H
#define WEAVESS_INDEX_H

#define _CONTROL_NUM 100

#include <utility>
#include <vector>
#include <cstring>
#include <set>
#include <omp.h>
#include "util.h"
#include "distance.h"
#include "neighbor.h"
#include "parameters.h"

namespace weavess {

    struct Node {
        int DivDim;
        float DivVal;
        size_t StartIdx, EndIdx;
        unsigned treeid;
        Node *Lchild, *Rchild;

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

    class Index {
    public:
        float *data_ = nullptr;
        unsigned dim_{};
        unsigned n_{};
        const Parameters param_;
        Distance *distance_ = new Distance();

        typedef std::vector<nhood> KNNGraph;
        typedef std::vector<std::vector<unsigned> > CompactGraph;
        typedef std::vector<LockNeighbor> LockGraph;

        Index *initializer_{};
        KNNGraph graph_;
        CompactGraph final_graph_;

        // KDTree
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
        std::vector<std::vector<unsigned>> LeafLists;
        omp_lock_t rootlock;
        bool error_flag;
        int max_deepth;
        int ml;   //merge_level
        unsigned K; //KNN Graph

        // nsg
        unsigned width;
        unsigned ep_;
        std::vector<std::mutex> locks;
        char* opt_graph_;
        size_t node_size;
        size_t data_len;
        size_t neighbor_len;
        KNNGraph nnd_graph;


        // Load
        enum FILE_TYPE {
            VECS
        };

        // Init
        enum INIT_TYPE {
            Random, KDTree, Other
        };

        // Coarse
        enum COARSE_TYPE {
            COARSE_NONE, NN_Descent
        };

        // prune
        enum PRUNE_TYPE {
            PRUNE_NONE, NSG, SSG
        };

        // connect
        enum CONN_TYPE {
            CONN_NONE, DFS
        };

        explicit Index(Parameters param) : param_(std::move(param)) {
            std::cout << param_.ToString() << std::endl;
        }

        virtual ~Index() = default;

        // 构建 Index 外部接口
        virtual void IndexLoadData(FILE_TYPE type, char *&data_file);

        virtual void IndexInit(INIT_TYPE type);

        virtual void IndexCoarseBuild(COARSE_TYPE type);

        virtual void IndexPrune(PRUNE_TYPE type);

        virtual void IndexConnect(CONN_TYPE type);

        virtual void
        Search(const float *query, const float *x, size_t K, const Parameters &parameter, unsigned *indices);

        virtual void FreeGraphData();

        inline float *GetData() { return data_; }

        inline unsigned int GetDim() const { return dim_; }

        inline unsigned GetNum() const { return n_; }

    };

}


#endif //WEAVESS_INDEX_H

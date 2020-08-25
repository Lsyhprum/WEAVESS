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
//#include "util.h"
#include "distance.h"
#include "neighbor.h"
#include "parameters.h"

namespace weavess {

    class IndexKDTree {
    public:
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
    };

    class IndexNSG {
    public:
        unsigned width;
        unsigned ep_;
        std::vector<std::mutex> locks;
        char* opt_graph_;
        size_t node_size;
        size_t data_len;
        size_t neighbor_len;
        std::vector<nhood> nnd_graph;
    };

    class IndexNSSG {
    public:
        std::vector<unsigned> eps_;
    };

    class Index : public IndexNSG, public IndexNSSG, public IndexKDTree {
    public:
        Parameters param_;
        float *data_ = nullptr;
        unsigned n_{};
        unsigned dim_{};

        Distance *distance_ = nullptr;

        std::vector<nhood> graph_;
        std::vector<std::vector<unsigned> > final_graph_;

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

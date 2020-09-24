//
// Created by MurphySL on 2020/9/14.
//

#ifndef WEAVESS_BUILDER_H
#define WEAVESS_BUILDER_H

#include "index.h"

namespace weavess {
    class IndexBuilder {
    public:
        explicit IndexBuilder() {
           final_index_ = new Index();
        }

        virtual ~IndexBuilder() {
            delete final_index_;
        }

        // dataset -> float*
        IndexBuilder *load(char *data_file, char *query_file, char *ground_file, Parameters &parameters);

        // float* -> vector<vector<vector<unsigned>>>
        IndexBuilder *init(TYPE type);

        // vector<vector<vector<unsigned>>> -> file
        IndexBuilder *save_graph(char *graph_file);

        // file -> vector<vector<vector<unsigned>>>
        IndexBuilder *load_graph(char *graph_file);

        // vector<Node *> -> vector<vector<unsigned>>
        //      candidate : vector<Node *> -> vector<Node *>
        //      prune     : vector<Node *> -> vector<vector<unsigned>>
        IndexBuilder *refine(TYPE type, bool debug);

        IndexBuilder *entry(TYPE type);

        IndexBuilder *route(TYPE type);

        IndexBuilder *search(TYPE entry_type, TYPE route_type);

        std::chrono::duration<double> GetBuildTime() { return e - s; }

        void degree_info();

        void conn_info();

        void DFS(boost::dynamic_bitset<> &flag, unsigned root, unsigned &cnt);

        void findroot(boost::dynamic_bitset<> &flag, unsigned &root);

    private:
        Index *final_index_;

        std::chrono::high_resolution_clock::time_point s;
        std::chrono::high_resolution_clock::time_point e;
    };
}

#endif //WEAVESS_BUILDER_H

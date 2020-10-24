//
// Created by MurphySL on 2020/10/23.
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

        IndexBuilder *load(char *data_file, char *query_file, char *ground_file, Parameters &parameters);

        IndexBuilder *init(TYPE type);

        IndexBuilder *save_graph(char *graph_file);

        IndexBuilder *load_graph(char *graph_file);

        IndexBuilder *refine(TYPE type, bool debug);

        IndexBuilder *search(TYPE entry_type, TYPE route_type);

        void degree_info(std::unordered_map<unsigned, unsigned> &degree);

        void conn_info();

        void DFS(boost::dynamic_bitset<> &flag, unsigned root, unsigned &cnt);

        void findRoot(boost::dynamic_bitset<> &flag, unsigned &root);

        IndexBuilder *draw();

        std::chrono::duration<double> GetBuildTime() { return e - s; }

    private:
        Index *final_index_;

        std::chrono::high_resolution_clock::time_point s;
        std::chrono::high_resolution_clock::time_point e;
    };
}

#endif //WEAVESS_BUILDER_H

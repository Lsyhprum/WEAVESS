//
// Created by MurphySL on 2020/10/23.
//

#ifndef WEAVESS_BUILDER_H
#define WEAVESS_BUILDER_H

#include "index.h"

namespace weavess {
    class IndexBuilder {
    public:
        explicit IndexBuilder(const unsigned num_threads) {
            final_index_ = new Index();
            omp_set_num_threads(num_threads);
        }

        virtual ~IndexBuilder() {
            delete final_index_;
        }

        IndexBuilder *load(char *data_file, char *query_file, char *ground_file, Parameters &parameters);

        IndexBuilder *serialization(TYPE type, char *graph_file);

        IndexBuilder *deserialization(TYPE type, char *graph_file);

        IndexBuilder *init(TYPE type, bool debug = false);

        IndexBuilder *save_graph(TYPE type, char *graph_file);

        IndexBuilder *load_graph(TYPE type, char *graph_file);

        IndexBuilder *refine(TYPE type, bool debug);

        IndexBuilder *refine(TYPE type, bool debug, Parameters &parameters) {
            final_index_->setParam(parameters);
            std::cout << final_index_->getParam().toString() << std::endl;
            refine(type, debug);
        }

        IndexBuilder *pre_entry(TYPE type) ;

        IndexBuilder *search(TYPE entry_type, TYPE route_type, TYPE L_type);

        void print_graph();

        void degree_info(std::unordered_map<unsigned, unsigned> &in_degree, std::unordered_map<unsigned, unsigned> &out_degree);

        void conn_info();

        void DFS(boost::dynamic_bitset<> &flag, unsigned root, unsigned &cnt);

        void findRoot(boost::dynamic_bitset<> &flag, unsigned &root);

        IndexBuilder *draw();

        std::chrono::duration<double> GetBuildTime() { return e - s; }

        void peak_memory_footprint();

    private:
        Index *final_index_;

        std::chrono::high_resolution_clock::time_point s;
        std::chrono::high_resolution_clock::time_point e;
    };
}

#endif //WEAVESS_BUILDER_H

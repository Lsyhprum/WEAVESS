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

        // float* -> vector<Node *>
        IndexBuilder *init(TYPE type);

        // vector<Node *> -> vector<vector<unsigned>>
        //      candidate : vector<Node *> -> vector<Node *>
        //      prune     : vector<Node *> -> vector<vector<unsigned>>
        IndexBuilder *refine(TYPE type);

        IndexBuilder *entry(TYPE type);

        IndexBuilder *route(TYPE type);

        std::chrono::duration<double> GetBuildTime() { return e - s; }

    private:
        Index *final_index_;

        std::chrono::high_resolution_clock::time_point s;
        std::chrono::high_resolution_clock::time_point e;
    };
}

#endif //WEAVESS_BUILDER_H

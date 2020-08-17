//
// Created by Murph on 2020/8/17.
//

#ifndef WEAVESS_INDEX_COMPONENT_H
#define WEAVESS_INDEX_COMPONENT_H

#include <cassert>
#include <string>
#include <ctime>
#include <chrono>
#include "index.h"
#include "parameters.h"

namespace weavess {

    class IndexBuilder {
    public:
        explicit IndexBuilder(Parameters parameters) {
            std::cout << "__Init Builder__" << std::endl;

            s = std::chrono::high_resolution_clock::now();
            index = new Index(std::move(parameters));
        }

        IndexBuilder *load(char *data_file, Index::FILE_TYPE type) {
            std::cout << "__Load Data__" << std::endl;

            index->IndexLoadData(type, data_file);
            return builder;
        }

        IndexBuilder *init(Index::INIT_TYPE type) {
            std::cout << "__Init Graph__" << std::endl;

            index->IndexInit(type);
            return builder;
        }

        IndexBuilder *coarse(Index::COARSE_TYPE type) {
            std::cout << "__COARSE KNN__" << std::endl;

            index->IndexCoarseBuild(type);
            return builder;
        }

        IndexBuilder *prune(Index::PRUNE_TYPE type) {
            std::cout << "__PRUNE__" << std::endl;

            index->IndexPrune(type);
            return builder;
        }

        IndexBuilder *connect(Index::CONN_TYPE type) {
            std::cout << "__CONNECT__" << std::endl;

            index->IndexConnect(type);
            e = std::chrono::high_resolution_clock::now();

            return builder;
        }

        Index *GetIndex() { return index; }

        std::chrono::duration<double> GetBuildTime() { return e-s; }

    private:
        IndexBuilder *builder = this;
        Index *index;

        std::chrono::high_resolution_clock::time_point s ;
        std::chrono::high_resolution_clock::time_point e ;
    };

}

#endif //WEAVESS_INDEX_COMPONENT_H

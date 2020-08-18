//
// Created by Murph on 2020/8/18.
//

#ifndef WEAVESS_INDEX_COMPONENT_LOAD_H
#define WEAVESS_INDEX_COMPONENT_LOAD_H

#include "index_component.h"

namespace weavess{

    class IndexComponentLoad : public IndexComponent {
    public:
        explicit IndexComponentLoad(Index *index) : IndexComponent(index) {}

        virtual void LoadDataInner(char *data_file, float *&data, unsigned int &dim, unsigned int &n) = 0;
    };

    class IndexComponentLoadVECS : public IndexComponentLoad {
    public:
        explicit IndexComponentLoadVECS(Index *index) : IndexComponentLoad(index) {}

        void LoadDataInner(char *data_file, float *&data, unsigned int &dim, unsigned int &n) override {
            load_data(data_file, data, n, dim);
        }
    };
}

#endif //WEAVESS_INDEX_COMPONENT_LOAD_H

//
// Created by Murph on 2020/8/18.
//

#ifndef WEAVESS_INDEX_COMPONENT_H
#define WEAVESS_INDEX_COMPONENT_H

#include "index.h"
#include <boost/dynamic_bitset.hpp>

namespace weavess{
    class IndexComponent {
    public:
        explicit IndexComponent(Index *index) : index_(index) {}
        explicit IndexComponent(Index *index, Parameters param) : index_(index), param_(std::move(param)) {}

        virtual ~IndexComponent() = default;

    protected:
        Index *index_;
        Parameters param_;
    };
}

#endif //WEAVESS_INDEX_COMPONENT_H

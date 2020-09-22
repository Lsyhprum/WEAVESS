//
// Created by Murph on 2020/9/15.
//

#include "weavess/component.h"

namespace weavess {

    void ComponentSerializationNNDescent::SaveGraphInner(const char *filename) {
//        std::ofstream out(filename, std::ios::binary | std::ios::out);
//        assert(index->getFinalGraph().size() == index->getBaseLen());
//
//        unsigned n = index->getBaseLen();
//        out.write((char *)&n, sizeof(unsigned));
//        for (unsigned i = 0; i < index->getBaseLen(); i++) {
//            unsigned GK = (unsigned)index->getFinalGraph()[i]->getFriendsAtLayer(0).size();
//            out.write((char *)&GK, sizeof(unsigned));
//            out.write((char *)index->getFinalGraph()[i]->getFriendsAtLayer(0).data(), GK * sizeof(unsigned));
//        }
//        out.close();
    }

    void ComponentSerializationNNDescent::LoadGraphInner(const char *filename) {
//        std::ifstream in(filename, std::ios::binary);
//
//        int n = 0;
//        in.read((char *)&n, sizeof(unsigned));
//        index->getFinalGraph().resize(n);
//
//        int id = 0;
//        while (!in.eof()) {
//            unsigned k;
//            in.read((char *)&k, sizeof(unsigned));
//            if (in.eof()) break;
//
//            auto node = new Index::Node(id, 0);
//            in.read((char *)node->getFriendsAtLayer(0).data(), k * sizeof(unsigned));
//
//            id ++;
//        }
    }
}

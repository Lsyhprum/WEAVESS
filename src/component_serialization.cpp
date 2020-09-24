//
// Created by Murph on 2020/9/15.
//

#include "weavess/component.h"

namespace weavess {

    void ComponentSerializationCompactGraph::SaveGraphInner(const char *filename) {
        std::ofstream out(filename, std::ios::binary | std::ios::out);
        assert(index->getFinalGraph().size() == index->getBaseLen());

        unsigned n = index->getBaseLen();
        out.write((char *)&n, sizeof(unsigned));
        for (unsigned i = 0; i < index->getBaseLen(); i++) {
            auto GK = (unsigned)index->getFinalGraph()[i][0].size();
            out.write((char *)&GK, sizeof(unsigned));
            out.write((char *)index->getFinalGraph()[i][0].data(), GK * sizeof(unsigned));
        }
        out.close();
    }

    void ComponentSerializationCompactGraph::LoadGraphInner(const char *filename) {
        std::ifstream in(filename, std::ios::binary);

        int n = 0;
        in.read((char *)&n, sizeof(unsigned));
        index->getFinalGraph().resize(n);

        int id = 0;
        while (!in.eof()) {
            unsigned k;
            in.read((char *) &k, sizeof(unsigned));
            if (in.eof()) break;
            //if(k != 200) std::cout << k << std::endl;

            index->getFinalGraph()[id].resize(1);
            index->getFinalGraph()[id][0].resize(k);
            in.read((char *) index->getFinalGraph()[id][0].data(), k * sizeof(unsigned));

            id++;
        }
    }
}

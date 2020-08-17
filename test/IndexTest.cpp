//
// Created by Murph on 2020/8/17.
//

#include <weavess/index_builder.h>


int main(int argc, char **argv) {

    std::string base_path = "F:\\ANNS\\dataset\\sift1M\\sift_base.fvecs";

    weavess::Parameters parameters;
    parameters.Set<unsigned>("K", 200);
    parameters.Set<unsigned>("L", 200);
    parameters.Set<unsigned>("iter", 10);
    parameters.Set<unsigned>("S", 10);
    parameters.Set<unsigned>("R", 100);

    auto *builder = new weavess::IndexBuilder(parameters);
    builder->load(&base_path[0], weavess::Index::FILE_TYPE::VECS)
            ->init(weavess::Index::INIT_TYPE::Random)
            ->coarse(weavess::Index::COARSE_TYPE::NN_Descent)
            ->prune(weavess::Index::PRUNE_NONE)
            ->connect(weavess::Index::CONN_NONE);
    delete builder;

    return 0;
}


//
// Created by Murph on 2020/8/17.
//

#include <weavess/index_builder.h>

weavess::Index *KGraph(std::string base_path) {
    weavess::Parameters parameters;
    parameters.Set<unsigned>("K", 200);
    parameters.Set<unsigned>("L", 200);
    parameters.Set<unsigned>("iter", 10);
    parameters.Set<unsigned>("S", 10);
    parameters.Set<unsigned>("R", 100);

    auto *builder = new weavess::IndexBuilder(parameters);
    builder->load(&base_path[0], weavess::Index::FILE_TYPE::VECS)
            ->init(weavess::Index::INIT_TYPE::Random)
            ->coarse(weavess::Index::COARSE_TYPE::NN_Descent);

    std::cout << "Time cost: " << builder->GetBuildTime().count() << std::endl;

    return builder->GetIndex();
}

weavess::Index *EFANNA(std::string base_path) {
    weavess::Parameters parameters;
    parameters.Set<unsigned>("K", 200);
    parameters.Set<unsigned>("nTrees", 200);
    parameters.Set<unsigned>("mLevel", 10);
    parameters.Set<unsigned>("L", 10);
    parameters.Set<unsigned>("iter", 10);
    parameters.Set<unsigned>("S", 10);
    parameters.Set<unsigned>("R", 10);

    auto *builder = new weavess::IndexBuilder(parameters);
    builder->load(&base_path[0], weavess::Index::FILE_TYPE::VECS)
            ->init(weavess::Index::INIT_TYPE::KDTree)
            ->coarse(weavess::Index::COARSE_TYPE::NN_Descent);

    std::cout << "Time cost: " << builder->GetBuildTime().count() << std::endl;

    return builder->GetIndex();
}

weavess::Index *NSG(std::string base_path) {
    weavess::Parameters parameters;
    parameters.Set<unsigned>("L", 100);
    parameters.Set<unsigned>("R", 100);
    parameters.Set<unsigned>("C", 100);

    auto *builder = new weavess::IndexBuilder(parameters);
    builder->load(&base_path[0], weavess::Index::FILE_TYPE::VECS)
            ->init(weavess::Index::INIT_TYPE::Random)
            ->coarse(weavess::Index::COARSE_TYPE::NN_Descent)
            ->connect(weavess::Index::DFS);

    std::cout << "Time cost: " << builder->GetBuildTime().count() << std::endl;

    return builder->GetIndex();
}

int main(int argc, char **argv) {

    std::string base_path = "F:\\ANNS\\dataset\\sift1M\\sift_base.fvecs";

    NSG(base_path);


    return 0;
}


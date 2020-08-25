//
// Created by Murph on 2020/8/17.
//

#include <weavess/index_builder.h>

void KGraph(std::string base_path, std::string query_path, std::string ground_path) {

    weavess::Parameters parameters;
    parameters.set<unsigned>("K", 200);
    parameters.set<unsigned>("L", 200);
    parameters.set<unsigned>("iter", 10);
    parameters.set<unsigned>("S", 10);
    parameters.set<unsigned>("R", 100);

    auto *builder = new weavess::IndexBuilder();
    builder -> load(&base_path[0], parameters)
            -> init(weavess::IndexBuilder::INIT_RAND)
            -> coarse(weavess::IndexBuilder::COARSE_NN_DESCENT)
            -> eva(weavess::IndexBuilder::SEARCH_RAND, &query_path[0], &ground_path[0]);

    std::cout << "Time cost: " << builder->GetBuildTime().count() << std::endl;
}

//weavess::Index *EFANNA(std::string base_path) {
//    weavess::Parameters parameters;
//    parameters.set<unsigned>("K", 200);
//    parameters.set<unsigned>("nTrees", 200);
//    parameters.set<unsigned>("mLevel", 10);
//    parameters.set<unsigned>("L", 10);
//    parameters.set<unsigned>("iter", 10);
//    parameters.set<unsigned>("S", 10);
//    parameters.set<unsigned>("R", 10);
//
//    auto *builder = new weavess::IndexBuilder(parameters);
//    builder->load(&base_path[0], weavess::Index::FILE_TYPE::VECS)
//            ->init(weavess::Index::INIT_TYPE::KDTree)
//            ->coarse(weavess::Index::COARSE_TYPE::NN_Descent);
//
//    std::cout << "Time cost: " << builder->GetBuildTime().count() << std::endl;
//
//    return builder->GetIndex();
//}
//
void NSG(std::string base_path, std::string query_path, std::string ground_path) {
    weavess::Parameters parameters;
    parameters.set<unsigned>("K", 200);
    parameters.set<unsigned>("L", 200);
    parameters.set<unsigned>("iter", 10);
    parameters.set<unsigned>("S", 10);
    parameters.set<unsigned>("R", 100);

    parameters.set<unsigned>("L_nsg", 40);
    parameters.set<unsigned>("R_nsg", 50);
    parameters.set<unsigned>("C_nsg", 500);

    auto *builder = new weavess::IndexBuilder();
    builder -> load(&base_path[0], parameters)
            -> init(weavess::IndexBuilder::INIT_RAND)
            -> coarse(weavess::IndexBuilder::COARSE_NN_DESCENT)
            -> prune(weavess::IndexBuilder::PRUNE_NSG)
            -> connect(weavess::IndexBuilder::CONN_DFS)
            -> eva(weavess::IndexBuilder::SEARCH_NSG, &query_path[0], &ground_path[0]);

    std::cout << "Time cost: " << builder->GetBuildTime().count() << std::endl;
}

void SSG(std::string base_path, std::string query_path, std::string ground_path){
    weavess::Parameters parameters;
    parameters.set<unsigned>("K", 200);
    parameters.set<unsigned>("L", 200);
    parameters.set<unsigned>("iter", 12);
    parameters.set<unsigned>("S", 10);
    parameters.set<unsigned>("R", 100);

    parameters.set<unsigned>("L_nsg", 100);
    parameters.set<unsigned>("R_nsg", 50);
    parameters.set<float>("A", 60);
    parameters.set<unsigned>("n_try", 10);

    auto *builder = new weavess::IndexBuilder();
    builder -> load(&base_path[0], parameters)
            -> init(weavess::IndexBuilder::INIT_RAND)
            -> coarse(weavess::IndexBuilder::COARSE_NN_DESCENT)
            -> prune(weavess::IndexBuilder::PRUNE_NSSG)
            -> connect(weavess::IndexBuilder::CONN_NSSG)
            -> eva(weavess::IndexBuilder::SEARCH_RAND, &query_path[0], &ground_path[0]);  // 待定

    std::cout << "Time cost: " << builder->GetBuildTime().count() << std::endl;
}

int main(int argc, char **argv) {

    std::string base_path = "F:\\ANNS\\dataset\\sift1M\\sift_base.fvecs";
    std::string query_path = "F:\\ANNS\\dataset\\sift1M\\sift_query.fvecs";
    std::string ground_path = "F:\\ANNS\\dataset\\sift1M\\sift_groundtruth.ivecs";

    NSG(base_path, query_path, ground_path);

    return 0;
}


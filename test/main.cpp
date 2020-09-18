//
// Created by MurphySL on 2020/9/14.
//

#include "weavess/builder.h"

void KGraph(std::string base_path, std::string query_path, std::string ground_path) {
    weavess::Parameters parameters;
    parameters.set<unsigned>("K", 10);
    parameters.set<unsigned>("L", 20);
    parameters.set<unsigned>("ITER", 1);
    parameters.set<unsigned>("S", 10);
    parameters.set<unsigned>("R", 100);

    auto *builder = new weavess::IndexBuilder();
    builder -> load(&base_path[0], &query_path[0], &ground_path[0], parameters)
            -> init(weavess::INIT_NN_DESCENT);

    std::cout << "Time cost: " << builder->GetBuildTime().count() << std::endl;
}

void NSG(std::string base_path, std::string query_path, std::string ground_path) {
    weavess::Parameters parameters;
    parameters.set<unsigned>("K", 30);
    parameters.set<unsigned>("L", 45);
    parameters.set<unsigned>("ITER", 1);
    parameters.set<unsigned>("S", 10);
    parameters.set<unsigned>("R", 20);

    parameters.set<unsigned>("L_nsg", 20);
    parameters.set<unsigned>("R_nsg", 30);
    parameters.set<unsigned>("C_nsg", 100);

    auto *builder = new weavess::IndexBuilder();
    builder -> load(&base_path[0], &query_path[0], &ground_path[0], parameters)
            -> init(weavess::INIT_NN_DESCENT)
            -> refine(weavess::REFINE_NSG);     // entry + candidate + prune

    std::cout << "Time cost: " << builder->GetBuildTime().count() << std::endl;
}

void NSSG(std::string base_path, std::string query_path, std::string ground_path) {
    weavess::Parameters parameters;
    parameters.set<unsigned>("K", 10);
    parameters.set<unsigned>("L", 10);
    parameters.set<unsigned>("ITER", 1);
    parameters.set<unsigned>("S", 10);
    parameters.set<unsigned>("R", 20);

    parameters.set<unsigned>("L_nsg", 40);
    parameters.set<unsigned>("R_nsg", 50);
    parameters.set<unsigned>("A", 500);
    parameters.set<unsigned>("n_try", 10);

    auto *builder = new weavess::IndexBuilder();
    builder -> load(&base_path[0], &query_path[0], &ground_path[0], parameters)
            -> init(weavess::INIT_NN_DESCENT)
            -> refine(weavess::REFINE_NSSG);     // entry + candidate + prune

    std::cout << "Time cost: " << builder->GetBuildTime().count() << std::endl;
}

void DPG(std::string base_path, std::string query_path, std::string ground_path) {
    weavess::Parameters parameters;
    parameters.set<unsigned>("K", 200);
    parameters.set<unsigned>("L", 200);
    parameters.set<unsigned>("iter", 10);
    parameters.set<unsigned>("S", 10);
    parameters.set<unsigned>("R", 100);

    //parameters.set<unsigned>("L_dpg", 100);  // half of "L"  ---> 200K --> 100 L

    auto *builder = new weavess::IndexBuilder();
    builder -> load(&base_path[0], &query_path[0], &ground_path[0], parameters)
            -> init(weavess::INIT_NN_DESCENT)
            -> refine(weavess::REFINE_DPG);

    std::cout << "Time cost: " << builder->GetBuildTime().count() << std::endl;
}

void EFANNA(std::string base_path, std::string query_path, std::string ground_path) {
    weavess::Parameters parameters;
    parameters.set<unsigned>("nTrees", 8);
    parameters.set<unsigned>("mLevel", 8);

    parameters.set<unsigned>("K", 200);
    parameters.set<unsigned>("L", 200);
    parameters.set<unsigned>("ITER", 8);
    parameters.set<unsigned>("S", 10);
    parameters.set<unsigned>("R", 10);

    auto *builder = new weavess::IndexBuilder();
    builder->load(&base_path[0], &query_path[0], &ground_path[0], parameters)
            ->init(weavess::INIT_KDT);
//            ->init(weavess::IndexBuilder::REFINE_NN_DESCENT)
//            -> eva(weavess::IndexBuilder::ENTRY_KDT, weavess::IndexBuilder::ROUTER_GREEDY);

    std::cout << "Time cost: " << builder->GetBuildTime().count() << std::endl;
}

void HNSW(std::string base_path, std::string query_path, std::string ground_path) {
    weavess::Parameters parameters;
    parameters.set<unsigned>("max_m", 1);
    parameters.set<unsigned>("max_m0", 11);
    parameters.set<unsigned>("ef_construction", 50);
    parameters.set<unsigned>("n_threads", 1);
    parameters.set<unsigned>("mult", 1);


    auto *builder = new weavess::IndexBuilder();
    builder -> load(&base_path[0], &query_path[0], &ground_path[0], parameters)
            -> refine(weavess::REFINE_HNSW);
}

void VAMANA(std::string base_path, std::string query_path, std::string ground_path) {
    weavess::Parameters parameters;
    parameters.set<float>("alpha", 1.0);

    auto *builder = new weavess::IndexBuilder();
    builder -> load(&base_path[0], &query_path[0], &ground_path[0], parameters)
            -> init(weavess::INIT_RANDOM)
            -> refine(weavess::REFINE_VAMANA);
//            ->init(weavess::IndexBuilder::COARSE_NN_DESCENT)
//            ->init(weavess::IndexBuilder::REFINE_VAMANA);
            //-> eva(weavess::IndexBuilder::ENTRY_RAND, weavess::IndexBuilder::ROUTER_GREEDY);

    std::cout << "Time cost: " << builder->GetBuildTime().count() << std::endl;
}


void NSW(std::string base_path, std::string query_path, std::string ground_path) {}

void HCNNG(std::string base_path, std::string query_path, std::string ground_path) {
    weavess::Parameters parameters;
    parameters.set<unsigned>("S", 1.0);
    parameters.set<unsigned>("N", 10);

    auto *builder = new weavess::IndexBuilder();
    builder -> load(&base_path[0], &query_path[0], &ground_path[0], parameters)
            -> init(weavess::INIT_HCNNG);
//            ->init(weavess::IndexBuilder::COARSE_NN_DESCENT)
//            ->init(weavess::IndexBuilder::REFINE_VAMANA);
    //-> eva(weavess::IndexBuilder::ENTRY_RAND, weavess::IndexBuilder::ROUTER_GREEDY);

    std::cout << "Time cost: " << builder->GetBuildTime().count() << std::endl;
}

void IEH(std::string base_path, std::string query_path, std::string ground_path) {}

void NGT(std::string base_path, std::string query_path, std::string ground_path) {}

void SPTAG(std::string base_path, std::string query_path, std::string ground_path) {}



int main() {
    std::string base_path = R"(F:\ANNS\dataset\sift1M\sift_base.fvecs)";
    std::string query_path = R"(F:\ANNS\dataset\sift1M\sift_query.fvecs)";
    std::string ground_path = R"(F:\ANNS\dataset\sift1M\sift_groundtruth.ivecs)";

    //KGraph(base_path, query_path, ground_path);
    //NSG(base_path, query_path, ground_path);
    //NSSG(base_path, query_path, ground_path);

    //EFANNA(base_path, query_path, ground_path);
    //HNSW(base_path, query_path, ground_path);
    //VAMANA(base_path, query_path, ground_path);
    //DPG(base_path, query_path, ground_path);

    //NSW(base_path, query_path, ground_path);
    //HCNNG(base_path, query_path, ground_path);
    //IEH(base_path, query_path, ground_path);

    //NGT(base_path, query_path, ground_path);
    //SPTAG(base_path, query_path, ground_path);

    return 0;
}


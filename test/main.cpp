//
// Created by MurphySL on 2020/9/14.
//

#include "weavess/builder.h"

void KGraph(std::string base_path, std::string query_path, std::string ground_path) {
    std::string graph_file = R"(kgraph.knng)";

    weavess::Parameters parameters;
    parameters.set<unsigned>("K", 25);
    parameters.set<unsigned>("L", 100);
    parameters.set<unsigned>("ITER", 30);
    parameters.set<unsigned>("S", 10);
    parameters.set<unsigned>("R", 100);
    parameters.set<float>("delta", 0.002);

    auto *builder = new weavess::IndexBuilder();
    builder -> load(&base_path[0], &query_path[0], &ground_path[0], parameters)
            -> init(weavess::INIT_NN_DESCENT, true)
            //-> draw()
            //-> load_graph(&graph_file[0]);
            -> search(weavess::ENTRY_RANDOM, weavess::ROUTER_GREEDY);

    std::cout << "Time cost: " << builder->GetBuildTime().count() << std::endl;
}

void NSG(std::string base_path, std::string query_path, std::string ground_path) {
    std::string graph_file = R"(nsg.knng)";

    weavess::Parameters parameters;
    parameters.set<unsigned>("K", 200);
    parameters.set<unsigned>("L", 200);
    parameters.set<unsigned>("ITER", 10);
    parameters.set<unsigned>("S", 10);
    parameters.set<unsigned>("R", 100);

    parameters.set<unsigned>("L_nsg", 40);
    parameters.set<unsigned>("R_nsg", 50);
    parameters.set<unsigned>("C_nsg", 500);

    auto *builder = new weavess::IndexBuilder();
    builder -> load(&base_path[0], &query_path[0], &ground_path[0], parameters)
            -> init(weavess::INIT_NN_DESCENT, false)
            //-> save_graph(&graph_file[0])
            //-> load_graph(&graph_file[0])
            -> refine(weavess::REFINE_NSG, true)
            -> search(weavess::ENTRY_NSG_CENTROID, weavess::ROUTER_GREEDY);

    std::cout << "Time cost: " << builder->GetBuildTime().count() << std::endl;
}

void NSSG(std::string base_path, std::string query_path, std::string ground_path) {
    std::string graph_file = R"(nsg.knng)";

    weavess::Parameters parameters;
    parameters.set<unsigned>("K", 200);
    parameters.set<unsigned>("L", 200);
    parameters.set<unsigned>("ITER", 1);
    parameters.set<unsigned>("S", 10);
    parameters.set<unsigned>("R", 100);

    parameters.set<unsigned>("L_nsg", 100);
    parameters.set<unsigned>("R_nsg", 50);
    parameters.set<float>("A", 60);
    parameters.set<unsigned>("n_try", 10);

    auto *builder = new weavess::IndexBuilder();
    builder -> load(&base_path[0], &query_path[0], &ground_path[0], parameters)
            -> init(weavess::INIT_NN_DESCENT, false)
            //-> load_graph(&graph_file[0])
            -> refine(weavess::REFINE_NSSG, true)
            -> search(weavess::SEARCH_ENTRY_NSSG_CENTROID, weavess::ROUTER_GREEDY);

    std::cout << "Time cost: " << builder->GetBuildTime().count() << std::endl;
}

void DPG(std::string base_path, std::string query_path, std::string ground_path) {
    std::string graph_file = R"(nsg.knng)";

    weavess::Parameters parameters;
    parameters.set<unsigned>("K", 200);
    parameters.set<unsigned>("L", 200);
    parameters.set<unsigned>("ITER", 10);
    parameters.set<unsigned>("S", 10);
    parameters.set<unsigned>("R", 100);

    auto *builder = new weavess::IndexBuilder();
    builder -> load(&base_path[0], &query_path[0], &ground_path[0], parameters)
            -> load_graph(&graph_file[0])
            -> refine(weavess::REFINE_DPG, true);

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
            ->init(weavess::INIT_KDT, true)
            ->refine(weavess::REFINE_NN_DESCENT, true);
//            -> eva(weavess::IndexBuilder::ENTRY_KDT, weavess::IndexBuilder::ROUTER_GREEDY);

    std::cout << "Time cost: " << builder->GetBuildTime().count() << std::endl;
}

void HNSW(std::string base_path, std::string query_path, std::string ground_path) {
    weavess::Parameters parameters;
    parameters.set<unsigned>("max_m", 1);
    parameters.set<unsigned>("max_m0", 11);
    parameters.set<unsigned>("ef_construction", 10);
    parameters.set<unsigned>("n_threads", 10);
    parameters.set<unsigned>("mult", 1);

    auto *builder = new weavess::IndexBuilder();
    builder -> load(&base_path[0], &query_path[0], &ground_path[0], parameters)
            -> refine(weavess::REFINE_HNSW, true);
}

void NSW(std::string base_path, std::string query_path, std::string ground_path) {
    weavess::Parameters parameters;
    parameters.set<unsigned>("NN", 1);
    parameters.set<unsigned>("ef_construction", 10);
    parameters.set<unsigned>("n_threads", 1);

    auto *builder = new weavess::IndexBuilder();
    builder -> load(&base_path[0], &query_path[0], &ground_path[0], parameters)
            -> refine(weavess::REFINE_NSW, true);
}

void VAMANA(std::string base_path, std::string query_path, std::string ground_path) {
    weavess::Parameters parameters;
    parameters.set<unsigned>("L", 200);
    parameters.set<unsigned>("L_nsg", 100);
    parameters.set<unsigned>("R_nsg", 10);
    parameters.set<float>("alpha", 1.0);

    auto *builder = new weavess::IndexBuilder();
    builder -> load(&base_path[0], &query_path[0], &ground_path[0], parameters)
            -> init(weavess::INIT_RANDOM, true)
            -> refine(weavess::REFINE_VAMANA, true);
    //-> eva(weavess::IndexBuilder::ENTRY_RAND, weavess::IndexBuilder::ROUTER_GREEDY);

    std::cout << "Time cost: " << builder->GetBuildTime().count() << std::endl;
}

void HCNNG(std::string base_path, std::string query_path, std::string ground_path) {
    weavess::Parameters parameters;
    parameters.set<unsigned>("S", 1000);
    parameters.set<unsigned>("N", 200);

    auto *builder = new weavess::IndexBuilder();
    builder -> load(&base_path[0], &query_path[0], &ground_path[0], parameters)
            -> init(weavess::INIT_HCNNG, false)
            -> search(weavess::ENTRY_RANDOM, weavess::ROUTE_GUIDE);

    std::cout << "Time cost: " << builder->GetBuildTime().count() << std::endl;
}

void IEH(std::string base_path, std::string query_path, std::string ground_path) {

}

void ANNG(std::string base_path, std::string query_path, std::string ground_path) {
    weavess::Parameters parameters;
    parameters.set<unsigned>("edgeSizeForCreation", 10); // 初始化边数阈值
    parameters.set<unsigned>("truncationThreshold", 10);
    parameters.set<unsigned>("edgeSizeForSearch", 10);
    parameters.set<unsigned>("size", 10);

    auto *builder = new weavess::IndexBuilder();
    builder -> load(&base_path[0], &query_path[0], &ground_path[0], parameters)
            -> refine(weavess::REFINE_ANNG, true)
            -> search();

    std::cout << "Time cost: " << builder->GetBuildTime().count() << std::endl;
}

void ONNG(std::string base_path, std::string query_path, std::string ground_path) {
    weavess::Parameters parameters;
    parameters.set<unsigned>("numOfOutgoingEdges", 10);
    parameters.set<unsigned>("numOfIncomingEdges", 100);
    parameters.set<unsigned>("numOfQueries", 200);
    parameters.set<unsigned>("numOfResultantObjects", 20);  // k

    auto *builder = new weavess::IndexBuilder();
    builder -> load(&base_path[0], &query_path[0], &ground_path[0], parameters)
            -> init() // ANNG
            -> refine(weavess::REFINE_ONNG);

    std::cout << "Time cost: " << builder->GetBuildTime().count() << std::endl;
}

void SPTAG(std::string base_path, std::string query_path, std::string ground_path) {}

void FANNG(std::string base_path, std::string query_path, std::string ground_path) {}

void InitTest(std::string base_path, std::string query_path, std::string ground_path) {
    weavess::Parameters parameters;

    // init
    parameters.set<unsigned>("K", 15);           // 初始阶段后 近邻最大个数

    parameters.set<unsigned>("L", 20);          // 初始阶段中 候选最大个数 (NN-Descent)
    parameters.set<unsigned>("ITER", 12);       // 初始阶段中 迭代次数 （NN-Descent)
    parameters.set<unsigned>("S", 10);          // 初始阶段中 局部连接近邻数 (NN-Descent)
    parameters.set<unsigned>("R", 20);          // 初始阶段中 反向集个数 （NN-Descent)
    parameters.set<float>("delta", 0.002);      // 初始阶段中 提前终止 （NN-Descent)

    parameters.set<unsigned>("nTrees", 3);      // 初始阶段中
    parameters.set<unsigned>("mLevel", 3);

    parameters.set<unsigned>("R_nsg", 5);      // 增强阶段后 近邻个数

    parameters.set<unsigned>("L_nsg", 40);      //
    parameters.set<unsigned>("C_nsg", 500);

    auto builder = new weavess::IndexBuilder();
    builder -> load(&base_path[0], &query_path[0], &ground_path[0], parameters)
            //-> init(weavess::INIT_RANDOM)
            -> init(weavess::INIT_NN_DESCENT, false)
            //-> init(weavess::INIT_KDT)
            -> refine(weavess::REFINE_TEST, true)
            //-> refine(weavess::REFINE_NSG, false)
            //-> refine(weavess::REFINE_DPG, false)
            -> draw()
            -> search(weavess::ENTRY_RANDOM, weavess::ROUTER_GREEDY);

    std::cout << "Time cost: " << builder->GetBuildTime().count() << std::endl;
}


int main() {
//    std::string test_path = R"(F:\ANNS\DATASET\test.txt)";

    std::string base_path = R"(F:\ANNS\DATASET\sift1M\sift_sample_base.fvecs)";
    std::string query_path = R"(F:\ANNS\DATASET\sift1M\sift_sample_query.fvecs)";
    std::string ground_path = R"(F:\ANNS\DATASET\sift1M\sift_sample_groundtruth.ivecs)";
//    std::string base_path = R"(F:\ANNS\DATASET\sift1M\sift_base.fvecs)";
//    std::string query_path = R"(F:\ANNS\DATASET\sift1M\sift_query.fvecs)";
//    std::string ground_path = R"(F:\ANNS\DATASET\sift1M\sift_groundtruth.ivecs)";
//    std::string base_path = R"(F:\ANNS\DATASET\siftsmall\siftsmall_base.fvecs)";
//    std::string query_path = R"(F:\ANNS\DATASET\siftsmall\siftsmall_query.fvecs)";
//    std::string ground_path = R"(F:\ANNS\DATASET\siftsmall\siftsmall_groundtruth.ivecs)";

    KGraph(base_path, query_path, ground_path);
    //NSG(base_path, query_path, ground_path);
    //NSSG(base_path, query_path, ground_path);
    //DPG(base_path, query_path, ground_path);
    //EFANNA(base_path, query_path, ground_path);

    //HNSW(base_path, query_path, ground_path);
    //NSW(base_path, query_path, ground_path);
    //VAMANA(base_path, query_path, ground_path);
    //HCNNG(base_path, query_path, ground_path);
    //IEH(base_path, query_path, ground_path);

    //NGT(base_path, query_path, ground_path);
    //SPTAG(base_path, query_path, ground_path);

    //InitTest(test_path, query_path, ground_path);

    return 0;
}


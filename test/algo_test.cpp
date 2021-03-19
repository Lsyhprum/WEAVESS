//
// Created by MurphySL on 2020/12/7.
//
#include <weavess/builder.h>
#include <iostream>

void KDRG(std::string base_path, std::string query_path, std::string ground_path) {
    std::string graph = "basin.graph";
    weavess::Parameters parameters;
    parameters.set<unsigned>("S", 50);
    parameters.set<unsigned>("R_refine", 10);

    auto* builder= new weavess::IndexBuilder(8);
    builder -> load(&base_path[0], &query_path[0], &ground_path[0], parameters)
            -> init(weavess::INIT_KNNG, true)
            -> refine(weavess::REFINE_KDRG, true)
            -> save_graph(weavess::INDEX_KGRAPH, &graph[0]);

    auto *builder2 = new weavess::IndexBuilder(8);
    builder2 -> load(&base_path[0], &query_path[0], &ground_path[0], parameters)
            -> load_graph(weavess::INDEX_KGRAPH, &graph[0])
            -> search(weavess::SEARCH_ENTRY_RAND, weavess::ROUTER_GREEDY, weavess::TYPE::L_SEARCH_ASCEND);

    std::cout << "Time cost: " << builder->GetBuildTime().count() << std::endl;
}

void EFANNA(std::string base_path, std::string query_path, std::string ground_path) {
    std::string graph = "efanna.graph";
    weavess::Parameters parameters;
    parameters.set<unsigned>("nTrees", 8);
    parameters.set<unsigned>("mLevel", 8);
    parameters.set<unsigned>("S", 100);

    parameters.set<unsigned>("K", 200);
    parameters.set<unsigned>("L", 200);
    parameters.set<unsigned>("ITER", 12);
    parameters.set<unsigned>("S", 10);
    parameters.set<unsigned>("R", 100);

    auto *builder = new weavess::IndexBuilder(8);
    builder -> load(&base_path[0], &query_path[0], &ground_path[0], parameters)
            -> init(weavess::INIT_KDT, true)
            -> refine(weavess::REFINE_EFANNA, true)
            -> save_graph(weavess::INDEX_EFANNA, &graph[0]);

    auto *builder2 = new weavess::IndexBuilder(8);
    builder2 -> load(&base_path[0], &query_path[0], &ground_path[0], parameters)
            -> load_graph(weavess::INDEX_EFANNA, &graph[0])
            -> search(weavess::SEARCH_ENTRY_KDT, weavess::ROUTER_GREEDY, weavess::TYPE::L_SEARCH_ASCEND);

    std::cout << "Time cost: " << builder->GetBuildTime().count() << std::endl;
}

void NSW(std::string base_path, std::string query_path, std::string ground_path) {
    weavess::Parameters parameters;
    parameters.set<unsigned>("NN", 10);          // K
    parameters.set<unsigned>("ef_construction", 50);        //L
    parameters.set<unsigned>("n_threads_", 1);

    auto *builder = new weavess::IndexBuilder(8);
    builder -> load(&base_path[0], &query_path[0], &ground_path[0], parameters)
            -> init(weavess::INIT_NSW)
            -> search(weavess::SEARCH_ENTRY_NONE, weavess::ROUTER_NSW, weavess::TYPE::L_SEARCH_ASCEND);

    std::cout << "Time cost: " << builder->GetBuildTime().count() << std::endl;
}

void PANNG(std::string base_path, std::string query_path, std::string ground_path) {
    std::string graph_file = R"(panng.graph)";

    weavess::Parameters parameters;
    parameters.set<unsigned>("NN", 10);          // K
    parameters.set<unsigned>("ef_construction", 50);        //L
    parameters.set<unsigned>("n_threads_", 1);
    //parameters.set<unsigned>("batchSizeForCreation", 200);

    auto *builder = new weavess::IndexBuilder(8);
    builder -> load(&base_path[0], &query_path[0], &ground_path[0], parameters)
            -> init(weavess::INIT_ANNG)
            -> refine(weavess::REFINE_PANNG, false)
            -> save_graph(weavess::INDEX_PANNG, &graph_file[0]);

    auto *builder2 = new weavess::IndexBuilder(8);
    builder2 -> load(&base_path[0], &query_path[0], &ground_path[0], parameters)
            -> load_graph(weavess::INDEX_PANNG, &graph_file[0])
            -> search(weavess::SEARCH_ENTRY_VPT, weavess::ROUTER_NGT, weavess::TYPE::L_SEARCH_ASCEND);

//    std::cout << "Time cost: " << builder->GetBuildTime().count() << std::endl;
}

void ONNG(std::string base_path, std::string query_path, std::string ground_path) {
    std::string graph_file = R"(onng.graph)";

    weavess::Parameters parameters;
    parameters.set<unsigned>("NN", 50);          // K
    parameters.set<unsigned>("ef_construction", 100);        //L
    parameters.set<unsigned>("n_threads_", 1);

    parameters.set<unsigned>("numOfOutgoingEdges", 20);
    parameters.set<unsigned>("numOfIncomingEdges", 50);
    parameters.set<unsigned>("numOfQueries", 200);
    parameters.set<unsigned>("numOfResultantObjects", 20);

    auto *builder = new weavess::IndexBuilder(8);
    builder -> load(&base_path[0], &query_path[0], &ground_path[0], parameters)
            -> init(weavess::INIT_ANNG, true)
            -> refine(weavess::REFINE_ONNG, true)
            -> refine(weavess::REFINE_PANNG, true)
            -> load_graph(weavess::INDEX_ONNG, &graph_file[0]);

    auto *builder2 = new weavess::IndexBuilder(8);
    builder2 -> load(&base_path[0], &query_path[0], &ground_path[0], parameters)
             -> save_graph(weavess::INDEX_ONNG, &graph_file[0])
            -> search(weavess::SEARCH_ENTRY_VPT, weavess::ROUTER_NGT, weavess::TYPE::L_SEARCH_SET_RECALL);

    std::cout << "Time cost: " << builder->GetBuildTime().count() << std::endl;
}

void SPTAG_KDT(std::string base_path, std::string query_path, std::string ground_path) {
    std::string graph = "sptag_kdt.graph";
    weavess::Parameters parameters;
    parameters.set<unsigned>("KDTNumber", 1);
    parameters.set<unsigned>("TPTNumber", 32);
    parameters.set<unsigned>("TPTLeafSize", 500);
    parameters.set<unsigned>("NeighborhoodSize", 32);
    parameters.set<unsigned>("GraphNeighborhoodScale", 2);
    parameters.set<unsigned>("CEF", 500);
    parameters.set<unsigned>("numOfThreads", 10);

    auto *builder = new weavess::IndexBuilder(8);
    builder -> load(&base_path[0], &query_path[0], &ground_path[0], parameters)
            -> init(weavess::INIT_SPTAG_KDT)
            -> refine(weavess::REFINE_SPTAG_KDT, false)
            -> save_graph(weavess::INDEX_SPTAG_KDT, &graph[0])
//     builder        -> load(&base_path[0], &query_path[0], &ground_path[0], parameters)
            -> load_graph(weavess::INDEX_SPTAG_KDT, &graph[0])
            -> search(weavess::SEARCH_ENTRY_NONE, weavess::ROUTER_SPTAG_KDT, weavess::TYPE::L_SEARCH_ASCEND);
    std::cout << "Time cost: " << builder->GetBuildTime().count() << std::endl;
}

void SPTAG_BKT(std::string base_path, std::string query_path, std::string ground_path) {
    std::string graph = "sptag_bkt.graph";

    weavess::Parameters parameters;

    parameters.set<unsigned>("BKTNumber", 1);
    parameters.set<unsigned>("BKTKMeansK", 4);
    parameters.set<unsigned>("TPTNumber", 4);
    parameters.set<unsigned>("TPTLeafSize", 1000);
    parameters.set<unsigned>("NeighborhoodSize", 32);
    parameters.set<unsigned>("GraphNeighborhoodScale", 2);
    parameters.set<unsigned>("CEF", 500);
    parameters.set<unsigned>("numOfThreads", 10);

    auto *builder = new weavess::IndexBuilder(8);
    builder -> load(&base_path[0], &query_path[0], &ground_path[0], parameters)
            -> init(weavess::INIT_SPTAG_BKT)
            -> refine(weavess::REFINE_SPTAG_BKT, false)
            -> save_graph(weavess::INDEX_SPTAG_BKT, &graph[0]);

    auto *builder2 = new weavess::IndexBuilder(8);
    builder2 -> load(&base_path[0], &query_path[0], &ground_path[0], parameters)
            -> load_graph(weavess::INDEX_SPTAG_BKT, &graph[0])
            -> search(weavess::SEARCH_ENTRY_NONE, weavess::ROUTER_SPTAG_BKT, weavess::TYPE::L_SEARCH_ASCEND);

    //std::cout << "Time cost: " << builder->GetBuildTime().count() << std::endl;
}

void HCNNG(std::string base_path, std::string query_path, std::string ground_path) {
    std::string graph = "hcnng.graph";

    weavess::Parameters parameters;
    parameters.set<unsigned>("minsize_cl", 1000);
    parameters.set<unsigned>("num_cl", 20);

    parameters.set<unsigned>("nTrees", 10);
    parameters.set<unsigned>("mLevel", 4);
    parameters.set<unsigned>("K", 10);

    auto *builder = new weavess::IndexBuilder(8);
    builder -> load(&base_path[0], &query_path[0], &ground_path[0], parameters)
            -> init(weavess::INIT_HCNNG)
            -> save_graph(weavess::INDEX_HCNNG, &graph[0]);

    auto *builder2 = new weavess::IndexBuilder(8);
    builder2 -> load(&base_path[0], &query_path[0], &ground_path[0], parameters)
            -> load_graph(weavess::INDEX_HCNNG, &graph[0])
            -> search(weavess::SEARCH_ENTRY_KDT, weavess::ROUTER_GUIDE, weavess::TYPE::L_SEARCH_ASCEND);

    //std::cout << "Time cost: " << builder->GetBuildTime().count() << std::endl;
}

int main(int argc, char** argv) {
//    std::string base_path = R"(G:\ANNS\dataset\n100000\random_base_n100000_d32_c10_s5.fvecs)";
//    std::string query_path = R"(G:\ANNS\dataset\n100000\random_query_n1000_d32_c10_s5.fvecs)";
//    std::string ground_path = R"(G:\ANNS\dataset\n100000\random_ground_truth_n1000_d32_c10_s5.ivecs)";

    std::string base_path = R"(G:\ANNS\dataset\siftsmall\siftsmall_base.fvecs)";
    std::string query_path = R"(G:\ANNS\dataset\siftsmall\siftsmall_query.fvecs)";
    std::string ground_path = R"(G:\ANNS\dataset\siftsmall\siftsmall_groundtruth.ivecs)";

//    std::string base_path = R"(G:\ANNS\dataset\sift1M\sift_base.fvecs)";
//    std::string query_path = R"(G:\ANNS\dataset\sift1M\sift_query.fvecs)";
//    std::string ground_path = R"(G:\ANNS\dataset\sift1M\sift_groundtruth.ivecs)";

    KDRG(base_path, query_path, ground_path);

    //ONNG(base_path, query_path, ground_path);
    //PANNG(base_path, query_path, ground_path);

    //HCNNG(base_path, query_path, ground_path);

    return 0;
}
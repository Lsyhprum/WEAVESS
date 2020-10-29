#include <weavess/builder.h>
#include <iostream>


void KGraph(std::string base_path, std::string query_path, std::string ground_path) {
    std::string graph_file = R"(kgraph.graph)";

    weavess::Parameters parameters;
    parameters.set<unsigned>("K", 25);
    parameters.set<unsigned>("L", 100);
    parameters.set<unsigned>("ITER", 10);
    parameters.set<unsigned>("S", 10);
    parameters.set<unsigned>("R", 100);

    auto *builder = new weavess::IndexBuilder();
    builder -> load(&base_path[0], &query_path[0], &ground_path[0], parameters)
            -> init(weavess::TYPE::INIT_NN_DESCENT)
            -> refine(weavess::TYPE::REFINE_NN_DESCENT, true)
            -> search(weavess::TYPE::SEARCH_ENTRY_RAND, weavess::TYPE::ROUTER_GREEDY);

    std::cout << "Time cost: " << builder->GetBuildTime().count() << std::endl;
}

void NSG(std::string base_path, std::string query_path, std::string ground_path) {
    std::string graph_file = R"(nsg.graph)";

    weavess::Parameters parameters;
    parameters.set<unsigned>("K", 200);
    parameters.set<unsigned>("L", 200);
    parameters.set<unsigned>("ITER", 3);
    parameters.set<unsigned>("S", 10);
    parameters.set<unsigned>("R", 100);

    parameters.set<unsigned>("L_refine", 40);
    parameters.set<unsigned>("R_refine", 50);
    parameters.set<unsigned>("C_refine", 500);

    auto *builder = new weavess::IndexBuilder();
    builder -> load(&base_path[0], &query_path[0], &ground_path[0], parameters)
            -> init(weavess::INIT_NN_DESCENT)
            -> refine(weavess::REFINE_NN_DESCENT, true)
            -> refine(weavess::REFINE_NSG, true)
            -> search(weavess::TYPE::SEARCH_ENTRY_CENTROID, weavess::TYPE::ROUTER_GREEDY);

    std::cout << "Time cost: " << builder->GetBuildTime().count() << std::endl;
}

void SSG(std::string base_path, std::string query_path, std::string ground_path) {
    std::string graph_file = R"(ssg.graph)";

    weavess::Parameters parameters;
    parameters.set<unsigned>("K", 200);
    parameters.set<unsigned>("L", 200);
    parameters.set<unsigned>("ITER", 1);
    parameters.set<unsigned>("S", 10);
    parameters.set<unsigned>("R", 100);

    parameters.set<unsigned>("L_refine", 100);
    parameters.set<unsigned>("R_refine", 50);
    parameters.set<float>("A", 60);
    parameters.set<unsigned>("n_try", 10);

    auto *builder = new weavess::IndexBuilder();
    builder -> load(&base_path[0], &query_path[0], &ground_path[0], parameters)
            -> init(weavess::INIT_NN_DESCENT)
            -> refine(weavess::REFINE_NN_DESCENT, true)
            -> refine(weavess::REFINE_SSG, false)
            -> search(weavess::SEARCH_ENTRY_SUB_CENTROID, weavess::ROUTER_GREEDY);

    std::cout << "Time cost: " << builder->GetBuildTime().count() << std::endl;
}

void DPG(std::string base_path, std::string query_path, std::string ground_path) {
    std::string graph_file = R"(dpg.knng)";

    weavess::Parameters parameters;
    parameters.set<unsigned>("K", 200);
    parameters.set<unsigned>("L", 200);
    parameters.set<unsigned>("ITER", 3);
    parameters.set<unsigned>("S", 10);
    parameters.set<unsigned>("R", 100);

    auto *builder = new weavess::IndexBuilder();
    builder -> load(&base_path[0], &query_path[0], &ground_path[0], parameters)
            -> init(weavess::INIT_NN_DESCENT)
            -> refine(weavess::REFINE_NN_DESCENT, true)
            -> refine(weavess::REFINE_DPG, true)
            -> search(weavess::TYPE::SEARCH_ENTRY_RAND, weavess::TYPE::ROUTER_GREEDY);

    std::cout << "Time cost: " << builder->GetBuildTime().count() << std::endl;
}

void VAMANA(std::string base_path, std::string query_path, std::string ground_path) {
    weavess::Parameters parameters;
    parameters.set<unsigned>("L", 200);
    parameters.set<unsigned>("R_refine", 100);
    parameters.set<float>("alpha", 1.5);

    auto *builder = new weavess::IndexBuilder();
    builder -> load(&base_path[0], &query_path[0], &ground_path[0], parameters)
            -> init(weavess::INIT_RAND)
            -> refine(weavess::REFINE_VAMANA, true)
            -> search(weavess::TYPE::SEARCH_ENTRY_CENTROID, weavess::TYPE::ROUTER_GREEDY);

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
    builder -> load(&base_path[0], &query_path[0], &ground_path[0], parameters)
            -> init(weavess::INIT_KDT)
            -> refine(weavess::REFINE_EFANNA, true)
            -> search(weavess::SEARCH_ENTRY_KDT, weavess::ROUTER_GREEDY);

    std::cout << "Time cost: " << builder->GetBuildTime().count() << std::endl;
}

void IEH(std::string base_path, std::string query_path, std::string ground_path) {
    weavess::Parameters parameters;
    parameters.set<std::string>("func", "");
    parameters.set<std::string>("basecode", "");

    parameters.set<unsigned>("expand", 8);
    parameters.set<unsigned>("iterlimit", 8);

    auto *builder = new weavess::IndexBuilder();
    builder -> load(&base_path[0], &query_path[0], &ground_path[0], parameters)
            -> init(weavess::INIT_IEH);
            //-> search(weavess::SEARCH_ENTRY_KDT, weavess::ROUTER_GREEDY);

    std::cout << "Time cost: " << builder->GetBuildTime().count() << std::endl;
}

void NSW(std::string base_path, std::string query_path, std::string ground_path) {
    weavess::Parameters parameters;
    parameters.set<unsigned>("NN", 10);          // K
    parameters.set<unsigned>("ef_construction", 50);        //L
    parameters.set<unsigned>("n_threads_", 1);

    auto *builder = new weavess::IndexBuilder();
    builder -> load(&base_path[0], &query_path[0], &ground_path[0], parameters)
            -> init(weavess::INIT_NSW)
            -> search(weavess::SEARCH_ENTRY_NONE, weavess::ROUTER_NSW);

    std::cout << "Time cost: " << builder->GetBuildTime().count() << std::endl;
}

void HNSW(std::string base_path, std::string query_path, std::string ground_path) {
    weavess::Parameters parameters;
    parameters.set<unsigned>("max_m", 50);
    parameters.set<unsigned>("max_m0", 100);
    parameters.set<unsigned>("ef_construction", 150);
    parameters.set<unsigned>("n_threads", 10);
    parameters.set<unsigned>("mult", 5);

    auto *builder = new weavess::IndexBuilder();
    builder -> load(&base_path[0], &query_path[0], &ground_path[0], parameters)
            -> init(weavess::INIT_HNSW)
            -> search(weavess::SEARCH_ENTRY_NONE, weavess::ROUTER_HNSW);

    std::cout << "Time cost: " << builder->GetBuildTime().count() << std::endl;
}

void NGT(std::string base_path, std::string query_path, std::string ground_path) {
    weavess::Parameters parameters;
    parameters.set<unsigned>("edgeSizeForCreation", 10); // 初始化边数阈值
    parameters.set<unsigned>("truncationThreshold", 10);
    parameters.set<unsigned>("edgeSizeForSearch", 10);
    parameters.set<unsigned>("size", 10);

    auto *builder = new weavess::IndexBuilder();
    builder -> load(&base_path[0], &query_path[0], &ground_path[0], parameters)
            -> init(weavess::INIT_ANNG)
            -> refine(weavess::REFINE_ONNG, false)
            -> search(weavess::SEARCH_ENTRY_NONE, weavess::ROUTER_NGT);

    std::cout << "Time cost: " << builder->GetBuildTime().count() << std::endl;
}

void HCNNG(std::string base_path, std::string query_path, std::string ground_path) {

}

void SPTAG(std::string base_path, std::string query_path, std::string ground_path) {
    weavess::Parameters parameters;
    parameters.set<unsigned>("numOfThreads", 1);

    auto *builder = new weavess::IndexBuilder();
    builder -> load(&base_path[0], &query_path[0], &ground_path[0], parameters)
            //-> init(weavess::INIT_SPTAG_KDT)
            -> init(weavess::INIT_SPTAG_BKT)
            -> refine(weavess::REFINE_RNG, true);

    std::cout << "Time cost: " << builder->GetBuildTime().count() << std::endl;
}

void FANNG(std::string base_path, std::string query_path, std::string ground_path) {

}


int main() {
    std::string base_path = R"(G:\ANNS\dataset\siftsmall\siftsmall_base.fvecs)";
    std::string query_path = R"(G:\ANNS\dataset\siftsmall\siftsmall_query.fvecs)";
    std::string ground_path = R"(G:\ANNS\dataset\siftsmall\siftsmall_groundtruth.ivecs)";

    //KGraph(base_path, query_path, ground_path);
    //NSG(base_path, query_path, ground_path);
    //SSG(base_path, query_path, ground_path);
    //DPG(base_path, query_path, ground_path);
    //VAMANA(base_path, query_path, ground_path);
    //EFANNA(base_path, query_path, ground_path);
    //NSW(base_path, query_path, ground_path);

    //HNSW(base_path, query_path, ground_path);
    //IEH(base_path, query_path, ground_path);

    //NGT(base_path, query_path, ground_path);
    //HCNNG(base_path, query_path, ground_path);
    SPTAG(base_path, query_path, ground_path);
    //FANNG(base_path, query_path, ground_path);


    return 0;
}
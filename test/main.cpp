#include <weavess/builder.h>
#include <weavess/exp_data.h>
#include <iostream>


void KGraph(weavess::Parameters &parameters) {

    const unsigned num_threads = parameters.get<unsigned>("n_threads");
    std::string base_path = parameters.get<std::string>("base_path");
    std::string query_path = parameters.get<std::string>("query_path");
    std::string ground_path = parameters.get<std::string>("ground_path");
    std::string graph_file = parameters.get<std::string>("graph_file");
    auto *builder = new weavess::IndexBuilder(num_threads);

    if (parameters.get<std::string>("exc_type") == "build") {   // build
        builder -> load(&base_path[0], &query_path[0], &ground_path[0], parameters)
                -> init(weavess::TYPE::INIT_RANDOM, false);
        std::cout << "Init cost: " << builder->GetBuildTime().count() << std::endl;
        // 注意正式实验要关闭NN-Descent迭代图质量信息输出 index.h文件中
        builder -> refine(weavess::TYPE::REFINE_NN_DESCENT, false)
                -> save_graph(weavess::TYPE::INDEX_KGRAPH, &graph_file[0]);
        std::cout << "Build cost: " << builder->GetBuildTime().count() << std::endl;
    }else if (parameters.get<std::string>("exc_type") == "search") {    // search
        builder -> load(&base_path[0], &query_path[0], &ground_path[0], parameters)
                -> load_graph(weavess::TYPE::INDEX_KGRAPH, &graph_file[0])
                -> search(weavess::TYPE::SEARCH_ENTRY_RAND, weavess::TYPE::ROUTER_GREEDY, weavess::TYPE::L_SEARCH_ASSIGN);
        builder -> peak_memory_footprint();
    }else {
        std::cout << "exc_type input error!" << std::endl;
    }
}

void FANNG(weavess::Parameters &parameters) {

    const unsigned num_threads = parameters.get<unsigned>("n_threads");
    std::string base_path = parameters.get<std::string>("base_path");
    std::string query_path = parameters.get<std::string>("query_path");
    std::string ground_path = parameters.get<std::string>("ground_path");
    std::string graph_file = parameters.get<std::string>("graph_file");
    auto *builder = new weavess::IndexBuilder(num_threads);

    if (parameters.get<std::string>("exc_type") == "build") {   // build
        builder -> load(&base_path[0], &query_path[0], &ground_path[0], parameters)
                -> init(weavess::TYPE::INIT_FANNG);
        std::cout << "Init cost: " << builder->GetBuildTime().count() << std::endl;
        builder -> refine(weavess::TYPE::REFINE_FANNG, false)
                -> save_graph(weavess::TYPE::INDEX_FANNG, &graph_file[0]);
        std::cout << "Build cost: " << builder->GetBuildTime().count() << std::endl;
    }else if (parameters.get<std::string>("exc_type") == "search") {    // search
        builder -> load(&base_path[0], &query_path[0], &ground_path[0], parameters)
                -> load_graph(weavess::TYPE::INDEX_FANNG, &graph_file[0])
                -> search(weavess::TYPE::SEARCH_ENTRY_RAND, weavess::TYPE::ROUTER_BACKTRACK, weavess::TYPE::L_SEARCH_SET_RECALL);
        builder -> peak_memory_footprint();
    }else {
        std::cout << "exc_type input error!" << std::endl;
    }
}

void NSG(weavess::Parameters &parameters) {

    const unsigned num_threads = parameters.get<unsigned>("n_threads");
    std::string base_path = parameters.get<std::string>("base_path");
    std::string query_path = parameters.get<std::string>("query_path");
    std::string ground_path = parameters.get<std::string>("ground_path");
    std::string graph_file = parameters.get<std::string>("graph_file");
    auto *builder = new weavess::IndexBuilder(num_threads);

    if (parameters.get<std::string>("exc_type") == "build") {   // build
        builder -> load(&base_path[0], &query_path[0], &ground_path[0], parameters)
                -> init(weavess::INIT_RANDOM)
                -> refine(weavess::REFINE_NN_DESCENT, false);
        std::cout << "Init cost: " << builder->GetBuildTime().count() << std::endl;
        builder -> refine(weavess::REFINE_NSG, false)
                -> save_graph(weavess::TYPE::INDEX_NSG, &graph_file[0]);
        std::cout << "Build cost: " << builder->GetBuildTime().count() << std::endl;
    }else if (parameters.get<std::string>("exc_type") == "search") {    // search
        builder -> load(&base_path[0], &query_path[0], &ground_path[0], parameters)
                -> load_graph(weavess::TYPE::INDEX_NSG, &graph_file[0])
                -> search(weavess::TYPE::SEARCH_ENTRY_CENTROID, weavess::TYPE::ROUTER_GREEDY, weavess::TYPE::L_SEARCH_SET_RECALL);
        builder -> peak_memory_footprint();
    }else {
        std::cout << "exc_type input error!" << std::endl;
    }
}

void SSG(weavess::Parameters &parameters) {

    const unsigned num_threads = parameters.get<unsigned>("n_threads");
    std::string base_path = parameters.get<std::string>("base_path");
    std::string query_path = parameters.get<std::string>("query_path");
    std::string ground_path = parameters.get<std::string>("ground_path");
    std::string graph_file = parameters.get<std::string>("graph_file");
    auto *builder = new weavess::IndexBuilder(num_threads);

    if (parameters.get<std::string>("exc_type") == "build") {   // build
        builder -> load(&base_path[0], &query_path[0], &ground_path[0], parameters)
                -> init(weavess::INIT_RANDOM)
                -> refine(weavess::REFINE_NN_DESCENT, false);
        std::cout << "Init cost: " << builder->GetBuildTime().count() << std::endl;
        builder -> refine(weavess::REFINE_SSG, false)
                -> save_graph(weavess::TYPE::INDEX_SSG, &graph_file[0]);
        std::cout << "Build cost: " << builder->GetBuildTime().count() << std::endl;
    }else if (parameters.get<std::string>("exc_type") == "search") {    // search
        builder -> load(&base_path[0], &query_path[0], &ground_path[0], parameters)
                -> load_graph(weavess::TYPE::INDEX_SSG, &graph_file[0])
                -> search(weavess::SEARCH_ENTRY_SUB_CENTROID, weavess::ROUTER_GREEDY, weavess::TYPE::L_SEARCH_SET_RECALL);
        builder -> peak_memory_footprint();
    }else {
        std::cout << "exc_type input error!" << std::endl;
    }
}

void DPG(weavess::Parameters &parameters) {

    const unsigned num_threads = parameters.get<unsigned>("n_threads");
    std::string base_path = parameters.get<std::string>("base_path");
    std::string query_path = parameters.get<std::string>("query_path");
    std::string ground_path = parameters.get<std::string>("ground_path");
    std::string graph_file = parameters.get<std::string>("graph_file");
    auto *builder = new weavess::IndexBuilder(num_threads);

    if (parameters.get<std::string>("exc_type") == "build") {   // build
        builder -> load(&base_path[0], &query_path[0], &ground_path[0], parameters)
                -> init(weavess::INIT_RANDOM)
                -> refine(weavess::REFINE_NN_DESCENT, false);
        std::cout << "Init cost: " << builder->GetBuildTime().count() << std::endl;
        builder -> refine(weavess::REFINE_DPG, false)
                -> save_graph(weavess::TYPE::INDEX_DPG,  &graph_file[0]);
        std::cout << "Build cost: " << builder->GetBuildTime().count() << std::endl;
    }else if (parameters.get<std::string>("exc_type") == "search") {    // search
        builder -> load(&base_path[0], &query_path[0], &ground_path[0], parameters)
                -> load_graph(weavess::TYPE::INDEX_DPG,  &graph_file[0])
                -> search(weavess::TYPE::SEARCH_ENTRY_RAND, weavess::TYPE::ROUTER_GREEDY, weavess::TYPE::L_SEARCH_SET_RECALL);
        builder -> peak_memory_footprint();
    }else {
        std::cout << "exc_type input error!" << std::endl;
    }
}

void VAMANA(weavess::Parameters &parameters) {

    const unsigned num_threads = parameters.get<unsigned>("n_threads");
    std::string base_path = parameters.get<std::string>("base_path");
    std::string query_path = parameters.get<std::string>("query_path");
    std::string ground_path = parameters.get<std::string>("ground_path");
    std::string graph_file = parameters.get<std::string>("graph_file");
    auto *builder = new weavess::IndexBuilder(num_threads);

    if (parameters.get<std::string>("exc_type") == "build") {   // build
        builder -> load(&base_path[0], &query_path[0], &ground_path[0], parameters)
                -> init(weavess::INIT_RAND);
        std::cout << "Init cost: " << builder->GetBuildTime().count() << std::endl;
        builder -> refine(weavess::REFINE_VAMANA, false)
                -> refine(weavess::REFINE_VAMANA, false)
                -> save_graph(weavess::TYPE::INDEX_VAMANA, &graph_file[0]);
        std::cout << "Build cost: " << builder->GetBuildTime().count() << std::endl;
    }else if (parameters.get<std::string>("exc_type") == "search") {    // search
        builder -> load(&base_path[0], &query_path[0], &ground_path[0], parameters)
                -> load_graph(weavess::TYPE::INDEX_VAMANA, &graph_file[0])
                -> search(weavess::TYPE::SEARCH_ENTRY_CENTROID, weavess::TYPE::ROUTER_GREEDY, weavess::TYPE::L_SEARCH_SET_RECALL);
        builder -> peak_memory_footprint();
    }else {
        std::cout << "exc_type input error!" << std::endl;
    }
}

void EFANNA(weavess::Parameters &parameters) {

    const unsigned num_threads = parameters.get<unsigned>("n_threads");
    std::string base_path = parameters.get<std::string>("base_path");
    std::string query_path = parameters.get<std::string>("query_path");
    std::string ground_path = parameters.get<std::string>("ground_path");
    std::string graph_file = parameters.get<std::string>("graph_file");
    auto *builder = new weavess::IndexBuilder(num_threads);

    if (parameters.get<std::string>("exc_type") == "build") {   // build
        builder -> load(&base_path[0], &query_path[0], &ground_path[0], parameters)
                -> init(weavess::INIT_KDT);
        std::cout << "Init cost: " << builder->GetBuildTime().count() << std::endl;
        builder -> refine(weavess::REFINE_EFANNA, false)
                -> save_graph(weavess::TYPE::INDEX_EFANNA, &graph_file[0]);
        std::cout << "Build cost: " << builder->GetBuildTime().count() << std::endl;
    }else if (parameters.get<std::string>("exc_type") == "search") {    // search
        builder -> load(&base_path[0], &query_path[0], &ground_path[0], parameters)
                -> load_graph(weavess::TYPE::INDEX_EFANNA, &graph_file[0])
                -> search(weavess::TYPE::SEARCH_ENTRY_KDT, weavess::TYPE::ROUTER_GREEDY, weavess::TYPE::L_SEARCH_SET_RECALL);
        builder -> peak_memory_footprint();
    }else {
        std::cout << "exc_type input error!" << std::endl;
    }
}

void IEH(std::string base_path, std::string query_path, std::string ground_path) {
    weavess::Parameters parameters;
    parameters.set<std::string>("train", "F:\\ANNS\\DATASET\\sift1M\\sift_base.fvecs");
    parameters.set<std::string>("test", "F:\\ANNS\\DATASET\\sift1M\\sift_query.fvecs");
    parameters.set<std::string>("func", "F:\\ANNS\\DATASET\\sift1M\\LSHfuncSift.txt");
    parameters.set<std::string>("basecode", "F:\\ANNS\\DATASET\\sift1M\\LSHtableSift.txt");
    parameters.set<std::string>("knntable", "F:\\ANNS\\DATASET\\sift1M\\sift_bf.index");

    parameters.set<unsigned>("expand", 8);
    parameters.set<unsigned>("iterlimit", 8);

    auto *builder = new weavess::IndexBuilder(8);
    builder -> load(&base_path[0], &query_path[0], &ground_path[0], parameters) // useless
            -> init(weavess::INIT_IEH)
            -> search(weavess::SEARCH_ENTRY_HASH, weavess::ROUTER_IEH, weavess::TYPE::L_SEARCH_SET_RECALL);

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
            -> search(weavess::SEARCH_ENTRY_NONE, weavess::ROUTER_NSW, weavess::TYPE::L_SEARCH_SET_RECALL);

    std::cout << "Time cost: " << builder->GetBuildTime().count() << std::endl;
}

void HNSW(std::string base_path, std::string query_path, std::string ground_path) {
    weavess::Parameters parameters;
    parameters.set<unsigned>("max_m", 5);
    parameters.set<unsigned>("max_m0", 10);
    parameters.set<unsigned>("ef_construction", 150);
    parameters.set<unsigned>("n_threads", 1);
    parameters.set<int>("mult", -1);

    auto *builder = new weavess::IndexBuilder(8);
    builder -> load(&base_path[0], &query_path[0], &ground_path[0], parameters)
            -> init(weavess::INIT_HNSW)
            -> search(weavess::SEARCH_ENTRY_NONE, weavess::ROUTER_HNSW, weavess::TYPE::L_SEARCH_SET_RECALL);

    std::cout << "Time cost: " << builder->GetBuildTime().count() << std::endl;
}

void PANNG(std::string base_path, std::string query_path, std::string ground_path) {
    weavess::Parameters parameters;
    parameters.set<unsigned>("NN", 10);          // K
    parameters.set<unsigned>("ef_construction", 50);        //L
    parameters.set<unsigned>("n_threads_", 1);
    //parameters.set<unsigned>("batchSizeForCreation", 200);

    auto *builder = new weavess::IndexBuilder(8);
    builder -> load(&base_path[0], &query_path[0], &ground_path[0], parameters)
            -> init(weavess::INIT_ANNG)
            -> refine(weavess::REFINE_PANNG, false)
            -> search(weavess::SEARCH_ENTRY_VPT, weavess::ROUTER_NGT, weavess::TYPE::L_SEARCH_SET_RECALL);

    std::cout << "Time cost: " << builder->GetBuildTime().count() << std::endl;
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
            -> search(weavess::SEARCH_ENTRY_VPT, weavess::ROUTER_NGT, weavess::TYPE::L_SEARCH_SET_RECALL);

    std::cout << "Time cost: " << builder->GetBuildTime().count() << std::endl;
}

void HCNNG(std::string base_path, std::string query_path, std::string ground_path) {
    weavess::Parameters parameters;
    parameters.set<unsigned>("minsize_cl", 1000);
    parameters.set<unsigned>("num_cl", 20);

    parameters.set<unsigned>("nTrees", 10);
    parameters.set<unsigned>("mLevel", 4);
    parameters.set<unsigned>("K", 10);

    auto *builder = new weavess::IndexBuilder(8);
    builder -> load(&base_path[0], &query_path[0], &ground_path[0], parameters)
            -> init(weavess::INIT_HCNNG)
            -> search(weavess::SEARCH_ENTRY_KDT, weavess::ROUTER_GREEDY, weavess::TYPE::L_SEARCH_SET_RECALL);

    std::cout << "Time cost: " << builder->GetBuildTime().count() << std::endl;
}

void SPTAG_KDT(std::string base_path, std::string query_path, std::string ground_path) {
    weavess::Parameters parameters;
    parameters.set<unsigned>("KDTNumber", 1);
    parameters.set<unsigned>("TPTNumber", 32);
    parameters.set<unsigned>("TPTLeafSize", 2000);
    parameters.set<unsigned>("NeighborhoodSize", 32);
    parameters.set<unsigned>("GraphNeighborhoodScale", 2);
    parameters.set<unsigned>("CEF", 1000);
    parameters.set<unsigned>("numOfThreads", 10);

    auto *builder = new weavess::IndexBuilder(8);
    builder -> load(&base_path[0], &query_path[0], &ground_path[0], parameters)
            -> init(weavess::INIT_SPTAG_KDT)
            -> refine(weavess::REFINE_SPTAG_KDT, false)
            -> search(weavess::SEARCH_ENTRY_NONE, weavess::ROUTER_SPTAG_KDT, weavess::TYPE::L_SEARCH_SET_RECALL);
    std::cout << "Time cost: " << builder->GetBuildTime().count() << std::endl;
}

void SPTAG_BKT(std::string base_path, std::string query_path, std::string ground_path) {
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
            -> search(weavess::SEARCH_ENTRY_NONE, weavess::ROUTER_SPTAG_BKT, weavess::TYPE::L_SEARCH_SET_RECALL);

    std::cout << "Time cost: " << builder->GetBuildTime().count() << std::endl;
}


int main(int argc, char** argv) {
    if (argc < 4 || argc > 5) {
        std::cout << "./main algorithm dataset exc_type [L_search]"
                << std::endl;
        exit(-1);
    }
    weavess::Parameters parameters;
    std::string dataset_root = R"(/Users/wmz/Documents/Postgraduate/Code/dataset/)";
    parameters.set<std::string>("dataset_root", dataset_root);
    parameters.set<unsigned>("n_threads", 8);
    std::string alg(argv[1]);
    std::string dataset(argv[2]);
    std::string exc_type(argv[3]);
    if (argc == 5) {
        unsigned L_search = (unsigned)atoi(argv[4]);
        parameters.set<unsigned>("L_search", L_search);
    }
    std::cout << "algorithm: " << alg << std::endl;
    std::cout << "dataset: " << dataset << std::endl;
    std::string graph_file(alg + "_" + dataset + ".graph");
    parameters.set<std::string>("graph_file", graph_file);
    parameters.set<std::string>("exc_type", exc_type);
    set_para(alg, dataset, parameters);

    // alg
    if (alg == "kgraph") {
        KGraph(parameters);
    }else if (alg == "fanng") {
        FANNG(parameters);
    }else if (alg == "nsg") {
        NSG(parameters);
    }else if (alg == "ssg") {
        SSG(parameters);
    }else if (alg == "dpg") {
        DPG(parameters);
    }else if (alg == "vamana") {
        VAMANA(parameters);
    }else if (alg == "efanna") {
        EFANNA(parameters);
    }
    else {
        std::cout << "alg input error!\n";
        exit(-1);
    }
    return 0;
}
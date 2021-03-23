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
        builder -> refine(weavess::TYPE::REFINE_NN_DESCENT, false)
                -> save_graph(weavess::TYPE::INDEX_KGRAPH, &graph_file[0]);
        std::cout << "Build cost: " << builder->GetBuildTime().count() << std::endl;
    }else if (parameters.get<std::string>("exc_type") == "search") {    // search
        builder -> load(&base_path[0], &query_path[0], &ground_path[0], parameters)
                -> load_graph(weavess::TYPE::INDEX_KGRAPH, &graph_file[0])
                -> search(weavess::TYPE::SEARCH_ENTRY_RAND, weavess::TYPE::ROUTER_GREEDY, weavess::TYPE::L_SEARCH_ASSIGN);
        builder -> peak_memory_footprint();
    }else if (parameters.get<std::string>("exc_type") == "info") {
        builder -> load(&base_path[0], &query_path[0], &ground_path[0], parameters)
                -> load_graph(weavess::TYPE::INDEX_KGRAPH, &graph_file[0])
                -> print_index_info(weavess::TYPE::INDEX_KGRAPH);
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
    }else if (parameters.get<std::string>("exc_type") == "info") {
        builder -> load(&base_path[0], &query_path[0], &ground_path[0], parameters)
                -> load_graph(weavess::TYPE::INDEX_FANNG, &graph_file[0])
                -> print_index_info(weavess::TYPE::INDEX_FANNG);
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
    }else if (parameters.get<std::string>("exc_type") == "info") {
        builder -> load(&base_path[0], &query_path[0], &ground_path[0], parameters)
                -> load_graph(weavess::TYPE::INDEX_NSG, &graph_file[0])
                -> print_index_info(weavess::TYPE::INDEX_NSG);
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
    }else if (parameters.get<std::string>("exc_type") == "info") {
        builder -> load(&base_path[0], &query_path[0], &ground_path[0], parameters)
                -> load_graph(weavess::TYPE::INDEX_SSG, &graph_file[0])
                -> print_index_info(weavess::TYPE::INDEX_SSG);
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
    }else if (parameters.get<std::string>("exc_type") == "info") {
        builder -> load(&base_path[0], &query_path[0], &ground_path[0], parameters)
                -> load_graph(weavess::TYPE::INDEX_DPG, &graph_file[0])
                -> print_index_info(weavess::TYPE::INDEX_DPG);
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
    }else if (parameters.get<std::string>("exc_type") == "info") {
        builder -> load(&base_path[0], &query_path[0], &ground_path[0], parameters)
                -> load_graph(weavess::TYPE::INDEX_VAMANA, &graph_file[0])
                -> print_index_info(weavess::TYPE::INDEX_VAMANA);
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
    }else if (parameters.get<std::string>("exc_type") == "info") {
        builder -> load(&base_path[0], &query_path[0], &ground_path[0], parameters)
                -> load_graph(weavess::TYPE::INDEX_EFANNA, &graph_file[0])
                -> print_index_info(weavess::TYPE::INDEX_EFANNA);
    }else {
        std::cout << "exc_type input error!" << std::endl;
    }
}

void IEH(weavess::Parameters &parameters) {

    const unsigned num_threads = parameters.get<unsigned>("n_threads");
    std::string base_path = parameters.get<std::string>("base_path");
    std::string query_path = parameters.get<std::string>("query_path");
    std::string ground_path = parameters.get<std::string>("ground_path");
    std::string graph_file = parameters.get<std::string>("graph_file");
    auto *builder = new weavess::IndexBuilder(num_threads);

    if (parameters.get<std::string>("exc_type") == "build") {   // build
        std::cout << "index should be built in WEAVESS/algo/" << std::endl;
    }else if (parameters.get<std::string>("exc_type") == "search") {    // search
        builder -> load(&base_path[0], &query_path[0], &ground_path[0], parameters)
                -> init(weavess::INIT_IEH)
                -> search(weavess::SEARCH_ENTRY_HASH, weavess::ROUTER_IEH, weavess::TYPE::L_SEARCH_SET_RECALL);
    }else if (parameters.get<std::string>("exc_type") == "info") {
        builder -> load(&base_path[0], &query_path[0], &ground_path[0], parameters)
                -> load_graph(weavess::TYPE::INDEX_IEH, &graph_file[0])
                -> print_index_info(weavess::TYPE::INDEX_IEH);
    }else {
        std::cout << "exc_type input error!" << std::endl;
    }
}

void NSW(weavess::Parameters &parameters) {

    const unsigned num_threads = parameters.get<unsigned>("n_threads");
    std::string base_path = parameters.get<std::string>("base_path");
    std::string query_path = parameters.get<std::string>("query_path");
    std::string ground_path = parameters.get<std::string>("ground_path");
    std::string graph_file = parameters.get<std::string>("graph_file");
    auto *builder = new weavess::IndexBuilder(num_threads);

    if (parameters.get<std::string>("exc_type") == "build") {   // build
        builder -> load(&base_path[0], &query_path[0], &ground_path[0], parameters)
                -> init(weavess::INIT_NSW)
                -> save_graph(weavess::TYPE::INDEX_NSW, &graph_file[0]);
        std::cout << "Build cost: " << builder->GetBuildTime().count() << std::endl;
    }else if (parameters.get<std::string>("exc_type") == "search") {    // search
        builder -> load(&base_path[0], &query_path[0], &ground_path[0], parameters)
                -> load_graph(weavess::TYPE::INDEX_NSW, &graph_file[0])
                -> search(weavess::TYPE::SEARCH_ENTRY_NONE, weavess::TYPE::ROUTER_NSW, weavess::TYPE::L_SEARCH_SET_RECALL);
        builder -> peak_memory_footprint();
    }else if (parameters.get<std::string>("exc_type") == "info") {
        builder -> load(&base_path[0], &query_path[0], &ground_path[0], parameters)
                -> load_graph(weavess::TYPE::INDEX_NSW, &graph_file[0])
                -> print_index_info(weavess::TYPE::INDEX_NSW);
    }else {
        std::cout << "exc_type input error!" << std::endl;
    }
}

void HNSW(weavess::Parameters &parameters) {
    
    const unsigned num_threads = parameters.get<unsigned>("n_threads");
    std::string base_path = parameters.get<std::string>("base_path");
    std::string query_path = parameters.get<std::string>("query_path");
    std::string ground_path = parameters.get<std::string>("ground_path");
    std::string graph_file = parameters.get<std::string>("graph_file");
    auto *builder = new weavess::IndexBuilder(num_threads);

    if (parameters.get<std::string>("exc_type") == "build") {   // build
        builder -> load(&base_path[0], &query_path[0], &ground_path[0], parameters)
                -> init(weavess::INIT_HNSW)
                -> save_graph(weavess::TYPE::INDEX_HNSW, &graph_file[0]);
        std::cout << "Build cost: " << builder->GetBuildTime().count() << std::endl;
    }else if (parameters.get<std::string>("exc_type") == "search") {    // search
        builder -> load(&base_path[0], &query_path[0], &ground_path[0], parameters)
                -> load_graph(weavess::TYPE::INDEX_HNSW, &graph_file[0])
                -> search(weavess::TYPE::SEARCH_ENTRY_NONE, weavess::TYPE::ROUTER_HNSW, weavess::TYPE::L_SEARCH_SET_RECALL);
        builder -> peak_memory_footprint();
    }else if (parameters.get<std::string>("exc_type") == "info") {
        builder -> load(&base_path[0], &query_path[0], &ground_path[0], parameters)
                -> load_graph(weavess::TYPE::INDEX_HNSW, &graph_file[0])
                -> print_index_info(weavess::TYPE::INDEX_HNSW);
    }else {
        std::cout << "exc_type input error!" << std::endl;
    }
}

void PANNG(weavess::Parameters &parameters) {
    const unsigned num_threads = parameters.get<unsigned>("n_threads");
    std::string base_path = parameters.get<std::string>("base_path");
    std::string query_path = parameters.get<std::string>("query_path");
    std::string ground_path = parameters.get<std::string>("ground_path");
    std::string graph_file = parameters.get<std::string>("graph_file");
    auto *builder = new weavess::IndexBuilder(num_threads);

    if (parameters.get<std::string>("exc_type") == "build") {   // build
        builder -> load(&base_path[0], &query_path[0], &ground_path[0], parameters)
                -> init(weavess::INIT_ANNG)
                -> refine(weavess::REFINE_PANNG, false)
                -> save_graph(weavess::TYPE::INDEX_PANNG, &graph_file[0]);
        std::cout << "Build cost: " << builder->GetBuildTime().count() << std::endl;
    }else if (parameters.get<std::string>("exc_type") == "search") {    // search
        builder -> load(&base_path[0], &query_path[0], &ground_path[0], parameters)
                -> load_graph(weavess::TYPE::INDEX_PANNG, &graph_file[0])
                -> search(weavess::TYPE::SEARCH_ENTRY_VPT, weavess::TYPE::ROUTER_NGT, weavess::TYPE::L_SEARCH_SET_RECALL);
        builder -> peak_memory_footprint();
    }else if (parameters.get<std::string>("exc_type") == "info") {
        builder -> load(&base_path[0], &query_path[0], &ground_path[0], parameters)
                -> load_graph(weavess::TYPE::INDEX_PANNG, &graph_file[0])
                -> print_index_info(weavess::TYPE::INDEX_PANNG);
    }else {
        std::cout << "exc_type input error!" << std::endl;
    }
}

void ONNG(weavess::Parameters &parameters) {

    const unsigned num_threads = parameters.get<unsigned>("n_threads");
    std::string base_path = parameters.get<std::string>("base_path");
    std::string query_path = parameters.get<std::string>("query_path");
    std::string ground_path = parameters.get<std::string>("ground_path");
    std::string graph_file = parameters.get<std::string>("graph_file");
    auto *builder = new weavess::IndexBuilder(num_threads);

    if (parameters.get<std::string>("exc_type") == "build") {   // build
        builder -> load(&base_path[0], &query_path[0], &ground_path[0], parameters)
                -> init(weavess::INIT_ANNG);
        std::cout << "Init cost: " << builder->GetBuildTime().count() << std::endl;
        builder -> refine(weavess::REFINE_ONNG, false);
        std::cout << "Refine_onng cost: " << builder->GetBuildTime().count() << std::endl;
        builder -> refine(weavess::REFINE_PANNG, false)
                -> save_graph(weavess::TYPE::INDEX_ONNG, &graph_file[0]);
        std::cout << "Build cost: " << builder->GetBuildTime().count() << std::endl;
    }else if (parameters.get<std::string>("exc_type") == "search") {    // search
        builder -> load(&base_path[0], &query_path[0], &ground_path[0], parameters)
                -> load_graph(weavess::TYPE::INDEX_ONNG, &graph_file[0])
                -> search(weavess::TYPE::SEARCH_ENTRY_VPT, weavess::TYPE::ROUTER_NGT, weavess::TYPE::L_SEARCH_SET_RECALL);
        builder -> peak_memory_footprint();
    }else if (parameters.get<std::string>("exc_type") == "info") {
        builder -> load(&base_path[0], &query_path[0], &ground_path[0], parameters)
                -> load_graph(weavess::TYPE::INDEX_ONNG, &graph_file[0])
                -> print_index_info(weavess::TYPE::INDEX_ONNG);
    }else {
        std::cout << "exc_type input error!" << std::endl;
    }
}

void HCNNG(weavess::Parameters &parameters) {

    const unsigned num_threads = parameters.get<unsigned>("n_threads");
    std::string base_path = parameters.get<std::string>("base_path");
    std::string query_path = parameters.get<std::string>("query_path");
    std::string ground_path = parameters.get<std::string>("ground_path");
    std::string graph_file = parameters.get<std::string>("graph_file");
    auto *builder = new weavess::IndexBuilder(num_threads);

    if (parameters.get<std::string>("exc_type") == "build") {   // build
        builder -> load(&base_path[0], &query_path[0], &ground_path[0], parameters)
                -> init(weavess::INIT_HCNNG)
                -> save_graph(weavess::TYPE::INDEX_HCNNG, &graph_file[0]);
        std::cout << "Build cost: " << builder->GetBuildTime().count() << std::endl;
    }else if (parameters.get<std::string>("exc_type") == "search") {    // search
        builder -> load(&base_path[0], &query_path[0], &ground_path[0], parameters)
                -> load_graph(weavess::TYPE::INDEX_HCNNG, &graph_file[0])
                -> search(weavess::TYPE::SEARCH_ENTRY_KDT, weavess::TYPE::ROUTER_GUIDE, weavess::TYPE::L_SEARCH_SET_RECALL);
        builder -> peak_memory_footprint();
    }else if (parameters.get<std::string>("exc_type") == "info") {
        builder -> load(&base_path[0], &query_path[0], &ground_path[0], parameters)
                -> load_graph(weavess::TYPE::INDEX_HCNNG, &graph_file[0])
                // -> save_graph(weavess::TYPE::INDEX_HCNNG, &graph_file[0])
                -> print_index_info(weavess::TYPE::INDEX_HCNNG);
    }else {
        std::cout << "exc_type input error!" << std::endl;
    }
}

void SPTAG_KDT(weavess::Parameters &parameters) {

    const unsigned num_threads = parameters.get<unsigned>("n_threads");
    std::string base_path = parameters.get<std::string>("base_path");
    std::string query_path = parameters.get<std::string>("query_path");
    std::string ground_path = parameters.get<std::string>("ground_path");
    std::string graph_file = parameters.get<std::string>("graph_file");
    auto *builder = new weavess::IndexBuilder(num_threads);

    if (parameters.get<std::string>("exc_type") == "build") {   // build
        builder -> load(&base_path[0], &query_path[0], &ground_path[0], parameters)
                -> init(weavess::INIT_SPTAG_KDT);
        std::cout << "Init cost: " << builder->GetBuildTime().count() << std::endl;
        builder -> refine(weavess::REFINE_SPTAG_KDT, false)
                -> save_graph(weavess::TYPE::INDEX_SPTAG_KDT, &graph_file[0]);
        std::cout << "Build cost: " << builder->GetBuildTime().count() << std::endl;
    }else if (parameters.get<std::string>("exc_type") == "search") {    // search
        builder -> load(&base_path[0], &query_path[0], &ground_path[0], parameters)
                -> load_graph(weavess::TYPE::INDEX_SPTAG_KDT, &graph_file[0])
                -> search(weavess::TYPE::SEARCH_ENTRY_NONE, weavess::TYPE::ROUTER_SPTAG_KDT, weavess::TYPE::L_SEARCH_SET_RECALL);
        builder -> peak_memory_footprint();
    }else if (parameters.get<std::string>("exc_type") == "info") {
        builder -> load(&base_path[0], &query_path[0], &ground_path[0], parameters)
                -> load_graph(weavess::TYPE::INDEX_SPTAG_KDT, &graph_file[0])
                -> print_index_info(weavess::TYPE::INDEX_SPTAG_KDT);
    }else {
        std::cout << "exc_type input error!" << std::endl;
    }
}

void SPTAG_BKT(weavess::Parameters &parameters) {

    const unsigned num_threads = parameters.get<unsigned>("n_threads");
    std::string base_path = parameters.get<std::string>("base_path");
    std::string query_path = parameters.get<std::string>("query_path");
    std::string ground_path = parameters.get<std::string>("ground_path");
    std::string graph_file = parameters.get<std::string>("graph_file");
    auto *builder = new weavess::IndexBuilder(num_threads);

    if (parameters.get<std::string>("exc_type") == "build") {   // build
        builder -> load(&base_path[0], &query_path[0], &ground_path[0], parameters)
                -> init(weavess::INIT_SPTAG_BKT);
        std::cout << "Init cost: " << builder->GetBuildTime().count() << std::endl;
        builder -> refine(weavess::REFINE_SPTAG_BKT, false)
                -> save_graph(weavess::TYPE::INDEX_SPTAG_BKT, &graph_file[0]);
        std::cout << "Build cost: " << builder->GetBuildTime().count() << std::endl;
    }else if (parameters.get<std::string>("exc_type") == "search") {    // search
        builder -> load(&base_path[0], &query_path[0], &ground_path[0], parameters)
                -> load_graph(weavess::TYPE::INDEX_SPTAG_BKT, &graph_file[0])
                -> search(weavess::TYPE::SEARCH_ENTRY_NONE, weavess::TYPE::ROUTER_SPTAG_BKT, weavess::TYPE::L_SEARCH_SET_RECALL);
        builder -> peak_memory_footprint();
    }else if (parameters.get<std::string>("exc_type") == "info") {
        builder -> load(&base_path[0], &query_path[0], &ground_path[0], parameters)
                -> load_graph(weavess::TYPE::INDEX_SPTAG_BKT, &graph_file[0])
                -> print_index_info(weavess::TYPE::INDEX_SPTAG_BKT);
    }else {
        std::cout << "exc_type input error!" << std::endl;
    }
}

void KDRG(weavess::Parameters &parameters) {

    const unsigned num_threads = parameters.get<unsigned>("n_threads");
    std::string base_path = parameters.get<std::string>("base_path");
    std::string query_path = parameters.get<std::string>("query_path");
    std::string ground_path = parameters.get<std::string>("ground_path");
    std::string graph_file = parameters.get<std::string>("graph_file");
    auto *builder = new weavess::IndexBuilder(num_threads);

    if (parameters.get<std::string>("exc_type") == "build") {   // build
        builder -> load(&base_path[0], &query_path[0], &ground_path[0], parameters)
                -> init(weavess::INIT_KNNG, true);
        std::cout << "Init cost: " << builder->GetBuildTime().count() << std::endl;
        builder -> refine(weavess::REFINE_KDRG, true)
                -> save_graph(weavess::TYPE::INDEX_KDRG, &graph_file[0]);
        std::cout << "Build cost: " << builder->GetBuildTime().count() << std::endl;
    }else if (parameters.get<std::string>("exc_type") == "search") {    // search
        builder -> load(&base_path[0], &query_path[0], &ground_path[0], parameters)
                -> load_graph(weavess::TYPE::INDEX_KDRG, &graph_file[0])
                -> search(weavess::SEARCH_ENTRY_RAND, weavess::ROUTER_GREEDY, weavess::TYPE::L_SEARCH_ASCEND);
        builder -> peak_memory_footprint();
    }else if (parameters.get<std::string>("exc_type") == "info") {
        builder -> load(&base_path[0], &query_path[0], &ground_path[0], parameters)
                -> load_graph(weavess::TYPE::INDEX_KDRG, &graph_file[0])
                -> print_index_info(weavess::TYPE::INDEX_KDRG);
    }else {
        std::cout << "exc_type input error!" << std::endl;
    }
}


int main(int argc, char** argv) {
    if (argc < 4 || argc > 5) {
        std::cout << "./main algorithm dataset exc_type [L_search]"
                << std::endl;
        exit(-1);
    }
    weavess::Parameters parameters;
    std::string dataset_root = R"(/Users/wmz/Documents/Postgraduate/Code/dataset/)";
    std::string index_path = R"(/)";
    parameters.set<std::string>("dataset_root", dataset_root);
    parameters.set<std::string>("index_path", index_path);
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
    std::string exact_knng_path("/Users/wmz/Documents/Postgraduate/Code/WEAVESS/build/algo/knn/siftsmall_50nn.graph");
    parameters.set<std::string>("graph_file", graph_file);
    parameters.set<std::string>("exact_knng_path", exact_knng_path);
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
    }else if (alg == "nsw") {
        NSW(parameters);
    }else if (alg == "hcnng") {
        HCNNG(parameters);
    }else if (alg == "ieh") {
        IEH(parameters);
    }else if (alg == "hnsw") {
        HNSW(parameters);
    }else if (alg == "panng") {
        PANNG(parameters);
    }else if (alg == "onng") {
        ONNG(parameters);
    }else if (alg == "sptag_kdt") {
        SPTAG_KDT(parameters);
    }else if (alg == "sptag_bkt") {
        SPTAG_BKT(parameters);
    }else if (alg == "kdrg") {
        KDRG(parameters);
    }else {
        std::cout << "alg input error!\n";
        exit(-1);
    }
    return 0;
}
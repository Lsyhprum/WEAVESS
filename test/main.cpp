#include <weavess/builder.h>
#include <iostream>


void KGraph(std::string base_path, std::string query_path, std::string ground_path) {
    std::string graph_file = R"(kgraph.graph)";

    weavess::Parameters parameters;
    parameters.set<unsigned>("K", 25);
    parameters.set<unsigned>("L", 50);
    parameters.set<unsigned>("ITER", 8);
    parameters.set<unsigned>("S", 10);
    parameters.set<unsigned>("R", 100);

    auto *builder = new weavess::IndexBuilder();
    builder -> load(&base_path[0], &query_path[0], &ground_path[0], parameters)
            -> init(weavess::TYPE::INIT_RANDOM, true)
            -> refine(weavess::TYPE::REFINE_NN_DESCENT, true)
            -> search(weavess::TYPE::SEARCH_ENTRY_RAND, weavess::TYPE::ROUTER_GREEDY);

    std::cout << "Time cost: " << builder->GetBuildTime().count() << std::endl;
}

void FANNG(std::string base_path, std::string query_path, std::string ground_path, std::string graph_file, unsigned L, unsigned R_refine) {

    weavess::Parameters parameters;
    parameters.set<unsigned>("L", L);
    parameters.set<unsigned>("R_refine", R_refine);

    auto *builder = new weavess::IndexBuilder();

    // build
    builder -> load(&base_path[0], &query_path[0], &ground_path[0], parameters)
            -> init(weavess::TYPE::INIT_FANNG);
    std::cout << "Init cost: " << builder->GetBuildTime().count() << std::endl;
    builder -> refine(weavess::TYPE::REFINE_FANNG, false)
            -> save_graph(weavess::TYPE::INDEX_FANNG, &graph_file[0]);
    std::cout << "Build cost: " << builder->GetBuildTime().count() << std::endl;

    // seach
    // builder -> load(&base_path[0], &query_path[0], &ground_path[0], parameters)
    //         -> load_graph(weavess::TYPE::INDEX_FANNG, &graph_file[0])
    //         -> search(weavess::TYPE::SEARCH_ENTRY_RAND, weavess::TYPE::ROUTER_BACKTRACK);

}

void NSG(std::string base_path, std::string query_path, std::string ground_path) {
    std::string graph_file = R"(nsg.graph)";

    weavess::Parameters parameters;
    parameters.set<unsigned>("K", 400);
    parameters.set<unsigned>("L", 400);
    parameters.set<unsigned>("ITER", 10);
    parameters.set<unsigned>("S", 10);
    parameters.set<unsigned>("R", 100);

    parameters.set<unsigned>("L_refine", 40);
    parameters.set<unsigned>("R_refine", 50);
    parameters.set<unsigned>("C_refine", 500);

    auto *builder = new weavess::IndexBuilder();
    builder -> load(&base_path[0], &query_path[0], &ground_path[0], parameters)
            -> init(weavess::INIT_RANDOM)
            -> refine(weavess::REFINE_NN_DESCENT, true)
            -> refine(weavess::REFINE_NSG, true);
            // -> search(weavess::TYPE::SEARCH_ENTRY_CENTROID, weavess::TYPE::ROUTER_GREEDY);

    std::cout << "Time cost: " << builder->GetBuildTime().count() << std::endl;
}

void SSG(std::string base_path, std::string query_path, std::string ground_path) {
    std::string graph_file = R"(ssg.graph)";

    weavess::Parameters parameters;
    parameters.set<unsigned>("K", 200);
    parameters.set<unsigned>("L", 200);
    parameters.set<unsigned>("ITER", 8);
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
    parameters.set<unsigned>("K", 50);
    parameters.set<unsigned>("L", 60);
    parameters.set<unsigned>("ITER", 8);
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

    const unsigned R = 70;
    const unsigned L = 125;

    parameters.set<unsigned>("L", R);
    parameters.set<unsigned>("L_refine", L);
    parameters.set<unsigned>("R_refine", R);

    auto *builder = new weavess::IndexBuilder();
    builder -> load(&base_path[0], &query_path[0], &ground_path[0], parameters)
            -> init(weavess::INIT_RAND)
            -> refine(weavess::REFINE_VAMANA, true)
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
    parameters.set<unsigned>("ITER", 12);
    parameters.set<unsigned>("S", 10);
    parameters.set<unsigned>("R", 100);

    auto *builder = new weavess::IndexBuilder();
    builder -> load(&base_path[0], &query_path[0], &ground_path[0], parameters)
            -> init(weavess::INIT_KDT)
            -> refine(weavess::REFINE_EFANNA, true);
            // -> search(weavess::SEARCH_ENTRY_KDT, weavess::ROUTER_GREEDY);

    std::cout << "Time cost: " << builder->GetBuildTime().count() << std::endl;
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

    auto *builder = new weavess::IndexBuilder();
    builder -> load(&base_path[0], &query_path[0], &ground_path[0], parameters) // useless
            -> init(weavess::INIT_IEH)
            -> search(weavess::SEARCH_ENTRY_HASH, weavess::ROUTER_IEH);

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
    parameters.set<unsigned>("max_m", 5);
    parameters.set<unsigned>("max_m0", 10);
    parameters.set<unsigned>("ef_construction", 150);
    parameters.set<unsigned>("n_threads", 1);
    parameters.set<int>("mult", -1);

    auto *builder = new weavess::IndexBuilder();
    builder -> load(&base_path[0], &query_path[0], &ground_path[0], parameters)
            -> init(weavess::INIT_HNSW)
            -> search(weavess::SEARCH_ENTRY_NONE, weavess::ROUTER_HNSW);

    std::cout << "Time cost: " << builder->GetBuildTime().count() << std::endl;
}

void PANNG(std::string base_path, std::string query_path, std::string ground_path) {
    weavess::Parameters parameters;
    parameters.set<unsigned>("NN", 10);          // K
    parameters.set<unsigned>("ef_construction", 50);        //L
    parameters.set<unsigned>("n_threads_", 1);
    //parameters.set<unsigned>("batchSizeForCreation", 200);

    auto *builder = new weavess::IndexBuilder();
    builder -> load(&base_path[0], &query_path[0], &ground_path[0], parameters)
            -> init(weavess::INIT_ANNG)
            -> refine(weavess::REFINE_PANNG, false)
            -> search(weavess::SEARCH_ENTRY_VPT, weavess::ROUTER_NGT);

    std::cout << "Time cost: " << builder->GetBuildTime().count() << std::endl;
}

void ONNG(std::string base_path, std::string query_path, std::string ground_path) {
    weavess::Parameters parameters;
    parameters.set<unsigned>("edgeSizeForCreation", 10); // 初始化边数阈值
    parameters.set<unsigned>("truncationThreshold", 10);
    parameters.set<unsigned>("edgeSizeForSearch", 10);
    parameters.set<unsigned>("size", 10);

    parameters.set<unsigned>("numOfOutgoingEdges", 10);
    parameters.set<unsigned>("numOfIncomingEdges", 100);
    parameters.set<unsigned>("numOfQueries", 200);
    parameters.set<unsigned>("numOfResultantObjects", 20);

    auto *builder = new weavess::IndexBuilder();
    builder -> load(&base_path[0], &query_path[0], &ground_path[0], parameters)
            -> init(weavess::INIT_ANNG)
            -> refine(weavess::REFINE_ONNG2, false)
            -> refine(weavess::REFINE_PANNG, false);
            //-> search(weavess::SEARCH_ENTRY_NONE, weavess::ROUTER_NGT);

    std::cout << "Time cost: " << builder->GetBuildTime().count() << std::endl;
}

void HCNNG(std::string base_path, std::string query_path, std::string ground_path) {
    weavess::Parameters parameters;
    parameters.set<unsigned>("minsize_cl", 1000);
    parameters.set<unsigned>("num_cl", 20);

    parameters.set<unsigned>("nTrees", 10);
    parameters.set<unsigned>("mLevel", 4);
    parameters.set<unsigned>("K", 10);

    auto *builder = new weavess::IndexBuilder();
    builder -> load(&base_path[0], &query_path[0], &ground_path[0], parameters)
            -> init(weavess::INIT_HCNNG)
            -> search(weavess::SEARCH_ENTRY_KDT, weavess::ROUTER_GREEDY);

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

    auto *builder = new weavess::IndexBuilder();
    builder -> load(&base_path[0], &query_path[0], &ground_path[0], parameters)
            -> init(weavess::INIT_SPTAG_KDT)
            -> refine(weavess::REFINE_SPTAG_KDT, false)
            -> search(weavess::SEARCH_ENTRY_NONE, weavess::ROUTER_SPTAG_KDT);
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

    auto *builder = new weavess::IndexBuilder();
    builder -> load(&base_path[0], &query_path[0], &ground_path[0], parameters)
            -> init(weavess::INIT_SPTAG_BKT)
            -> refine(weavess::REFINE_SPTAG_BKT, false)
            -> search(weavess::SEARCH_ENTRY_NONE, weavess::ROUTER_SPTAG_BKT);

    std::cout << "Time cost: " << builder->GetBuildTime().count() << std::endl;
}


int main(int argc, char** argv) {
    std::string dataset_root = R"(/Users/wmz/Documents/Postgraduate/Code/dataset/)";
    std::string base_path(dataset_root);
    std::string query_path(dataset_root);
    std::string ground_path(dataset_root);
    std::string alg(argv[1]);
    std::string dataset(argv[2]);
    std::string graph_file(alg + "_" + dataset + ".graph");
    unsigned L, R;
    if (dataset == "siftsmall") {
        base_path.append(R"(siftsmall/siftsmall_base.fvecs)");
        query_path.append(R"(siftsmall/siftsmall_query.fvecs)");
        ground_path.append(R"(siftsmall/siftsmall_groundtruth.ivecs)");
        L = 100, R = 25;
    }else if (dataset == "sift1M") {
        base_path.append(R"(sift1M\sift_base.fvecs)");
        query_path.append(R"(sift1M\sift_query.fvecs)");
        ground_path.append(R"(sift1M\sift_groundtruth.ivecs)");
        L = 110, R = 70;
    }
    // else if (dataset == "gist") {
    //     std::string base_path = dataset_root.append(R"(gist\gist_base.fvecs)");
    //     std::string query_path = dataset_root.append(R"(gist\gist_query.fvecs)");
    //     std::string ground_path = dataset_root.append(R"(gist\gist_groundtruth.ivecs)");
    //     unsigned L = 210, R = 50;
    // }
    // else if (dataset == "glove-100") {
    //     std::string base_path = dataset_root.append(R"(glove-100\glove-100_base.fvecs)");
    //     std::string query_path = dataset_root.append(R"(glove-100\glove-100_query.fvecs)");
    //     std::string ground_path = dataset_root.append(R"(glove-100\glove-100_groundtruth.ivecs)");
    //     unsigned L = 210, R = 70;
    // }
    // else if (dataset == "audio") {
    //     std::string base_path = dataset_root.append(R"(audio\audio_base.fvecs)");
    //     std::string query_path = dataset_root.append(R"(audio\audio_query.fvecs)");
    //     std::string ground_path = dataset_root.append(R"(audio\audio_groundtruth.ivecs)");
    //     unsigned L = 130, R = 50;
    // }
    // else if (dataset == "crawl") {
    //     std::string base_path = dataset_root.append(R"(crawl\crawl_base.fvecs)");
    //     std::string query_path = dataset_root.append(R"(crawl\crawl_query.fvecs)");
    //     std::string ground_path = dataset_root.append(R"(crawl\crawl_groundtruth.ivecs)");
    //     unsigned L = 110, R = 30;
    // }
    // else if (dataset == "msong") {

    // }
    // else if (dataset == "uqv") {

    // }
    // else if (dataset == "enron") {

    // }
    else {
        std::cout << "input dataset error!\n";
        exit(-1);
    }

    // alg
    if (alg == "kgraph") {
        KGraph(base_path, query_path, ground_path);
    }else if (alg == "fanng") {
        FANNG(base_path, query_path, ground_path, graph_file, L, R);
    }else if (alg == "nsg") {
        NSG(base_path, query_path, ground_path);
    }else if (alg == "ssg") {
        SSG(base_path, query_path, ground_path);
    }else if (alg == "efanna") {
        EFANNA(base_path, query_path, ground_path);
    }
    //KGraph(base_path, query_path, ground_path);
    //NSG(base_path, query_path, ground_path);
    //SSG(base_path, query_path, ground_path);
    //DPG(base_path, query_path, ground_path);
    //VAMANA(base_path, query_path, ground_path);
    //EFANNA(base_path, query_path, ground_path);
    //IEH(base_path, query_path, ground_path);
    //NSW(base_path, query_path, ground_path);
    //HNSW(base_path, query_path, ground_path);
    //PANNG(base_path, query_path, ground_path);
    //ONNG(base_path, query_path, ground_path);
    //HCNNG(base_path, query_path, ground_path);
    //SPTAG_KDT(base_path, query_path, ground_path);
    //SPTAG_BKT(base_path, query_path, ground_path);

    return 0;
}
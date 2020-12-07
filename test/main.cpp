#include <weavess/builder.h>
#include <iostream>


void KGraph(std::string base_path, std::string query_path, std::string ground_path, std::string graph_file, unsigned K, unsigned L, unsigned Iter, unsigned S, unsigned R, const unsigned num_threads) {
    // std::string graph_file = R"(kgraph.graph)";

    weavess::Parameters parameters;
    parameters.set<unsigned>("K", K);
    parameters.set<unsigned>("L", L);
    parameters.set<unsigned>("ITER", Iter);
    parameters.set<unsigned>("S", S);
    parameters.set<unsigned>("R", R);

    auto *builder = new weavess::IndexBuilder(num_threads);

    // build
    builder -> load(&base_path[0], &query_path[0], &ground_path[0], parameters)
            -> init(weavess::TYPE::INIT_RANDOM, false);
    std::cout << "Init cost: " << builder->GetBuildTime().count() << std::endl;
    // 注意正式实验要关闭NN-Descent迭代图质量信息输出 index.h文件中
    builder -> refine(weavess::TYPE::REFINE_NN_DESCENT, false)
            -> save_graph(weavess::TYPE::INDEX_KGRAPH, &graph_file[0]);
    std::cout << "Build cost: " << builder->GetBuildTime().count() << std::endl;
    
    // search
    builder -> load(&base_path[0], &query_path[0], &ground_path[0], parameters)
            -> load_graph(weavess::TYPE::INDEX_KGRAPH, &graph_file[0])
            -> search(weavess::TYPE::SEARCH_ENTRY_RAND, weavess::TYPE::ROUTER_GREEDY, true);

}

void FANNG(std::string base_path, std::string query_path, std::string ground_path, std::string graph_file, unsigned L, unsigned R_refine, const unsigned num_threads) {

    weavess::Parameters parameters;
    parameters.set<unsigned>("L", L);
    parameters.set<unsigned>("R_refine", R_refine);

    auto *builder = new weavess::IndexBuilder(num_threads);

    // build
    // builder -> load(&base_path[0], &query_path[0], &ground_path[0], parameters)
    //         -> init(weavess::TYPE::INIT_FANNG);
    // std::cout << "Init cost: " << builder->GetBuildTime().count() << std::endl;
    // builder -> refine(weavess::TYPE::REFINE_FANNG, false)
    //         -> save_graph(weavess::TYPE::INDEX_FANNG, &graph_file[0]);
    // std::cout << "Build cost: " << builder->GetBuildTime().count() << std::endl;

    // seach
    builder -> load(&base_path[0], &query_path[0], &ground_path[0], parameters)
            -> load_graph(weavess::TYPE::INDEX_FANNG, &graph_file[0])
            -> search(weavess::TYPE::SEARCH_ENTRY_RAND, weavess::TYPE::ROUTER_BACKTRACK, true);

}

void NSG(std::string base_path, std::string query_path, std::string ground_path, std::string graph_file, unsigned K, unsigned L, unsigned Iter, unsigned S, unsigned R, unsigned L_refine, unsigned R_refine, unsigned C, const unsigned num_threads) {
    // std::string graph_file = R"(nsg.graph)";

    weavess::Parameters parameters;
    parameters.set<unsigned>("K", K);
    parameters.set<unsigned>("L", L);
    parameters.set<unsigned>("ITER", Iter);
    parameters.set<unsigned>("S", S);
    parameters.set<unsigned>("R", R);

    parameters.set<unsigned>("L_refine", L_refine);
    parameters.set<unsigned>("R_refine", R_refine);
    parameters.set<unsigned>("C_refine", C);

    auto *builder = new weavess::IndexBuilder(num_threads);

    // build
    builder -> load(&base_path[0], &query_path[0], &ground_path[0], parameters)
            -> init(weavess::INIT_RANDOM)
            -> refine(weavess::REFINE_NN_DESCENT, false);
    std::cout << "Init cost: " << builder->GetBuildTime().count() << std::endl;
    builder -> refine(weavess::REFINE_NSG, false)
            -> save_graph(weavess::TYPE::INDEX_NSG, &graph_file[0]);
    std::cout << "Build cost: " << builder->GetBuildTime().count() << std::endl;

    // search
    builder -> load(&base_path[0], &query_path[0], &ground_path[0], parameters)
            -> load_graph(weavess::TYPE::INDEX_NSG, &graph_file[0])
            -> search(weavess::TYPE::SEARCH_ENTRY_CENTROID, weavess::TYPE::ROUTER_GREEDY, true);

}

void SSG(std::string base_path, std::string query_path, std::string ground_path, std::string graph_file, unsigned K, unsigned L, unsigned Iter, unsigned S, unsigned R, unsigned L_refine, unsigned R_refine, const unsigned num_threads) {
    // std::string graph_file = R"(ssg.graph)";

    weavess::Parameters parameters;
    parameters.set<unsigned>("K", K);
    parameters.set<unsigned>("L", L);
    parameters.set<unsigned>("ITER", Iter);
    parameters.set<unsigned>("S", S);
    parameters.set<unsigned>("R", R);

    parameters.set<unsigned>("L_refine", L_refine);
    parameters.set<unsigned>("R_refine", R_refine);
    parameters.set<float>("A", 60);
    parameters.set<unsigned>("n_try", 10);

    auto *builder = new weavess::IndexBuilder(num_threads);
    // build
    builder -> load(&base_path[0], &query_path[0], &ground_path[0], parameters)
            -> init(weavess::INIT_RANDOM)
            -> refine(weavess::REFINE_NN_DESCENT, false);
    std::cout << "Init cost: " << builder->GetBuildTime().count() << std::endl;
    builder -> refine(weavess::REFINE_SSG, false)
            -> save_graph(weavess::TYPE::INDEX_SSG, &graph_file[0]);
    std::cout << "Build cost: " << builder->GetBuildTime().count() << std::endl;

    // search
    builder -> load(&base_path[0], &query_path[0], &ground_path[0], parameters)
            -> load_graph(weavess::TYPE::INDEX_SSG, &graph_file[0])
            -> search(weavess::SEARCH_ENTRY_SUB_CENTROID, weavess::ROUTER_GREEDY, true);
}

void DPG(std::string base_path, std::string query_path, std::string ground_path) {
    std::string graph_file = R"(dpg.knng)";

    weavess::Parameters parameters;
    parameters.set<unsigned>("K", 50);
    parameters.set<unsigned>("L", 60);
    parameters.set<unsigned>("ITER", 8);
    parameters.set<unsigned>("S", 10);
    parameters.set<unsigned>("R", 100);

    auto *builder = new weavess::IndexBuilder(8);
    builder -> load(&base_path[0], &query_path[0], &ground_path[0], parameters)
            -> init(weavess::INIT_NN_DESCENT)
            -> refine(weavess::REFINE_NN_DESCENT, true)
            -> refine(weavess::REFINE_DPG, true)
            -> search(weavess::TYPE::SEARCH_ENTRY_RAND, weavess::TYPE::ROUTER_GREEDY, false);

    std::cout << "Time cost: " << builder->GetBuildTime().count() << std::endl;
}

void VAMANA(std::string base_path, std::string query_path, std::string ground_path) {
    weavess::Parameters parameters;

    const unsigned R = 70;
    const unsigned L = 125;

    parameters.set<unsigned>("L", R);
    parameters.set<unsigned>("L_refine", L);
    parameters.set<unsigned>("R_refine", R);

    auto *builder = new weavess::IndexBuilder(8);
    builder -> load(&base_path[0], &query_path[0], &ground_path[0], parameters)
            -> init(weavess::INIT_RAND)
            -> refine(weavess::REFINE_VAMANA, true)
            -> refine(weavess::REFINE_VAMANA, true)
            -> search(weavess::TYPE::SEARCH_ENTRY_CENTROID, weavess::TYPE::ROUTER_GREEDY, false);

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

    auto *builder = new weavess::IndexBuilder(8);
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

    auto *builder = new weavess::IndexBuilder(8);
    builder -> load(&base_path[0], &query_path[0], &ground_path[0], parameters) // useless
            -> init(weavess::INIT_IEH)
            -> search(weavess::SEARCH_ENTRY_HASH, weavess::ROUTER_IEH, false);

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
            -> search(weavess::SEARCH_ENTRY_NONE, weavess::ROUTER_NSW, false);

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
            -> search(weavess::SEARCH_ENTRY_NONE, weavess::ROUTER_HNSW, false);

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
            -> search(weavess::SEARCH_ENTRY_VPT, weavess::ROUTER_NGT, false);

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
            -> search(weavess::SEARCH_ENTRY_VPT, weavess::ROUTER_NGT, true);

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
            -> search(weavess::SEARCH_ENTRY_KDT, weavess::ROUTER_GREEDY, false);

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
            -> search(weavess::SEARCH_ENTRY_NONE, weavess::ROUTER_SPTAG_KDT, false);
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
            -> search(weavess::SEARCH_ENTRY_NONE, weavess::ROUTER_SPTAG_BKT, false);

    std::cout << "Time cost: " << builder->GetBuildTime().count() << std::endl;
}


int main(int argc, char** argv) {
    std::string dataset_root = R"(/Users/wmz/Documents/Postgraduate/Code/dataset/)";
    std::string base_path(dataset_root);
    std::string query_path(dataset_root);
    std::string ground_path(dataset_root);
    const unsigned num_threads = 4;
    std::string alg(argv[1]);
    std::string dataset(argv[2]);
    std::string graph_file(alg + "_" + dataset + ".graph");
    // unsigned L, R; // fanng
    unsigned K, L, Iter, S, R;  // kgraph ssg nsg dpg
    // unsigned L_refine, R_refine; // ssg
    unsigned L_refine, R_refine, C; // nsg

    // dataset
    {
        if (dataset == "siftsmall") {
            base_path.append(R"(siftsmall/siftsmall_base.fvecs)");
            query_path.append(R"(siftsmall/siftsmall_query.fvecs)");
            ground_path.append(R"(siftsmall/siftsmall_groundtruth.ivecs)");
            // L = 100, R = 25;    // fanng
            // K = 25, L = 50, Iter = 6, S = 10, R = 100;  // kgraph
            // K = 25, L = 50, Iter = 6, S = 10, R = 100, L_refine = 100, R_refine = 50;   // ssg
            K = 25, L = 50, Iter = 6, S = 10, R = 100, L_refine = 100, R_refine = 50, C = 500;   // nsg

        }else if (dataset == "sift1M") {
            base_path.append(R"(sift1M/sift_base.fvecs)");
            query_path.append(R"(sift1M/sift_query.fvecs)");
            ground_path.append(R"(sift1M/sift_groundtruth.ivecs)");
            // L = 110, R = 70;    // fanng
            // K = 90, L = 130, Iter = 12, S = 20, R = 50;  // kgraph
            // K = 400, L = 420, Iter = 12, S = 20, R = 100, L_refine = 50, R_refine = 20;   // ssg
            K = 100, L = 120, Iter = 12, S = 25, R = 300, L_refine = 150, R_refine = 30, C = 400;   // nsg

        }else if (dataset == "gist") {
            base_path.append(R"(gist/gist_base.fvecs)");
            query_path.append(R"(gist/gist_query.fvecs)");
            ground_path.append(R"(gist/gist_groundtruth.ivecs)");
            // L = 210, R = 50;    // fanng
            // K = 100, L = 120, Iter = 12, S = 25, R = 100;  // kgraph
            // K = 300, L = 330, Iter = 12, S = 20, R = 200, L_refine = 200, R_refine = 40;   // ssg
            K = 400, L = 430, Iter = 12, S = 10, R = 200, L_refine = 500, R_refine = 20, C = 400;   // nsg

        }else if (dataset == "glove-100") {
            base_path.append(R"(glove-100/glove-100_base.fvecs)");
            query_path.append(R"(glove-100/glove-100_query.fvecs)");
            ground_path.append(R"(glove-100/glove-100_groundtruth.ivecs)");
            // L = 210, R = 70;    // fanng
            // K = 100, L = 150, Iter = 12, S = 35, R = 150;  // kgraph
            // K = 300, L = 320, Iter = 12, S = 10, R = 200, L_refine = 150, R_refine = 30;   // ssg
            K = 400, L = 420, Iter = 12, S = 20, R = 300, L_refine = 150, R_refine = 90, C = 600;   // nsg

        }else if (dataset == "audio") {
            base_path.append(R"(audio/audio_base.fvecs)");
            query_path.append(R"(audio/audio_query.fvecs)");
            ground_path.append(R"(audio/audio_groundtruth.ivecs)");
            // L = 130, R = 50;    // fanng
            // K = 40, L = 60, Iter = 5, S = 20, R = 100;  // kgraph
            // K = 400, L = 400, Iter = 5, S = 25, R = 200, L_refine = 50, R_refine = 20;   // ssg
            K = 200, L = 230, Iter = 5, S = 10, R = 100, L_refine = 200, R_refine = 30, C = 600;   // nsg

        }else if (dataset == "crawl") {
            base_path.append(R"(crawl/crawl_base.fvecs)");
            query_path.append(R"(crawl/crawl_query.fvecs)");
            ground_path.append(R"(crawl/crawl_groundtruth.ivecs)");
            // L = 110, R = 30;    // fanng
            // K = 80, L = 100, Iter = 12, S = 10, R = 150;  // kgraph
            // K = 100, L = 100, Iter = 12, S = 10, R = 100, L_refine = 50, R_refine = 60;   // ssg
            K = 400, L = 430, Iter = 12, S = 15, R = 300, L_refine = 250, R_refine = 40, C = 600;   // nsg

        }else if (dataset == "msong") {
            base_path.append(R"(msong/msong_base.fvecs)");
            query_path.append(R"(msong/msong_query.fvecs)");
            ground_path.append(R"(msong/msong_groundtruth.ivecs)");
            // L = 150, R = 10;    // fanng
            // K = 100, L = 140, Iter = 12, S = 15, R = 150;  // kgraph
            // K = 400, L = 420, Iter = 12, S = 25, R = 300, L_refine = 100, R_refine = 70;   // ssg
            K = 300, L = 310, Iter = 12, S = 25, R = 300, L_refine = 350, R_refine = 20, C = 500;   // nsg

        }else if (dataset == "uqv") {
            base_path.append(R"(uqv/uqv_base.fvecs)");
            query_path.append(R"(uqv/uqv_query.fvecs)");
            ground_path.append(R"(uqv/uqv_groundtruth.ivecs)");
            // L = 250, R = 90;    // fanng
            // K = 40, L = 80, Iter = 6, S = 25, R = 100;  // kgraph
            // K = 400, L = 420, Iter = 6, S = 20, R = 300, L_refine = 250, R_refine = 20;   // ssg
            K = 300, L = 320, Iter = 6, S = 15, R = 200, L_refine = 350, R_refine = 30, C = 400;   // nsg

        }else if (dataset == "enron") {
            base_path.append(R"(enron/enron_base.fvecs)");
            query_path.append(R"(enron/enron_query.fvecs)");
            ground_path.append(R"(enron/enron_groundtruth.ivecs)");
            // L = 130, R = 110;    // fanng
            // K = 50, L = 80, Iter = 7, S = 15, R = 100;  // kgraph
            // K = 100, L = 110, Iter = 7, S = 20, R = 300, L_refine = 300, R_refine = 30;   // ssg
            K = 200, L = 200, Iter = 7, S = 25, R = 200, L_refine = 150, R_refine = 60, C = 600;   // nsg

        }else if (dataset == "mnist") {
            base_path.append(R"(mnist/mnist_base.fvecs)");
            query_path.append(R"(mnist/mnist_query.fvecs)");
            ground_path.append(R"(mnist/mnist_groundtruth.ivecs)");
            // L = 100, R = 25;    // fanng
            // K = 25, L = 50, Iter = 8, S = 10, R = 100;  // kgraph
            // K = 25, L = 50, Iter = 8, S = 10, R = 100, L_refine = 100, R_refine = 50;   // ssg
            K = 25, L = 50, Iter = 8, S = 10, R = 100, L_refine = 100, R_refine = 50, C = 600;   // nsg

        }else if (dataset == "c_1") {
            base_path.append(R"(c_1/random_base_n100000_d32_c1_s5.fvecs)");
            query_path.append(R"(c_1/random_query_n1000_d32_c1_s5.fvecs)");
            ground_path.append(R"(c_1/random_ground_truth_n1000_d32_c1_s5.ivecs)");
            L = 30, R = 30;    // fanng
            // K = 100, L = 110, Iter = 8, S = 25, R = 150;  // kgraph
            // K = 25, L = 50, Iter = 8, S = 10, R = 100, L_refine = 100, R_refine = 50;   // ssg unset
            // K = 25, L = 50, Iter = 8, S = 10, R = 100, L_refine = 100, R_refine = 50, C = 600;   // nsg unset

        }else if (dataset == "c_10") {
            base_path.append(R"(c_10/random_base_n100000_d32_c10_s5.fvecs)");
            query_path.append(R"(c_10/random_query_n1000_d32_c10_s5.fvecs)");
            ground_path.append(R"(c_10/random_ground_truth_n1000_d32_c10_s5.ivecs)");
            L = 90, R = 10;    // fanng
            // K = 100, L = 120, Iter = 8, S = 25, R = 50;  // kgraph
            // K = 25, L = 50, Iter = 8, S = 10, R = 100, L_refine = 100, R_refine = 50;   // ssg
            // K = 25, L = 50, Iter = 8, S = 10, R = 100, L_refine = 100, R_refine = 50, C = 600;   // nsg

        }else if (dataset == "c_100") {
            base_path.append(R"(c_100/random_base_n100000_d32_c100_s5.fvecs)");
            query_path.append(R"(c_100/random_query_n1000_d32_c100_s5.fvecs)");
            ground_path.append(R"(c_100/random_ground_truth_n1000_d32_c100_s5.ivecs)");
            L = 150, R = 30;    // fanng
            // K = 80, L = 130, Iter = 8, S = 35, R = 150;  // kgraph
            // K = 25, L = 50, Iter = 8, S = 10, R = 100, L_refine = 100, R_refine = 50;   // ssg
            // K = 25, L = 50, Iter = 8, S = 10, R = 100, L_refine = 100, R_refine = 50, C = 600;   // nsg

        }else if (dataset == "d_8") {
            base_path.append(R"(d_8/random_base_n100000_d8_c10_s5.fvecs)");
            query_path.append(R"(d_8/random_query_n1000_d8_c10_s5.fvecs)");
            ground_path.append(R"(d_8/random_ground_truth_n1000_d8_c10_s5.ivecs)");
            L = 210, R = 110;    // fanng
            // K = 50, L = 70, Iter = 8, S = 10, R = 150;  // kgraph
            // K = 25, L = 50, Iter = 8, S = 10, R = 100, L_refine = 100, R_refine = 50;   // ssg
            // K = 25, L = 50, Iter = 8, S = 10, R = 100, L_refine = 100, R_refine = 50, C = 600;   // nsg

        }else if (dataset == "d_128") {
            base_path.append(R"(d_128/random_base_n100000_d128_c10_s5.fvecs)");
            query_path.append(R"(d_128/random_query_n1000_d128_c10_s5.fvecs)");
            ground_path.append(R"(d_128/random_ground_truth_n1000_d128_c10_s5.ivecs)");
            L = 210, R = 70;    // fanng
            // K = 90, L = 90, Iter = 8, S = 30, R = 50;  // kgraph
            // K = 25, L = 50, Iter = 8, S = 10, R = 100, L_refine = 100, R_refine = 50;   // ssg
            // K = 25, L = 50, Iter = 8, S = 10, R = 100, L_refine = 100, R_refine = 50, C = 600;   // nsg

        }else if (dataset == "n_10000") {
            base_path.append(R"(n_10000/random_base_n10000_d32_c10_s5.fvecs)");
            query_path.append(R"(n_10000/random_query_n100_d32_c10_s5.fvecs)");
            ground_path.append(R"(n_10000/random_ground_truth_n100_d32_c10_s5.ivecs)");
            L = 110, R = 90;    // fanng
            // K = 100, L = 140, Iter = 8, S = 30, R = 100;  // kgraph
            // K = 25, L = 50, Iter = 8, S = 10, R = 100, L_refine = 100, R_refine = 50;   // ssg
            // K = 25, L = 50, Iter = 8, S = 10, R = 100, L_refine = 100, R_refine = 50, C = 600;   // nsg

        }else if (dataset == "n_1000000") {
            base_path.append(R"(n_1000000/random_base_n1000000_d32_c10_s5.fvecs)");
            query_path.append(R"(n_1000000/random_query_n10000_d32_c10_s5.fvecs)");
            ground_path.append(R"(n_1000000/random_ground_truth_n10000_d32_c10_s5.ivecs)");
            L = 110, R = 90;    // fanng
            // K = 100, L = 130, Iter = 8, S = 20, R = 50;  // kgraph
            // K = 25, L = 50, Iter = 8, S = 10, R = 100, L_refine = 100, R_refine = 50;   // ssg
            // K = 25, L = 50, Iter = 8, S = 10, R = 100, L_refine = 100, R_refine = 50, C = 600;   // nsg

        }else if (dataset == "s_1") {
            base_path.append(R"(s_1/random_base_n100000_d32_c10_s1.fvecs)");
            query_path.append(R"(s_1/random_query_n1000_d32_c10_s1.fvecs)");
            ground_path.append(R"(s_1/random_ground_truth_n1000_d32_c10_s1.ivecs)");
            L = 110, R = 30;    // fanng
            // K = 60, L = 60, Iter = 8, S = 20, R = 150;  // kgraph
            // K = 25, L = 50, Iter = 8, S = 10, R = 100, L_refine = 100, R_refine = 50;   // ssg
            // K = 25, L = 50, Iter = 8, S = 10, R = 100, L_refine = 100, R_refine = 50, C = 600;   // nsg

        }else if (dataset == "s_10") {
            base_path.append(R"(s_10/random_base_n100000_d32_c10_s10.fvecs)");
            query_path.append(R"(s_10/random_query_n1000_d32_c10_s10.fvecs)");
            ground_path.append(R"(s_10/random_ground_truth_n1000_d32_c10_s10.ivecs)");
            L = 250, R = 70;    // fanng
            // K = 80, L = 110, Iter = 8, S = 20, R = 150;  // kgraph
            // K = 25, L = 50, Iter = 8, S = 10, R = 100, L_refine = 100, R_refine = 50;   // ssg
            // K = 25, L = 50, Iter = 8, S = 10, R = 100, L_refine = 100, R_refine = 50, C = 600;   // nsg

        }else {
            std::cout << "input dataset error!\n";
            exit(-1);
        }
    }

    // alg
    if (alg == "kgraph") {
        KGraph(base_path, query_path, ground_path, graph_file, K, L, Iter, S, R, num_threads);
    }else if (alg == "fanng") {
        FANNG(base_path, query_path, ground_path, graph_file, L, R, num_threads);
    }else if (alg == "nsg") {
        NSG(base_path, query_path, ground_path, graph_file, K, L, Iter, S, R, L_refine, R_refine, C, num_threads);
    }else if (alg == "ssg") {
        SSG(base_path, query_path, ground_path, graph_file, K, L, Iter, S, R, L_refine, R_refine, num_threads);
    }else if (alg == "efanna") {
        EFANNA(base_path, query_path, ground_path);
    }else if (alg == "dpg") {
        DPG(base_path, query_path, ground_path);
    }else if (alg == "ieh") {
        IEH(base_path, query_path, ground_path);
    }else if (alg == "vamana") {
        VAMANA(base_path, query_path, ground_path);
    }else if (alg == "nsw") {
        NSW(base_path, query_path, ground_path);
    }else if (alg == "hnsw") {
        HNSW(base_path, query_path, ground_path);
    }else if (alg == "panng") {
        PANNG(base_path, query_path, ground_path);
    }else if (alg == "onng") {
        ONNG(base_path, query_path, ground_path);
    }else if (alg == "hcnng") {
        HCNNG(base_path, query_path, ground_path);
    }else if (alg == "sptag_kdt") {
        SPTAG_KDT(base_path, query_path, ground_path);
    }else if (alg == "sptag_bkt") {
        SPTAG_BKT(base_path, query_path, ground_path);
    }else {
        std::cout << "alg input error!\n";
        exit(-1);
    }

    return 0;
}
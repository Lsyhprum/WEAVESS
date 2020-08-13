//
// Created by Murph on 2020/8/12.
//
#include <vector>
#include <algorithm>
#include <weavess/util.h>
#include <weavess/index_random.h>
#include <weavess/index_base.h>


int main(int argc, char **argv) {

//    if (argc != 8) {
//        std::cout << argv[0] << "K L iter S R" << std::endl;
//        exit(-1);
//    }
//
//    unsigned K = (unsigned) atoi(argv[1]);
//    unsigned L = (unsigned) atoi(argv[2]);
//    unsigned iter = (unsigned) atoi(argv[3]);
//    unsigned S = (unsigned) atoi(argv[4]);
//    unsigned R = (unsigned) atoi(argv[5]);

    unsigned K = 100;
    unsigned L = 100;
    unsigned iter = 10;
    unsigned S = 15;
    unsigned R = 100;

    std::string base_path = "F:\\ANNS\\dataset\\mnist\\mnist_base.fvecs";
    std::string query_path = "F:\\ANNS\\dataset\\mnist\\mnist_query.fvecs";
    std::string ground_path = "F:\\ANNS\\dataset\\mnist\\mnist_groundtruth.ivecs";

    float *data_base = nullptr;
    float *data_query = nullptr;
    unsigned *data_ground = nullptr;

    unsigned base_num, base_dim;
    unsigned query_num, query_dim;
    unsigned ground_num, ground_dim;

    weavess::load_data(&base_path[0], data_base, base_num, base_dim);
    data_base = weavess::data_align(data_base, base_num, base_dim);
    weavess::load_data(&query_path[0], data_query, query_num, query_dim);
    data_base = weavess::data_align(data_query, query_num, query_dim);
    weavess::load_result_data(&ground_path[0], data_ground, ground_num, ground_dim);

    weavess::IndexRandom init_index(base_dim, base_num);
    weavess::IndexGraph index(base_dim, base_num, weavess::L2, (weavess::Index*)(&init_index));

    weavess::Parameters paras;
    paras.Set<unsigned>("K", K);
    paras.Set<unsigned>("L", L);
    paras.Set<unsigned>("iter", iter);
    paras.Set<unsigned>("S", S);
    paras.Set<unsigned>("R", R);

    auto s = std::chrono::high_resolution_clock::now();
    index.Build(base_num, data_base, paras);
    auto e = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = e-s;
    std::cout <<"Time cost: "<< diff.count() << "\n";
}


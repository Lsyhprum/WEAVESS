#include "weavess/exp_data.h"
#include "weavess/parameters.h"
#include <iostream>
#include <queue>
#include <omp.h>
#include <algorithm>
#include <fstream>
#include <vector>
#include <chrono>
#include <unordered_set>
#include <cassert>
#ifdef __APPLE__
#else
#include <malloc.h>
#endif

struct cmp{
    template<typename T, typename U>
    bool operator()(T const& left, U const &right) {
        if (left.first > right.first) return true;
        return false;
    }
};

template<typename T>
T compare(const T *a, const T *b, unsigned length) {
    T result = 0;

    float diff0, diff1, diff2, diff3;
    const T *last = a + length;
    const T *unroll_group = last - 3;

    /* Process 4 items with each loop for efficiency. */
    while (a < unroll_group) {
        diff0 = a[0] - b[0];
        diff1 = a[1] - b[1];
        diff2 = a[2] - b[2];
        diff3 = a[3] - b[3];
        result += diff0 * diff0 + diff1 * diff1 + diff2 * diff2 + diff3 * diff3;
        a += 4;
        b += 4;
    }
    /* Process last 0-3 pixels.  Not needed for standard vector lengths. */
    while (a < last) {
        diff0 = *a++ - *b++;
        result += diff0 * diff0;
    }

    return result;
}

template<typename T>
inline void load_data(const char *filename, T *&data, unsigned &num, unsigned &dim) {
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        std::cerr << "open file error" << std::endl;
        exit(-1);
    }
    in.read((char *) &dim, 4);
    in.seekg(0, std::ios::end);
    std::ios::pos_type ss = in.tellg();
    auto f_size = (size_t) ss;
    num = (unsigned) (f_size / (dim + 1) / 4);
    data = new T[num * dim];

    in.seekg(0, std::ios::beg);
    for (size_t i = 0; i < num; i++) {
        in.seekg(4, std::ios::cur);
        in.read((char *) (data + i * dim), dim * sizeof(T));
    }
    in.close();
}

template<typename T>
void save_data(const char *filename, const T *dataset, unsigned n, unsigned d) {
    FILE *ofp = fopen(filename, "wb");

    //int val = 0;
    for (int i = 0; i < n; i++) {
        fwrite(&d, 4, 1, ofp);
        fflush(ofp);
        fwrite(&dataset[i * d], sizeof(T), d, ofp);
        fflush(ofp);

        //val += (4 + sizeof(T) * d);
    }
    //std::cout << val << std::endl;
    fclose(ofp);
}

float* data_align(float* data_ori, unsigned point_num, unsigned& dim){

    unsigned align_len = 4;
    std::cout << "align with : "<< align_len << std::endl;
    float* data_new=0;
    unsigned new_dim = (dim + align_len - 1) / align_len * align_len;
    std::cout << "align to new dim: "<<new_dim << std::endl;
    #ifdef __APPLE__
    data_new = new float[new_dim * point_num];
    #else
    data_new = (float*)memalign(align_len * 4, point_num * new_dim * sizeof(float));
    #endif

    for(unsigned i=0; i<point_num; i++){
    memcpy(data_new + i * new_dim, data_ori + i * dim, dim * sizeof(float));
    memset(data_new + i * new_dim + dim, 0, (new_dim - dim) * sizeof(float));
    }
    dim = new_dim;
    #ifdef __APPLE__
    delete[] data_ori;
    #else
    free(data_ori);
    #endif
    return data_new;
}

void gen_ground_truth(std::string graph_file, float *base_data, float *query_data, unsigned base_num, unsigned query_num, unsigned dim) {
    unsigned ground_dim = 50;
    unsigned base_dim = dim;

    std::chrono::high_resolution_clock::time_point s = std::chrono::high_resolution_clock::now();
    unsigned *groundtruth = new unsigned[query_num * ground_dim];
    omp_set_num_threads(4);

#pragma omp parallel for
    for (int i = 0; i < query_num; i++) {
        // std::cout << "save : " << i << std::endl;
        std::priority_queue<std::pair<float, unsigned>, std::vector<std::pair<float, unsigned> >, cmp> dist;
        for (int j = 0; j < base_num; j++) {
            float d = compare(query_data + base_dim * i, base_data + base_dim * j, base_dim);
            dist.push(std::pair<float, unsigned>(d, j));
        }

        for(int j = 0; j < ground_dim; j ++) {
            std::pair<float, unsigned> p = dist.top();
            dist.pop();
            if (p.second == i) continue;
            groundtruth[i * ground_dim + j] = p.second;
            // if (i == 10) {
            //     std::cout << p.first << std::endl;
            //     std::cout << p.second << std::endl;
            // }
        }
    }
    std::chrono::high_resolution_clock::time_point e = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> time = e - s;
    std::cout << "time : " << time.count() << std::endl;
    save_data<unsigned>(&graph_file[0], groundtruth, query_num, ground_dim);
}

int main(int argc, char** argv) {
    if (argc < 2) {
    std::cout << "./run data_file"
                << std::endl;
    exit(-1);
    }
    // sift1M, gist, glove-100, crawl, enron, MSong, audio, UQ_V
    // std::string dataset[] = {"sift1M", "gist", "glove-100", "crawl", "audio", "msong", "uqv", "enron"};
    // std::vector<std::string> datasets(dataset, dataset + 8);

    weavess::Parameters parameters;
    std::string dataset(argv[1]);
    std::string dataset_root = R"(/Users/wmz/Documents/Postgraduate/Code/dataset/)";
    std::string graph_file = dataset + "_50nn.graph";
    parameters.set<std::string>("dataset_root", dataset_root);
    set_data_path(dataset, parameters);
    std::string base_path = parameters.get<std::string>("base_path");
    float *base_data = nullptr;
    unsigned base_num;
    unsigned base_dim;
    load_data(&base_path[0], base_data, base_num, base_dim);
    base_data = data_align(base_data, base_num, base_dim);
    float *query_data = base_data;
    gen_ground_truth(graph_file, base_data, query_data, base_num, base_num, base_dim);

    return 0;
}

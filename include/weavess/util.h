//
// Created by Murph on 2020/8/12.
//

#ifndef WEAVESS_UTIL_H
#define WEAVESS_UTIL_H

#include <random>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <cstring>
#include <algorithm>

#ifdef __APPLE__
#else

#include <malloc.h>

#endif
namespace weavess {
    static void GenRandom(std::mt19937 &rng, unsigned *addr, unsigned size, unsigned N) {
        for (unsigned i = 0; i < size; ++i) {
            addr[i] = rng() % (N - size);
        }
        std::sort(addr, addr + size);
        for (unsigned i = 1; i < size; ++i) {
            if (addr[i] <= addr[i - 1]) {
                addr[i] = addr[i - 1] + 1;
            }
        }
        unsigned off = rng() % N;
        for (unsigned i = 0; i < size; ++i) {
            addr[i] = (addr[i] + off) % N;
        }
    }

    inline void
    load_data(char *filename, float *&data, unsigned &num, unsigned &dim) { // load data with sift10K pattern
        std::ifstream in(filename, std::ios::binary);
        if (!in.is_open()) {
            std::cout << "open file error" << std::endl;
            exit(-1);
        }
        in.read((char *) &dim, 4);
        std::cout << "data dimension: " << dim << std::endl;
        in.seekg(0, std::ios::end);
        std::ios::pos_type ss = in.tellg();
        size_t fsize = (size_t) ss;
        num = (unsigned) (fsize / (dim + 1) / 4);
        data = new float[num * dim * sizeof(float)];

        in.seekg(0, std::ios::beg);
        for (size_t i = 0; i < num; i++) {
            in.seekg(4, std::ios::cur);
            in.read((char *) (data + i * dim), dim * 4);
        }
        in.close();
    }

    inline void
    load_result_data(char *filename, unsigned *&data, unsigned &num, unsigned &dim) { // 载入ground_truth.ivecs
        std::ifstream in(filename, std::ios::binary);
        if (!in.is_open()) {
            std::cout << "open file error : " << filename << std::endl;
            return;
        }
        in.read((char *) &dim, 4);
        in.seekg(0, std::ios::end);
        std::ios::pos_type ss = in.tellg();
        unsigned fsize = (unsigned) ss;
        num = fsize / (dim + 1) / 4;
        data = new unsigned[num * dim];
        in.seekg(0, std::ios::beg);
        for (unsigned i = 0; i < num; i++) {
            in.seekg(4, std::ios::cur);
            in.read((char *) (data + i * dim), dim * 4);
        }
        in.close();
    }

    struct recall_speedup {
        float r, s;

        recall_speedup() = default;

        recall_speedup(float r, float s) : r{r}, s{s} {}

        inline bool operator<(const recall_speedup &other) const {
            return r < other.r;
        }
    };

    inline float evaluate_performance(std::vector<recall_speedup> &rs, float acc_st, float acc_up) {
        unsigned rs_size = rs.size();
        if (rs_size < 2 || rs[(rs_size - 1) / 2].r < acc_st) {
            std::cout << "Not enough samples!\n";
            return 0;
        }
        std::sort(rs.begin(), rs.end());
        float k1 = (rs[rs_size - 1].s - rs[0].s) / (rs[rs_size - 1].r - rs[0].r);
        float b1 = (rs[0].s - k1 * rs[0].r);
        float sp_st = k1 * acc_st + b1;
        float k2 = (rs[rs_size - 1].s - rs[rs_size - 2].s) / (rs[rs_size - 1].r - rs[rs_size - 2].r);
        float b2 = (rs[rs_size - 2].s - k1 * rs[rs_size - 2].r);
        float sp_up = k2 * acc_st + b2;

        // for (std::vector <recall_speedup> :: iterator it = rs.begin(); it < rs.end(); it++) {
        //   std::cout << "recall: " << it->r << "\t" << "speedup: " << it->s << "\n";

        //}
        while (acc_st >= rs.begin()->r) {
            rs.erase(rs.begin());
        }
        while (acc_up <= (rs.end() - 1)->r) {
            rs.erase((rs.end() - 1));
        }
        rs.push_back(recall_speedup(acc_up, sp_up));
        float tmp_r = acc_st, tmp_s = sp_st;
        float cur_perf = 0;
        for (std::vector<recall_speedup>::iterator it = rs.begin(); it < rs.end(); it++) {
            cur_perf += (tmp_s + it->s) * (it->r - tmp_r) / 2.0;
            tmp_s = it->s;
            tmp_r = it->r;
        }
        return cur_perf;
    }

    inline float *data_align(float *data_ori, unsigned point_num, unsigned &dim) {
#ifdef __GNUC__
#ifdef __AVX__
#define DATA_ALIGN_FACTOR 8
#else
#ifdef __SSE2__
#define DATA_ALIGN_FACTOR 4
#else
#define DATA_ALIGN_FACTOR 1
#endif
#endif
#endif

        std::cout << "align with : " << DATA_ALIGN_FACTOR << std::endl;
        float *data_new = 0;
        unsigned new_dim = (dim + DATA_ALIGN_FACTOR - 1) / DATA_ALIGN_FACTOR * DATA_ALIGN_FACTOR;
        std::cout << "align to new dim: " << new_dim << std::endl;
#ifdef __APPLE__
        data_new = new float[new_dim * point_num];
#else
        // windows
        data_new = (float *) _aligned_malloc(point_num * new_dim * sizeof(float), DATA_ALIGN_FACTOR * 4);
        // linux
        //data_new = (float *)memalign(DATA_ALIGN_FACTOR * 4, point_num * new_dim * sizeof(float));
#endif

        for (unsigned i = 0; i < point_num; i++) {
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
}

#endif //WEAVESS_UTIL_H

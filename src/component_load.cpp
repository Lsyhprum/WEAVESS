//
// Created by Murph on 2020/9/9.
//

#include "weavess/component.h"

namespace weavess {
    template<typename T>
    inline void load_data(char *filename, T *&data, unsigned &num, unsigned &dim) {
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

    inline void load_data_txt(char *filename, float *&data) {
        std::ifstream in(filename, std::ios::in);
        if (!in.is_open()) {
            std::cerr << "open file error" << std::endl;
            exit(-1);
        }
        int n = 0;
        data = new float[150 * 2];

        char buffer[256];
        while(!in.eof()) {
            in.getline(buffer, 50);

            int i = 0;
            while(buffer[i] < '0' || buffer[i] > '9') i ++;

            std::string s = "";
            while(buffer[i] >= '0' && buffer[i] <= '9') {
                s += buffer[i];
                i ++;
            }
            float num = std::stof(s);
            data[n] = num; n ++;

            while(buffer[i] < '0' || buffer[i] > '9') i ++;

            s = "";
            while(buffer[i] >= '0' && buffer[i] <= '9') {
                s += buffer[i];
                i ++;
            }
            float num2 = std::stof(s);
            data[n] = num2; n ++;
        }
        in.close();
    }

    void ComponentLoad::LoadInner(char *data_file, char *query_file, char *ground_file,
                                  Parameters &parameters) {
        // base_data
        float *data = nullptr;
        unsigned n{};
        unsigned dim{};
        load_data(data_file, data, n, dim);
        index->setBaseData(data);
        index->setBaseLen(n);
        index->setBaseDim(dim);

        assert(index->getBaseData() != nullptr && index->getBaseLen() != 0 && index->getBaseDim() != 0);

        // query_data
        float *query_data = nullptr;
        unsigned query_num{};
        unsigned query_dim{};
        load_data<float>(query_file, query_data, query_num, query_dim);
        index->setQueryData(query_data);
        index->setQueryLen(query_num);
        index->setQueryDim(query_dim);

        assert(index->getQueryData() != nullptr && index->getQueryLen() != 0 && index->getQueryDim() != 0);
        assert(index->getBaseDim() == index->getQueryDim());

        // ground_data
        unsigned *ground_data = nullptr;
        unsigned ground_num{};
        unsigned ground_dim{};
        load_data<unsigned>(ground_file, ground_data, ground_num, ground_dim);
        index->setGroundData(ground_data);
        index->setGroundLen(ground_num);
        index->setGroundDim(ground_dim);

        assert(index->getGroundData() != nullptr && index->getGroundLen() != 0 && index->getGroundDim() != 0);

        index->setParam(parameters);
    }
}
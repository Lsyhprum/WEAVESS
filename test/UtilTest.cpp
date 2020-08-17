//
// Created by Murph on 2020/8/15.
//

#include "weavess/util.h"

void LoadDataTest(){
    std::string query_path = "F:\\ANNS\\dataset\\sift1M\\sift_query.fvecs";
    float *query_data = nullptr;
    unsigned query_num, query_dim;

    weavess::load_data<float>(&query_path[0], query_data, query_num, query_dim);

    for(unsigned i = 0; i < query_num; i ++){
        for(unsigned j = 0; j < query_dim; j ++){
            std::cout << query_data[i * query_dim + j] << " ";
        }
        std::cout << std::endl;
    }
}

int main(){
    LoadDataTest();
}

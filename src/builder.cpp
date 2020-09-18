//
// Created by MurphySL on 2020/9/14.
//

#include "weavess/builder.h"
#include "weavess/component.h"

namespace weavess {
    IndexBuilder *IndexBuilder::load(char *data_file, char *query_file, char *ground_file, Parameters &parameters) {
        std::cout << "__LOAD DATA__" << std::endl;

        auto *a = new ComponentLoad(final_index_);
        a->LoadInner(data_file, query_file, ground_file, parameters);

        e = std::chrono::high_resolution_clock::now();

        return this;
    }

    IndexBuilder *IndexBuilder::init(TYPE type) {
        std::cout << "base data len : " << final_index_->getBaseLen() << std::endl;
        std::cout << "base data dim : " << final_index_->getBaseDim() << std::endl;
        std::cout << "query data len : " << final_index_->getQueryLen() << std::endl;
        std::cout << "query data dim : " << final_index_->getQueryDim() << std::endl;
        std::cout << "ground truth data len : " << final_index_->getGroundLen() << std::endl;
        std::cout << "ground truth data dim : " << final_index_->getGroundDim() << std::endl;

        std::cout << final_index_->getParam().toString() << std::endl;

        ComponentInit *a = nullptr;

        if (type == INIT_NN_DESCENT) {
            std::cout << "__INIT : NN-Descent__" << std::endl;
            a = new ComponentInitNNDescent(final_index_);
        } else if (type == INIT_KDT) {
            std::cout << "__INIT : KDT__" << std::endl;
            a = new ComponentInitKDT(final_index_);
        }

        a->InitInner();

        e = std::chrono::high_resolution_clock::now();
        std::cout << "__INIT FINISH__" << std::endl;

        return this;
    }

    IndexBuilder *IndexBuilder::refine(TYPE type) {
        ComponentRefine *a = nullptr;

        if (type == REFINE_NSG) {
            std::cout << "__REFINE : NSG__" << std::endl;
            a = new ComponentRefineNSG(final_index_);
        } else if (type == REFINE_NSSG) {
            std::cout << "__REFINE : NSSG__" << std::endl;
            a = new ComponentRefineNSSG(final_index_);
        } else if (type == REFINE_DPG) {
            std::cout << "__REFINE : DPG__" << std::endl;
            a = new ComponentRefineDPG(final_index_);
        } else if (type == REFINE_EFANNA) {
            std::cout << "__REFINE : EFANNA__" << std::endl;
            a = new ComponentRefineEFANNA(final_index_);
        } else if (type == REFINE_HNSW) {
            std::cout << "__REFINE : HNSW__" << std::endl;
            a = new ComponentRefineHNSW(final_index_);
        } else if(type == REFINE_VAMANA) {
            std::cout << "__REFINE : VAMANA__" << std::endl;
            a = new ComponentRefineVAMANA(final_index_);
        }

        a->RefineInner();

        std::cout << "__REFINE : FINISH__" << std::endl;

        e = std::chrono::high_resolution_clock::now();

        return this;
    }
}


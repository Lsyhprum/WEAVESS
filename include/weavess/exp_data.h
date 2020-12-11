#include "parameters.h"
#include <string.h>
#include <iostream>

void FANNG_PARA(std::string dataset, weavess::Parameters &parameters) {
    unsigned L, R;
    if (dataset == "siftsmall") {
        L = 100, R = 25;    // siftsmall
    }else if (dataset == "sift1M") {
        L = 110, R = 70;    // sift1M
    }else if (dataset == "gist") {
        L = 210, R = 50;    // gist
    }else if (dataset == "glove-100") {
        L = 210, R = 70;    // glove
    }else if (dataset == "audio") {
        L = 130, R = 50;    // audio
    }else if (dataset == "crawl") {
        L = 110, R = 30;    // crawl
    }else if (dataset == "msong") {
        L = 150, R = 10;    // msong
    }else if (dataset == "uqv") {
        L = 250, R = 90;    // uqv
    }else if (dataset == "enron") {
        L = 130, R = 110;    // enron
    }else if (dataset == "mnist") {
        L = 100, R = 25;    // mnist
    }else if (dataset == "c_1") {
        L = 30, R = 30;    // c_1
    }else if (dataset == "c_10") {
        L = 90, R = 10;    // c_10
    }else if (dataset == "c_100") {
        L = 150, R = 30;    // c_100
    }else if (dataset == "d_8") {
        L = 210, R = 110;    // d_8
    }else if (dataset == "d_128") {
        L = 210, R = 70;    // d_128
    }else if (dataset == "n_10000") {
        L = 110, R = 90;    // n_10000
    }else if (dataset == "n_1000000") {
        L = 110, R = 90;    // n_1000000
    }else if (dataset == "s_1") {
        L = 110, R = 30;    // s_1
    }else if (dataset == "s_10") {
        L = 250, R = 70;    // s_10
    }else {
        std::cout << "dataset error!\n";
        exit(-1);
    }
    parameters.set<unsigned>("L", L);
    parameters.set<unsigned>("R_refine", R);
}

void KGRAPH_PARA(std::string dataset, weavess::Parameters &parameters) {
    unsigned K, L, Iter, S, R;
    if (dataset == "siftsmall") {
        K = 25, L = 50, Iter = 6, S = 10, R = 100;  // siftsmall
    }else if (dataset == "sift1M") {
        K = 90, L = 130, Iter = 12, S = 20, R = 50;  // sift1M
    }else if (dataset == "gist") {
        K = 100, L = 120, Iter = 12, S = 25, R = 100;  // gist
    }else if (dataset == "glove-100") {
        K = 100, L = 150, Iter = 12, S = 35, R = 150;  // glove
    }else if (dataset == "audio") {
        K = 40, L = 60, Iter = 5, S = 20, R = 100;  // audio
    }else if (dataset == "crawl") {
        K = 80, L = 100, Iter = 12, S = 10, R = 150;  // crawl
    }else if (dataset == "msong") {
        K = 100, L = 140, Iter = 12, S = 15, R = 150;  // msong
    }else if (dataset == "uqv") {
        K = 40, L = 80, Iter = 6, S = 25, R = 100;  // uqv
    }else if (dataset == "enron") {
        K = 50, L = 80, Iter = 7, S = 15, R = 100;  // enron
    }else if (dataset == "mnist") {
        K = 25, L = 50, Iter = 8, S = 10, R = 100;  // kgraph
    }else if (dataset == "c_1") {
        K = 100, L = 110, Iter = 8, S = 25, R = 150;  // kgraph
    }else if (dataset == "c_10") {
        K = 100, L = 120, Iter = 8, S = 25, R = 50;  // kgraph
    }else if (dataset == "c_100") {
        K = 80, L = 130, Iter = 8, S = 35, R = 150;  // kgraph
    }else if (dataset == "d_8") {
        K = 50, L = 70, Iter = 8, S = 10, R = 150;  // kgraph
    }else if (dataset == "d_128") {
        K = 90, L = 90, Iter = 8, S = 30, R = 50;  // kgraph
    }else if (dataset == "n_10000") {
        K = 100, L = 140, Iter = 7, S = 30, R = 100;  // kgraph
    }else if (dataset == "n_1000000") {
        K = 100, L = 130, Iter = 12, S = 20, R = 50;  // kgraph
    }else if (dataset == "s_1") {
        K = 60, L = 60, Iter = 5, S = 20, R = 150;  // kgraph
    }else if (dataset == "s_10") {
        K = 80, L = 110, Iter = 9, S = 20, R = 150;  // kgraph
    }else {
        std::cout << "dataset error!\n";
        exit(-1);
    }
    parameters.set<unsigned>("K", K);
    parameters.set<unsigned>("L", L);
    parameters.set<unsigned>("ITER", Iter);
    parameters.set<unsigned>("S", S);
    parameters.set<unsigned>("R", R);
}

void NSG_PARA(std::string dataset, weavess::Parameters &parameters) {
    unsigned K, L, Iter, S, R, L_refine, R_refine, C;
    if (dataset == "siftsmall") {
        K = 25, L = 50, Iter = 6, S = 10, R = 100, L_refine = 100, R_refine = 50, C = 500;   // nsg
    }else if (dataset == "sift1M") {
        K = 100, L = 120, Iter = 12, S = 25, R = 300, L_refine = 150, R_refine = 30, C = 400;   // nsg
    }else if (dataset == "gist") {
        K = 400, L = 430, Iter = 12, S = 10, R = 200, L_refine = 500, R_refine = 20, C = 400;   // nsg
    }else if (dataset == "glove-100") {
        K = 400, L = 420, Iter = 12, S = 20, R = 300, L_refine = 150, R_refine = 90, C = 600;   // nsg
    }else if (dataset == "audio") {
        K = 200, L = 230, Iter = 5, S = 10, R = 100, L_refine = 200, R_refine = 30, C = 600;   // nsg
    }else if (dataset == "crawl") {
        K = 400, L = 430, Iter = 12, S = 15, R = 300, L_refine = 250, R_refine = 40, C = 600;   // nsg
    }else if (dataset == "msong") {
        K = 300, L = 310, Iter = 12, S = 25, R = 300, L_refine = 350, R_refine = 20, C = 500;   // nsg
    }else if (dataset == "uqv") {
        K = 300, L = 320, Iter = 6, S = 15, R = 200, L_refine = 350, R_refine = 30, C = 400;   // nsg
    }else if (dataset == "enron") {
        K = 200, L = 200, Iter = 7, S = 25, R = 200, L_refine = 150, R_refine = 60, C = 600;   // nsg
    }else if (dataset == "mnist") {
        K = 25, L = 50, Iter = 8, S = 10, R = 100, L_refine = 100, R_refine = 50, C = 600;   // nsg
    }else if (dataset == "c_1") {
        K = 300, L = 310, Iter = 8, S = 20, R = 200, L_refine = 200, R_refine = 80, C = 400;   // nsg
    }else if (dataset == "c_10") {
        K = 200, L = 200, Iter = 8, S = 20, R = 100, L_refine = 100, R_refine = 80, C = 400;   // nsg
    }else if (dataset == "c_100") {
        K = 400, L = 410, Iter = 8, S = 20, R = 100, L_refine = 400, R_refine = 20, C = 400;   // nsg
    }else if (dataset == "d_8") {
        K = 100, L = 100, Iter = 8, S = 10, R = 100, L_refine = 150, R_refine = 20, C = 600;   // nsg
    }else if (dataset == "d_128") {
        K = 200, L = 210, Iter = 8, S = 10, R = 300, L_refine = 150, R_refine = 20, C = 400;   // nsg
    }else if (dataset == "n_10000") {
        K = 300, L = 300, Iter = 7, S = 15, R = 300, L_refine = 50, R_refine = 20, C = 500;   // nsg
    }else if (dataset == "n_1000000") {
        K = 200, L = 200, Iter = 12, S = 20, R = 100, L_refine = 100, R_refine = 80, C = 400;   // nsg
    }else if (dataset == "s_1") {
        K = 200, L = 220, Iter = 5, S = 25, R = 300, L_refine = 300, R_refine = 20, C = 500;   // nsg
    }else if (dataset == "s_10") {
        K = 300, L = 300, Iter = 9, S = 25, R = 300, L_refine = 200, R_refine = 80, C = 400;   // nsg
    }else {
        std::cout << "input dataset error!\n";
        exit(-1);
    }
    parameters.set<unsigned>("K", K);
    parameters.set<unsigned>("L", L);
    parameters.set<unsigned>("ITER", Iter);
    parameters.set<unsigned>("S", S);
    parameters.set<unsigned>("R", R);

    parameters.set<unsigned>("L_refine", L_refine);
    parameters.set<unsigned>("R_refine", R_refine);
    parameters.set<unsigned>("C_refine", C);
}

void SSG_PARA(std::string dataset, weavess::Parameters &parameters) {
    unsigned K, L, Iter, S, R, L_refine, R_refine;
    if (dataset == "siftsmall") {
        K = 25, L = 50, Iter = 6, S = 10, R = 100, L_refine = 100, R_refine = 50;   // ssg
    }else if (dataset == "sift1M") {
        K = 400, L = 420, Iter = 12, S = 20, R = 100, L_refine = 50, R_refine = 20;   // ssg
    }else if (dataset == "gist") {
        K = 300, L = 330, Iter = 12, S = 20, R = 200, L_refine = 200, R_refine = 40;   // ssg
    }else if (dataset == "glove-100") {
        K = 300, L = 320, Iter = 12, S = 10, R = 200, L_refine = 150, R_refine = 30;   // ssg
    }else if (dataset == "audio") {
        K = 400, L = 400, Iter = 5, S = 25, R = 200, L_refine = 50, R_refine = 20;   // ssg
    }else if (dataset == "crawl") {
        K = 100, L = 100, Iter = 12, S = 10, R = 100, L_refine = 50, R_refine = 60;   // ssg
    }else if (dataset == "msong") {
        K = 400, L = 420, Iter = 12, S = 25, R = 300, L_refine = 100, R_refine = 70;   // ssg
    }else if (dataset == "uqv") {
        K = 400, L = 420, Iter = 6, S = 20, R = 300, L_refine = 250, R_refine = 20;   // ssg
    }else if (dataset == "enron") {
        K = 100, L = 110, Iter = 7, S = 20, R = 300, L_refine = 300, R_refine = 30;   // ssg
    }else if (dataset == "mnist") {
        K = 25, L = 50, Iter = 8, S = 10, R = 100, L_refine = 100, R_refine = 50;   // ssg
    }else if (dataset == "c_1") {
        K = 100, L = 120, Iter = 8, S = 15, R = 200, L_refine = 150, R_refine = 40;   // ssg
    }else if (dataset == "c_10") {
        K = 400, L = 430, Iter = 8, S = 25, R = 200, L_refine = 100, R_refine = 40;   // nsg
    }else if (dataset == "c_100") {
        K = 200, L = 210, Iter = 8, S = 25, R = 300, L_refine = 150, R_refine = 40;   // ssg
    }else if (dataset == "d_8") {
        K = 200, L = 230, Iter = 8, S = 25, R = 100, L_refine = 50, R_refine = 70;   // ssg
    }else if (dataset == "d_128") {
        K = 400, L = 420, Iter = 8, S = 25, R = 300, L_refine = 200, R_refine = 30;   // ssg
    }else if (dataset == "n_10000") {
        K = 200, L = 230, Iter = 7, S = 25, R = 100, L_refine = 100, R_refine = 40;   // ssg
    }else if (dataset == "n_1000000") {
        K = 400, L = 430, Iter = 12, S = 25, R = 200, L_refine = 100, R_refine = 40;   // ssg
    }else if (dataset == "s_1") {
        K = 100, L = 130, Iter = 5, S = 10, R = 100, L_refine = 200, R_refine = 40;   // ssg
    }else if (dataset == "s_10") {
        K = 300, L = 320, Iter = 9, S = 20, R = 300, L_refine = 500, R_refine = 60;   // ssg
    }else {
        std::cout << "input dataset error!\n";
        exit(-1);
    }
    parameters.set<unsigned>("K", K);
    parameters.set<unsigned>("L", L);
    parameters.set<unsigned>("ITER", Iter);
    parameters.set<unsigned>("S", S);
    parameters.set<unsigned>("R", R);

    parameters.set<unsigned>("L_refine", L_refine);
    parameters.set<unsigned>("R_refine", R_refine);
    parameters.set<float>("A", 60);
    parameters.set<unsigned>("n_try", 10);
}

void DPG_PARA(std::string dataset, weavess::Parameters &parameters) {
    unsigned K, L, Iter, S, R;
    if (dataset == "siftsmall") {
        K = 25, L = 50, Iter = 6, S = 10, R = 100;   // siftsmall
    }else if (dataset == "sift1M") {
        K = 100, L = 100, Iter = 12, S = 25, R = 300;   // sift1M
    }else if (dataset == "gist") {
        K = 100, L = 100, Iter = 12, S = 20, R = 100;   // gist
    }else if (dataset == "glove-100") {
        K = 100, L = 130, Iter = 12, S = 20, R = 100;   // glove
    }else if (dataset == "audio") {
        K = 100, L = 130, Iter = 5, S = 25, R = 100;   // audio
    }else if (dataset == "crawl") {
        K = 100, L = 130, Iter = 12, S = 20, R = 100;   // crawl
    }else if (dataset == "msong") {
        K = 100, L = 110, Iter = 12, S = 20, R = 100;   // msong
    }else if (dataset == "uqv") {
        K = 100, L = 100, Iter = 6, S = 20, R = 300;   // uqv
    }else if (dataset == "enron") {
        K = 100, L = 120, Iter = 7, S = 15, R = 200;   // enron
    }else if (dataset == "mnist") {
        K = 25, L = 50, Iter = 8, S = 10, R = 100;   // mnist
    }else if (dataset == "c_1") {
        K = 100, L = 100, Iter = 8, S = 20, R = 300;   // c_1
    }else if (dataset == "c_10") {
        K = 200, L = 210, Iter = 8, S = 25, R = 300;   // c_10
    }else if (dataset == "c_100") {
        K = 200, L = 220, Iter = 8, S = 25, R = 200;   // c_100
    }else if (dataset == "d_8") {
        K = 100, L = 110, Iter = 8, S = 20, R = 200;   // d_8
    }else if (dataset == "d_128") {
        K = 100, L = 100, Iter = 8, S = 25, R = 200;   // d_128
    }else if (dataset == "n_10000") {
        K = 100, L = 130, Iter = 7, S = 20, R = 300;   // n_10000
    }else if (dataset == "n_1000000") {
        K = 100, L = 100, Iter = 12, S = 20, R = 300;   // n_1000000
    }else if (dataset == "s_1") {
        K = 100, L = 100, Iter = 5, S = 20, R = 100;   // s_1
    }else if (dataset == "s_10") {
        K = 100, L = 100, Iter = 9, S = 20, R = 200;   // s_10
    }else {
        std::cout << "input dataset error!\n";
        exit(-1);
    }
    parameters.set<unsigned>("K", K);
    parameters.set<unsigned>("L", L);
    parameters.set<unsigned>("ITER", Iter);
    parameters.set<unsigned>("S", S);
    parameters.set<unsigned>("R", R);
}

void VAMANA_PARA(std::string dataset, weavess::Parameters &parameters) {
    unsigned L, R;
    if (dataset == "siftsmall") {
        L = 100, R = 25;    // siftsmall
    }else if (dataset == "sift1M") {
        L = 70, R = 50;    // sift1M
    }else if (dataset == "gist") {
        L = 60, R = 50;    // gist
    }else if (dataset == "glove-100") {
        L = 120, R = 110;    // glove
    }else if (dataset == "audio") {
        L = 70, R = 50;    // audio
    }else if (dataset == "crawl") {
        L = 80, R = 50;    // crawl
    }else if (dataset == "msong") {
        L = 40, R = 30;    // msong
    }else if (dataset == "uqv") {
        L = 60, R = 30;    // uqv
    }else if (dataset == "enron") {
        L = 140, R = 110;    // enron
    }else if (dataset == "mnist") {
        L = 100, R = 25;    // mnist
    }else if (dataset == "c_1") {
        L = 140, R = 110;    // c_1
    }else if (dataset == "c_10") {
        L = 90, R = 70;    // c_10
    }else if (dataset == "c_100") {
        L = 60, R = 50;    // c_100
    }else if (dataset == "d_8") {
        L = 130, R = 110;    // d_8
    }else if (dataset == "d_128") {
        L = 80, R = 70;    // d_128
    }else if (dataset == "n_10000") {
        L = 60, R = 50;    // n_10000
    }else if (dataset == "n_1000000") {
        L = 80, R = 70;    // n_1000000
    }else if (dataset == "s_1") {
        L = 90, R = 70;    // s_1
    }else if (dataset == "s_10") {
        L = 120, R = 90;    // s_10
    }else {
        std::cout << "dataset error!\n";
        exit(-1);
    }
    parameters.set<unsigned>("L", R);
    parameters.set<unsigned>("L_refine", L);
    parameters.set<unsigned>("R_refine", R);
}

void EFANNA_PARA(std::string dataset, weavess::Parameters &parameters) {
    unsigned nTrees, mLevel, K, L, Iter, S, R;
    if (dataset == "siftsmall") {
        nTrees = 8, mLevel = 4, K = 60, L = 70, Iter = 10, S = 15, R = 150;    // siftsmall
    }else if (dataset == "sift1M") {
        nTrees = 8, mLevel = 8, K = 60, L = 70, Iter = 10, S = 15, R = 150;    // sift1M
    }else if (dataset == "gist") {
        nTrees = 16, mLevel = 8, K = 100, L = 190, Iter = 7, S = 30, R = 50;    // gist
    }else if (dataset == "glove-100") {
        nTrees = 8, mLevel = 8, K = 100, L = 170, Iter = 7, S = 10, R = 100;    // glove
    }else if (dataset == "audio") {
        nTrees = 16, mLevel = 8, K = 40, L = 10, Iter = 10, S = 30, R = 100;    // audio
    }else if (dataset == "crawl") {
        nTrees = 16, mLevel = 8, K = 100, L = 120, Iter = 8, S = 25, R = 100;    // crawl
    }else if (dataset == "msong") {
        nTrees = 8, mLevel = 8, K = 50, L = 130, Iter = 7, S = 10, R = 150;    // msong
    }else if (dataset == "uqv") {
        nTrees = 4, mLevel = 8, K = 40, L = 50, Iter = 7, S = 10, R = 150;    // uqv
    }else if (dataset == "enron") {
        nTrees = 4, mLevel = 8, K = 40, L = 140, Iter = 5, S = 35, R = 150;    // enron
    }else if (dataset == "mnist") {
        nTrees = 4, mLevel = 8, K = 40, L = 140, Iter = 5, S = 35, R = 150;    // mnist
    }else if (dataset == "c_1") {
        nTrees = 4, mLevel = 8, K = 90, L = 190, Iter = 7, S = 15, R = 50;    // c_1
    }else if (dataset == "c_10") {
        nTrees = 4, mLevel = 8, K = 100, L = 160, Iter = 7, S = 15, R = 100;    // c_10
    }else if (dataset == "c_100") {
        nTrees = 4, mLevel = 8, K = 70, L = 160, Iter = 7, S = 35, R = 150;    // c_100
    }else if (dataset == "d_8") {
        nTrees = 8, mLevel = 8, K = 50, L = 120, Iter = 7, S = 15, R = 50;    // d_8
    }else if (dataset == "d_128") {
        nTrees = 4, mLevel = 8, K = 40, L = 40, Iter = 7, S = 25, R = 150;    // d_128
    }else if (dataset == "n_10000") {
        nTrees = 4, mLevel = 8, K = 80, L = 140, Iter = 7, S = 25, R = 150;    // n_10000
    }else if (dataset == "n_1000000") {
        nTrees = 4, mLevel = 8, K = 100, L = 160, Iter = 7, S = 15, R = 100;    // n_1000000
    }else if (dataset == "s_1") {
        nTrees = 32, mLevel = 8, K = 100, L = 160, Iter = 7, S = 35, R = 150;    // s_1
    }else if (dataset == "s_10") {
        nTrees = 32, mLevel = 8, K = 100, L = 110, Iter = 7, S = 30, R = 100;    // s_10
    }else {
        std::cout << "dataset error!\n";
        exit(-1);
    }
    parameters.set<unsigned>("nTrees", nTrees);
    parameters.set<unsigned>("mLevel", mLevel);
    parameters.set<unsigned>("K", K);
    parameters.set<unsigned>("L", L);
    parameters.set<unsigned>("ITER", Iter);
    parameters.set<unsigned>("S", S);
    parameters.set<unsigned>("R", R);
}

void NSW_PARA(std::string dataset, weavess::Parameters &parameters) {
    unsigned max_m0, ef_construction;
    if (dataset == "siftsmall") {
        max_m0 = 10, ef_construction = 100;    // siftsmall
    }else if (dataset == "sift1M") {
        max_m0 = 40, ef_construction = 300;    // sift1M
    }else if (dataset == "gist") {
        max_m0 = 60, ef_construction = 200;    // gist
    }else if (dataset == "glove-100") {
        max_m0 = 80, ef_construction = 100;    // glove
    }else if (dataset == "audio") {
        max_m0 = 40, ef_construction = 800;    // audio
    }else if (dataset == "crawl") {
        max_m0 = 60, ef_construction = 400;    // crawl
    }else if (dataset == "msong") {
        max_m0 = 60, ef_construction = 300;    // msong
    }else if (dataset == "uqv") {
        max_m0 = 30, ef_construction = 400;    // uqv
    }else if (dataset == "enron") {
        max_m0 = 80, ef_construction = 600;    // enron
    }else if (dataset == "mnist") {
        max_m0 = 10, ef_construction = 25;    // mnist
    }else if (dataset == "c_1") {
        max_m0 = 100, ef_construction = 500;    // c_1
    }else if (dataset == "c_10") {
        max_m0 = 30, ef_construction = 100;    // c_10
    }else if (dataset == "c_100") {
        max_m0 = 70, ef_construction = 400;    // c_100
    }else if (dataset == "d_8") {
        max_m0 = 50, ef_construction = 500;    // d_8
    }else if (dataset == "d_128") {
        max_m0 = 80, ef_construction = 1000;    // d_128
    }else if (dataset == "n_10000") {
        max_m0 = 20, ef_construction = 300;    // n_10000
    }else if (dataset == "n_1000000") {
        max_m0 = 100, ef_construction = 400;    // n_1000000
    }else if (dataset == "s_1") {
        max_m0 = 60, ef_construction = 600;    // s_1
    }else if (dataset == "s_10") {
        max_m0 = 50, ef_construction = 1000;    // s_10
    }else {
        std::cout << "dataset error!\n";
        exit(-1);
    }
    parameters.set<unsigned>("NN", max_m0);
    parameters.set<unsigned>("ef_construction", ef_construction);
}

void HCNNG_PARA(std::string dataset, weavess::Parameters &parameters) {
    unsigned num_cl;
    if (dataset == "siftsmall") {
        num_cl = 10;    // siftsmall
    }else if (dataset == "sift1M") {
        num_cl = 90;    // sift1M
    }else if (dataset == "gist") {
        num_cl = 30;    // gist
    }else if (dataset == "glove-100") {
        num_cl = 100;    // glove
    }else if (dataset == "audio") {
        num_cl = 60;    // audio
    }else if (dataset == "crawl") {
        num_cl = 70;    // crawl
    }else if (dataset == "msong") {
        num_cl = 100;    // msong
    }else if (dataset == "uqv") {
        num_cl = 80;    // uqv
    }else if (dataset == "enron") {
        num_cl = 100;    // enron
    }else if (dataset == "mnist") {
        num_cl = 10;    // mnist
    }else if (dataset == "c_1") {
        num_cl = 100;    // c_1
    }else if (dataset == "c_10") {
        num_cl = 70;    // c_10
    }else if (dataset == "c_100") {
        num_cl = 90;    // c_100
    }else if (dataset == "d_8") {
        num_cl = 80;    // d_8
    }else if (dataset == "d_128") {
        num_cl = 100;    // d_128
    }else if (dataset == "n_10000") {
        num_cl = 90;    // n_10000
    }else if (dataset == "n_1000000") {
        num_cl = 100;    // n_1000000
    }else if (dataset == "s_1") {
        num_cl = 60;    // s_1
    }else if (dataset == "s_10") {
        num_cl = 90;    // s_10
    }else {
        std::cout << "dataset error!\n";
        exit(-1);
    }
    parameters.set<unsigned>("num_cl", num_cl);
    parameters.set<unsigned>("K", 10);
    parameters.set<unsigned>("nTrees", 10);
    parameters.set<unsigned>("mLevel", 1);
}

void set_data_path(std::string dataset, weavess::Parameters &parameters) {
    // dataset root path
    std::string dataset_root = parameters.get<std::string>("dataset_root");
    std::string base_path(dataset_root);
    std::string query_path(dataset_root);
    std::string ground_path(dataset_root);
    if (dataset == "siftsmall") {
        base_path.append(R"(siftsmall/siftsmall_base.fvecs)");
        query_path.append(R"(siftsmall/siftsmall_query.fvecs)");
        ground_path.append(R"(siftsmall/siftsmall_groundtruth.ivecs)");

    }else if (dataset == "sift1M") {
        base_path.append(R"(sift1M/sift_base.fvecs)");
        query_path.append(R"(sift1M/sift_query.fvecs)");
        ground_path.append(R"(sift1M/sift_groundtruth.ivecs)");

    }else if (dataset == "gist") {
        base_path.append(R"(gist/gist_base.fvecs)");
        query_path.append(R"(gist/gist_query.fvecs)");
        ground_path.append(R"(gist/gist_groundtruth.ivecs)");

    }else if (dataset == "glove-100") {
        base_path.append(R"(glove-100/glove-100_base.fvecs)");
        query_path.append(R"(glove-100/glove-100_query.fvecs)");
        ground_path.append(R"(glove-100/glove-100_groundtruth.ivecs)");

    }else if (dataset == "audio") {
        base_path.append(R"(audio/audio_base.fvecs)");
        query_path.append(R"(audio/audio_query.fvecs)");
        ground_path.append(R"(audio/audio_groundtruth.ivecs)");

    }else if (dataset == "crawl") {
        base_path.append(R"(crawl/crawl_base.fvecs)");
        query_path.append(R"(crawl/crawl_query.fvecs)");
        ground_path.append(R"(crawl/crawl_groundtruth.ivecs)");

    }else if (dataset == "msong") {
        base_path.append(R"(msong/msong_base.fvecs)");
        query_path.append(R"(msong/msong_query.fvecs)");
        ground_path.append(R"(msong/msong_groundtruth.ivecs)");

    }else if (dataset == "uqv") {
        base_path.append(R"(uqv/uqv_base.fvecs)");
        query_path.append(R"(uqv/uqv_query.fvecs)");
        ground_path.append(R"(uqv/uqv_groundtruth.ivecs)");

    }else if (dataset == "enron") {
        base_path.append(R"(enron/enron_base.fvecs)");
        query_path.append(R"(enron/enron_query.fvecs)");
        ground_path.append(R"(enron/enron_groundtruth.ivecs)");

    }else if (dataset == "mnist") {
        base_path.append(R"(mnist/mnist_base.fvecs)");
        query_path.append(R"(mnist/mnist_query.fvecs)");
        ground_path.append(R"(mnist/mnist_groundtruth.ivecs)");

    }else if (dataset == "c_1") {
        base_path.append(R"(c_1/random_base_n100000_d32_c1_s5.fvecs)");
        query_path.append(R"(c_1/random_query_n1000_d32_c1_s5.fvecs)");
        ground_path.append(R"(c_1/random_ground_truth_n1000_d32_c1_s5.ivecs)");

    }else if (dataset == "c_10") {
        base_path.append(R"(c_10/random_base_n100000_d32_c10_s5.fvecs)");
        query_path.append(R"(c_10/random_query_n1000_d32_c10_s5.fvecs)");
        ground_path.append(R"(c_10/random_ground_truth_n1000_d32_c10_s5.ivecs)");

    }else if (dataset == "c_100") {
        base_path.append(R"(c_100/random_base_n100000_d32_c100_s5.fvecs)");
        query_path.append(R"(c_100/random_query_n1000_d32_c100_s5.fvecs)");
        ground_path.append(R"(c_100/random_ground_truth_n1000_d32_c100_s5.ivecs)");

    }else if (dataset == "d_8") {
        base_path.append(R"(d_8/random_base_n100000_d8_c10_s5.fvecs)");
        query_path.append(R"(d_8/random_query_n1000_d8_c10_s5.fvecs)");
        ground_path.append(R"(d_8/random_ground_truth_n1000_d8_c10_s5.ivecs)");

    }else if (dataset == "d_128") {
        base_path.append(R"(d_128/random_base_n100000_d128_c10_s5.fvecs)");
        query_path.append(R"(d_128/random_query_n1000_d128_c10_s5.fvecs)");
        ground_path.append(R"(d_128/random_ground_truth_n1000_d128_c10_s5.ivecs)");

    }else if (dataset == "n_10000") {
        base_path.append(R"(n_10000/random_base_n10000_d32_c10_s5.fvecs)");
        query_path.append(R"(n_10000/random_query_n100_d32_c10_s5.fvecs)");
        ground_path.append(R"(n_10000/random_ground_truth_n100_d32_c10_s5.ivecs)");

    }else if (dataset == "n_1000000") {
        base_path.append(R"(n_1000000/random_base_n1000000_d32_c10_s5.fvecs)");
        query_path.append(R"(n_1000000/random_query_n10000_d32_c10_s5.fvecs)");
        ground_path.append(R"(n_1000000/random_ground_truth_n10000_d32_c10_s5.ivecs)");

    }else if (dataset == "s_1") {
        base_path.append(R"(s_1/random_base_n100000_d32_c10_s1.fvecs)");
        query_path.append(R"(s_1/random_query_n1000_d32_c10_s1.fvecs)");
        ground_path.append(R"(s_1/random_ground_truth_n1000_d32_c10_s1.ivecs)");

    }else if (dataset == "s_10") {
        base_path.append(R"(s_10/random_base_n100000_d32_c10_s10.fvecs)");
        query_path.append(R"(s_10/random_query_n1000_d32_c10_s10.fvecs)");
        ground_path.append(R"(s_10/random_ground_truth_n1000_d32_c10_s10.ivecs)");

    }else {
        std::cout << "dataset input error!\n";
        exit(-1);
    }
    parameters.set<std::string>("base_path", base_path);
    parameters.set<std::string>("query_path", query_path);
    parameters.set<std::string>("ground_path", ground_path);
}

void set_para(std::string alg, std::string dataset, weavess::Parameters &parameters) {
    
    set_data_path(dataset, parameters);
    if (parameters.get<std::string>("exc_type") == "search") {
        return;
    }

    if (alg == "fanng") {
        FANNG_PARA(dataset, parameters);
    }else if (alg == "kgraph") {
        KGRAPH_PARA(dataset, parameters);
    }else if (alg == "nsg") {
        NSG_PARA(dataset, parameters);
    }else if (alg == "ssg") {
        SSG_PARA(dataset, parameters);
    }else if (alg == "dpg") {
        DPG_PARA(dataset, parameters);
    }else if (alg == "vamana") {
        VAMANA_PARA(dataset, parameters);
    }else if (alg == "efanna") {
        EFANNA_PARA(dataset, parameters);
    }else if (alg == "nsw") {
        NSW_PARA(dataset, parameters);
    }else if (alg == "hcnng") {
        HCNNG_PARA(dataset, parameters);
    }
    else {
        std::cout << "algorithm input error!\n";
        exit(-1);
    }
}
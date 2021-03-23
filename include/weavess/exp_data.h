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
        L = 30, R = 20;    // siftsmall
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
        nTrees = 8, mLevel = 2, K = 30, L = 50, Iter = 10, S = 15, R = 150;    // siftsmall
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

void SPTAG_KDT_PARA(std::string dataset, weavess::Parameters &parameters) {
    unsigned KDT_Number, TPT_Number, TPT_leaf_size, scale, CEF;
    if (dataset == "siftsmall") {
        KDT_Number = 1, TPT_Number = 16, TPT_leaf_size = 500, scale = 8, CEF = 500;    // siftsmall
    }else if (dataset == "sift1M") {
        KDT_Number = 1, TPT_Number = 16, TPT_leaf_size = 500, scale = 8, CEF = 500;    // sift1M
    }else if (dataset == "gist") {
        KDT_Number = 2, TPT_Number = 32, TPT_leaf_size = 1000, scale = 32, CEF = 1000;    // gist
    }else if (dataset == "glove-100") {
        KDT_Number = 4, TPT_Number = 32, TPT_leaf_size = 1000, scale = 2, CEF = 1500;    // glove
    }else if (dataset == "audio") {
        KDT_Number = 2, TPT_Number = 32, TPT_leaf_size = 500, scale = 32, CEF = 1000;    // audio
    }else if (dataset == "crawl") {
        KDT_Number = 1, TPT_Number = 16, TPT_leaf_size = 1000, scale = 2, CEF = 1500;    // crawl
    }else if (dataset == "msong") {
        KDT_Number = 4, TPT_Number = 64, TPT_leaf_size = 1500, scale = 2, CEF = 500;    // msong
    }else if (dataset == "uqv") {
        KDT_Number = 1, TPT_Number = 16, TPT_leaf_size = 500, scale = 8, CEF = 1000;    // uqv
    }else if (dataset == "enron") {
        KDT_Number = 4, TPT_Number = 16, TPT_leaf_size = 500, scale = 8, CEF = 1500;    // enron
    }else if (dataset == "mnist") {
        KDT_Number = 1, TPT_Number = 16, TPT_leaf_size = 500, scale = 8, CEF = 500;    // mnist
    }else if (dataset == "c_1") {
        KDT_Number = 1, TPT_Number = 16, TPT_leaf_size = 1000, scale = 2, CEF = 1000;    // c_1
    }else if (dataset == "c_10") {
        KDT_Number = 2, TPT_Number = 16, TPT_leaf_size = 1000, scale = 8, CEF = 1500;    // c_10
    }else if (dataset == "c_100") {
        KDT_Number = 1, TPT_Number = 16, TPT_leaf_size = 1500, scale = 8, CEF = 1500;    // c_100
    }else if (dataset == "d_8") {
        KDT_Number = 4, TPT_Number = 32, TPT_leaf_size = 500, scale = 2, CEF = 1500;    // d_8
    }else if (dataset == "d_128") {
        KDT_Number = 4, TPT_Number = 16, TPT_leaf_size = 1500, scale = 2, CEF = 1500;    // d_128
    }else if (dataset == "n_10000") {
        KDT_Number = 2, TPT_Number = 16, TPT_leaf_size = 1000, scale = 8, CEF = 1500;    // n_10000
    }else if (dataset == "n_1000000") {
        KDT_Number = 2, TPT_Number = 16, TPT_leaf_size = 1000, scale = 8, CEF = 1500;    // n_1000000
    }else if (dataset == "s_1") {
        KDT_Number = 1, TPT_Number = 16, TPT_leaf_size = 1500, scale = 8, CEF = 1000;    // s_1
    }else if (dataset == "s_10") {
        KDT_Number = 1, TPT_Number = 32, TPT_leaf_size = 1500, scale = 32, CEF = 1500;    // s_10
    }else {
        std::cout << "dataset error!\n";
        exit(-1);
    }
    parameters.set<unsigned>("KDTNumber", KDT_Number);
    parameters.set<unsigned>("TPTNumber", TPT_Number);
    parameters.set<unsigned>("TPTLeafSize", TPT_leaf_size);
    parameters.set<unsigned>("NeighborhoodSize", 32);
    parameters.set<unsigned>("GraphNeighborhoodScale", scale);
    parameters.set<unsigned>("CEF", CEF);
    parameters.set<unsigned>("numOfThreads", parameters.get<unsigned>("n_threads"));
}

void SPTAG_BKT_PARA(std::string dataset, weavess::Parameters &parameters) {
    unsigned BKT_Number, BKT_kmeans_k, TPT_Number, TPT_leaf_size, scale, CEF;
    if (dataset == "siftsmall") {
        BKT_Number = 1, BKT_kmeans_k = 16, TPT_Number = 16, TPT_leaf_size = 500, scale = 2, CEF = 500;    // siftsmall
    }else if (dataset == "sift1M") {
        BKT_Number = 1, BKT_kmeans_k = 16, TPT_Number = 16, TPT_leaf_size = 1000, scale = 2, CEF = 1000;    // sift1M
    }else if (dataset == "gist") {
        BKT_Number = 2, BKT_kmeans_k = 16, TPT_Number = 32, TPT_leaf_size = 1000, scale = 8, CEF = 1000;    // gist
    }else if (dataset == "glove-100") {
        BKT_Number = 4, BKT_kmeans_k = 16, TPT_Number = 16, TPT_leaf_size = 500, scale = 8, CEF = 1500;    // glove
    }else if (dataset == "audio") {
        BKT_Number = 4, BKT_kmeans_k = 16, TPT_Number = 16, TPT_leaf_size = 1500, scale = 32, CEF = 1000;    // audio
    }else if (dataset == "crawl") {
        BKT_Number = 4, BKT_kmeans_k = 32, TPT_Number = 32, TPT_leaf_size = 1000, scale = 32, CEF = 500;    // crawl
    }else if (dataset == "msong") {
        BKT_Number = 4, BKT_kmeans_k = 64, TPT_Number = 16, TPT_leaf_size = 500, scale = 2, CEF = 500;    // msong
    }else if (dataset == "uqv") {
        BKT_Number = 4, BKT_kmeans_k = 64, TPT_Number = 16, TPT_leaf_size = 1000, scale = 2, CEF = 500;    // uqv
    }else if (dataset == "enron") {
        BKT_Number = 1, BKT_kmeans_k = 16, TPT_Number = 16, TPT_leaf_size = 1500, scale = 2, CEF = 500;;    // enron
    }else if (dataset == "mnist") {
        BKT_Number = 1, BKT_kmeans_k = 16, TPT_Number = 16, TPT_leaf_size = 500, scale = 2, CEF = 500;    // mnist
    }else if (dataset == "c_1") {
        BKT_Number = 1, BKT_kmeans_k = 16, TPT_Number = 16, TPT_leaf_size = 1000, scale = 2, CEF = 1000;    // c_1
    }else if (dataset == "c_10") {
        BKT_Number = 2, BKT_kmeans_k = 32, TPT_Number = 16, TPT_leaf_size = 1500, scale = 2, CEF = 500;    // c_10
    }else if (dataset == "c_100") {
        BKT_Number = 2, BKT_kmeans_k = 32, TPT_Number = 16, TPT_leaf_size = 1500, scale = 2, CEF = 500;    // c_100
    }else if (dataset == "d_8") {
        BKT_Number = 1, BKT_kmeans_k = 32, TPT_Number = 16, TPT_leaf_size = 1000, scale = 2, CEF = 1000;    // d_8
    }else if (dataset == "d_128") {
        BKT_Number = 4, BKT_kmeans_k = 16, TPT_Number = 32, TPT_leaf_size = 500, scale = 2, CEF = 500;    // d_128
    }else if (dataset == "n_10000") {
        BKT_Number = 2, BKT_kmeans_k = 32, TPT_Number = 16, TPT_leaf_size = 1500, scale = 2, CEF = 500;    // n_10000
    }else if (dataset == "n_1000000") {
        BKT_Number = 2, BKT_kmeans_k = 32, TPT_Number = 16, TPT_leaf_size = 1500, scale = 2, CEF = 500;    // n_1000000
    }else if (dataset == "s_1") {
        BKT_Number = 1, BKT_kmeans_k = 16, TPT_Number = 16, TPT_leaf_size = 1000, scale = 2, CEF = 500;    // s_1
    }else if (dataset == "s_10") {
        BKT_Number = 1, BKT_kmeans_k = 32, TPT_Number = 64, TPT_leaf_size = 2000, scale = 8, CEF = 1500;    // s_10
    }else {
        std::cout << "dataset error!\n";
        exit(-1);
    }
    parameters.set<unsigned>("BKTNumber", BKT_Number);
    parameters.set<unsigned>("BKTKMeansK", BKT_kmeans_k);
    parameters.set<unsigned>("TPTNumber", TPT_Number);
    parameters.set<unsigned>("TPTLeafSize", TPT_leaf_size);
    parameters.set<unsigned>("NeighborhoodSize", 32);
    parameters.set<unsigned>("GraphNeighborhoodScale", scale);
    parameters.set<unsigned>("CEF", CEF);
    parameters.set<unsigned>("numOfThreads", parameters.get<unsigned>("n_threads"));
}

void HNSW_PARA(std::string dataset, weavess::Parameters &parameters) {
    unsigned max_m, max_m0, ef_construction;
    if (dataset == "siftsmall") {
        max_m = 25, max_m0 = 50, ef_construction = 100;    // siftsmall
    }else if (dataset == "sift1M") {
        max_m = 40, max_m0 = 50, ef_construction = 800;    // sift1M
    }else if (dataset == "gist") {
        max_m = 50, max_m0 = 60, ef_construction = 400;    // gist
    }else if (dataset == "glove-100") {
        max_m = 50, max_m0 = 60, ef_construction = 700;    // glove
    }else if (dataset == "audio") {
        max_m = 10, max_m0 = 50, ef_construction = 700;    // audio
    }else if (dataset == "crawl") {
        max_m = 40, max_m0 = 70, ef_construction = 400;    // crawl
    }else if (dataset == "msong") {
        max_m = 30, max_m0 = 80, ef_construction = 100;    // msong
    }else if (dataset == "uqv") {
        max_m = 10, max_m0 = 40, ef_construction = 200;    // uqv
    }else if (dataset == "enron") {
        max_m = 50, max_m0 = 80, ef_construction = 900;    // enron
    }else if (dataset == "mnist") {
        max_m = 5, max_m0 = 10, ef_construction = 25;    // mnist
    }else if (dataset == "c_1") {
        max_m = 80, max_m0 = 90, ef_construction = 1000;    // c_1
    }else if (dataset == "c_10") {
        max_m = 40, max_m0 = 60, ef_construction = 300;    // c_10
    }else if (dataset == "c_100") {
        max_m = 30, max_m0 = 40, ef_construction = 300;    // c_100
    }else if (dataset == "d_8") {
        max_m = 10, max_m0 = 30, ef_construction = 900;    // d_8
    }else if (dataset == "d_128") {
        max_m = 90, max_m0 = 100, ef_construction = 900;    // d_128
    }else if (dataset == "n_10000") {
        max_m = 20, max_m0 = 30, ef_construction = 900;    // n_10000
    }else if (dataset == "n_1000000") {
        max_m = 50, max_m0 = 100, ef_construction = 200;    // n_1000000
    }else if (dataset == "s_1") {
        max_m = 40, max_m0 = 60, ef_construction = 200;    // s_1
    }else if (dataset == "s_10") {
        max_m = 60, max_m0 = 80, ef_construction = 1000;    // s_10
    }else {
        std::cout << "dataset error!\n";
        exit(-1);
    }
    parameters.set<unsigned>("max_m", max_m);
    parameters.set<unsigned>("max_m0", max_m0);
    parameters.set<unsigned>("ef_construction", ef_construction);
    parameters.set<int>("mult", -1);
}

void IEH_PARA(std::string dataset, weavess::Parameters &parameters) {
    std::string index_path = parameters.get<std::string>("index_path");
    std::string LSHtable_path(index_path);
    std::string LSHfunc_path(index_path);

    LSHtable_path.append("LSHtable_" + dataset + ".txt");
    LSHfunc_path.append("LSHfunc_" + dataset + ".txt");

    parameters.set<std::string>("train", parameters.get<std::string>("base_path"));
    parameters.set<std::string>("test", parameters.get<std::string>("query_path"));
    parameters.set<std::string>("func", LSHfunc_path);
    parameters.set<std::string>("basecode", LSHtable_path);
    parameters.set<std::string>("knntable", parameters.get<std::string>("graph_file"));

    // parameters.set<unsigned>("expand", 10);
    parameters.set<unsigned>("iterlimit", 3);
}

void PANNG_PARA(std::string dataset, weavess::Parameters &parameters) {
    unsigned K, L;
    if (dataset == "siftsmall") {
        K = 10, L = 50;    // siftsmall
    }else if (dataset == "sift1M") {
        K = 40, L = 50;    // sift1M
    }else if (dataset == "gist") {
        K = 40, L = 40;    // gist
    }else if (dataset == "glove-100") {
        K = 40, L = 40;    // glove
    }else if (dataset == "audio") {
        K = 40, L = 40;    // audio
    }else if (dataset == "crawl") {
        K = 40, L = 70;    // crawl
    }else if (dataset == "msong") {
        K = 40, L = 70;    // msong
    }else if (dataset == "uqv") {
        K = 40, L = 60;    // uqv
    }else if (dataset == "enron") {
        K = 40, L = 40;    // enron
    }else if (dataset == "mnist") {
        K = 40, L = 40;    // mnist
    }else if (dataset == "c_1") {
        K = 40, L = 80;    // c_1
    }else if (dataset == "c_10") {
        K = 50, L = 80;    // c_10
    }else if (dataset == "c_100") {
        K = 60, L = 80;    // c_100
    }else if (dataset == "d_8") {
        K = 80, L = 110;    // d_8
    }else if (dataset == "d_128") {
        K = 50, L = 90;    // d_128
    }else if (dataset == "n_10000") {
        K = 40, L = 60;    // n_10000
    }else if (dataset == "n_1000000") {
        K = 40, L = 50;    // n_1000000
    }else if (dataset == "s_1") {
        K = 50, L = 80;    // s_1
    }else if (dataset == "s_10") {
        K = 60, L = 80;    // s_10
    }else {
        std::cout << "dataset error!\n";
        exit(-1);
    }
    parameters.set<unsigned>("NN", K);          // K
    parameters.set<unsigned>("ef_construction", L);        //L
}

void ONNG_PARA(std::string dataset, weavess::Parameters &parameters) {
    unsigned K, L, numOfOutgoingEdges, numOfIngoingEdges;
    if (dataset == "siftsmall") {
        K = 10, L = 50, numOfOutgoingEdges = 20, numOfIngoingEdges = 50;    // siftsmall
    }else if (dataset == "sift1M") {
        K = 100, L = 110, numOfOutgoingEdges = 10, numOfIngoingEdges = 100;    // sift1M
    }else if (dataset == "gist") {
        K = 100, L = 120, numOfOutgoingEdges = 20, numOfIngoingEdges = 100;    // gist
    }else if (dataset == "glove-100") {
        K = 400, L = 430, numOfOutgoingEdges = 80, numOfIngoingEdges = 100;    // glove
    }else if (dataset == "audio") {
        K = 100, L = 120, numOfOutgoingEdges = 10, numOfIngoingEdges = 100;    // audio
    }else if (dataset == "crawl") {
        K = 200, L = 220, numOfOutgoingEdges = 10, numOfIngoingEdges = 100;    // crawl
    }else if (dataset == "msong") {
        K = 100, L = 120, numOfOutgoingEdges = 20, numOfIngoingEdges = 100;    // msong
    }else if (dataset == "uqv") {
        K = 100, L = 120, numOfOutgoingEdges = 10, numOfIngoingEdges = 100;    // uqv
    }else if (dataset == "enron") {
        K = 100, L = 100, numOfOutgoingEdges = 10, numOfIngoingEdges = 100;    // enron
    }else if (dataset == "mnist") {
        K = 40, L = 40, numOfOutgoingEdges = 20, numOfIngoingEdges = 100;    // mnist
    }else if (dataset == "c_1") {
        K = 100, L = 100, numOfOutgoingEdges = 20, numOfIngoingEdges = 100;    // c_1
    }else if (dataset == "c_10") {
        K = 100, L = 120, numOfOutgoingEdges = 10, numOfIngoingEdges = 100;    // c_10
    }else if (dataset == "c_100") {
        K = 100, L = 100, numOfOutgoingEdges = 10, numOfIngoingEdges = 100;    // c_100
    }else if (dataset == "d_8") {
        K = 100, L = 110, numOfOutgoingEdges = 10, numOfIngoingEdges = 100;    // d_8
    }else if (dataset == "d_128") {
        K = 100, L = 100, numOfOutgoingEdges = 20, numOfIngoingEdges = 100;    // d_128
    }else if (dataset == "n_10000") {
        K = 100, L = 120, numOfOutgoingEdges = 10, numOfIngoingEdges = 100;    // n_10000
    }else if (dataset == "n_1000000") {
        K = 100, L = 120, numOfOutgoingEdges = 10, numOfIngoingEdges = 100;    // n_1000000
    }else if (dataset == "s_1") {
        K = 100, L = 120, numOfOutgoingEdges = 10, numOfIngoingEdges = 100;    // s_1
    }else if (dataset == "s_10") {
        K = 100, L = 110, numOfOutgoingEdges = 10, numOfIngoingEdges = 100;    // s_10
    }else {
        std::cout << "dataset error!\n";
        exit(-1);
    }
    parameters.set<unsigned>("NN", K);          // K
    parameters.set<unsigned>("ef_construction", L);        //L
    parameters.set<unsigned>("numOfOutgoingEdges", numOfOutgoingEdges);
    parameters.set<unsigned>("numOfIncomingEdges", numOfIngoingEdges);
    parameters.set<unsigned>("numOfQueries", 200);
    parameters.set<unsigned>("numOfResultantObjects", 20);
}

void KDRG_PARA(std::string dataset, weavess::Parameters &parameters) {
    unsigned K = 50;
    parameters.set<unsigned>("S", K);
    parameters.set<unsigned>("R_refine", K);
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
    if (parameters.get<std::string>("exc_type") != "build") {
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
    }else if (alg == "sptag_kdt") {
        SPTAG_KDT_PARA(dataset, parameters);
    }else if (alg == "sptag_bkt") {
        SPTAG_BKT_PARA(dataset, parameters);
    }else if (alg == "hnsw") {
        HNSW_PARA(dataset, parameters);
    }else if (alg == "ieh") {
        IEH_PARA(dataset, parameters);
    }else if (alg == "panng") {
        PANNG_PARA(dataset, parameters);
    }else if (alg == "onng") {
        ONNG_PARA(dataset, parameters);
    }else if (alg == "kdrg") {
        KDRG_PARA(dataset, parameters);
    }
    else {
        std::cout << "algorithm input error!\n";
        exit(-1);
    }
}
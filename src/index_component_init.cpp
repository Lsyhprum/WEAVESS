//
// Created by Murph on 2020/8/24.
//
#include "weavess/index_builder.h"

namespace weavess {
    // RAND
    void IndexComponentCoarseKDT::CoarseInner() {
        std::cout << "base data num : " << index_->n_ << std::endl;
        std::cout << "query data num : " << index_->query_num_ << std::endl;
        std::cout << "ground_data num : " << index_->ground_num_ << std::endl;
        std::cout << index_->param_.ToString() << std::endl;

        const unsigned L = index_->param_.get<unsigned>("L");
        const unsigned S = index_->param_.get<unsigned>("S");

        index_->graph_.reserve(index_->n_);
        std::mt19937 rng(rand());

        // 初始化 graph
        for (unsigned i = 0; i < index_->n_; i++) {
            index_->graph_.push_back(nhood(L, S, rng, (unsigned) index_->n_));
        }

#pragma omp parallel for
        // 生成随机入口点，即局部连接中初始邻居nn_new, 数量为 S
        for (unsigned i = 0; i < index_->n_; i++) {
            std::vector<unsigned> tmp(S + 1);

            weavess::GenRandom(rng, tmp.data(), S + 1, index_->n_);

            for (unsigned j = 0; j < S; j++) {
                unsigned id = tmp[j];

                if (id == i)continue;
                float dist = index_->distance_->compare(index_->data_ + i * index_->dim_,
                                                        index_->data_ + id * index_->dim_,
                                                        (unsigned) index_->dim_);

                index_->graph_[i].pool.emplace_back(id, dist, true);
            }
            std::make_heap(index_->graph_[i].pool.begin(), index_->graph_[i].pool.end());
            index_->graph_[i].pool.reserve(L);
        }
    }

    // HASH
    void IndexComponentInitHash::InitInner() {
        std::cout << index_->n_ << std::endl;
        std::cout << index_->param_.ToString() << std::endl;

        verify();

        buildIndexImpl();
    }

    void IndexComponentInitHash::verify() {
        std::cout<<"HASHING initial, max code length : 64" <<std::endl;

        index_->codelength = index_->param_.get<int>("codelen");
        std::cout << "use  "<< index_->codelength << " bit code"<< std::endl;

        index_->codelengthshift = index_->param_.get<int>("lenshift");
        int actuallen = index_->codelength - index_->codelengthshift;
        if(actuallen > 0){
            std::cout << "Actually use  "<< actuallen<< " bit code"<< std::endl;
        }else{
            std::cout << "lenShift error: could not be larger than the code length!  "<<  std::endl;
        }

        index_->tablenum = index_->param_.get<int>("tablenum");
        std::cout << "use  "<< index_->tablenum << " hashtables"<< std::endl;

        index_->upbits = index_->param_.get<int>("upbits");
        std::cout << "use upper "<< index_->upbits << " bits as first level index of hashtable"<< std::endl;
        std::cout << "use lower "<< index_->codelength - index_->codelengthshift - index_->upbits << " bits as second level index of hashtable"<< std::endl;

        if(index_->upbits >= index_->codelength-index_->codelengthshift){
            std::cout << "upbits should be smaller than the actual codelength!" << std::endl;
            return;
        }

        actuallen = index_->codelength - index_->codelengthshift;
        index_->radius = index_->param_.get<int>("radius");
        if(actuallen<=32){
            if(index_->radius > 13){
                std::cout << "radius greater than 13 not supported yet!" << std::endl;
                index_->radius = 13;
            }
        }else if(actuallen<=36){
            if(index_->radius > 11){
                std::cout << "radius greater than 11 not supported yet!" << std::endl;
                index_->radius = 11;
            }
        }else if(actuallen<=40){
            if(index_->radius > 10){
                std::cout << "radius greater than 10 not supported yet!" << std::endl;
                index_->radius = 10;
            }
        }else if(actuallen<=48){
            if(index_->radius > 9){
                std::cout << "radius greater than 9 not supported yet!" << std::endl;
                index_->radius = 9;
            }
        }else if(actuallen<=60){
            if(index_->radius > 8){
                std::cout << "radius greater than 8 not supported yet!" << std::endl;
                index_->radius = 8;
            }
        }else{ //actuallen<=64
            if(index_->radius > 7){
                std::cout << "radius greater than 7 not supported yet!" << std::endl;
                index_->radius = 7;
            }
        }
        std::cout << "search hamming radius "<<index_->radius<< std::endl;

//        std::string fpath = index_->param_.get<std::string>("bcfile");
//        std::string str(fpath);
//        std::cout << "Loading base code from " << str << std::endl;
//
//        if (index_->codelength <= 32 ){
//            LoadCode32(fpath, index_->BaseCode);
//        }else if(index_->codelength <= 64 ){
//            LoadCode64(fpath, index_->BaseCode64);
//        }else{
//            std::cout<<"code length not supported yet!"<<std::endl;
//        }

        std::cout << "code length is "<<index_->codelength<<std::endl;

//        fpath = index_->param_.get<std::string>("qcfile");
//        std::cout << "Loading query code from " << str << std::endl;
//
//        if (index_->codelength <= 32 ){
//            LoadCode32(fpath, index_->QueryCode);
//        }else if(index_->codelength <= 64 ){
//            LoadCode64(fpath, index_->QueryCode64);
//        }else{
//            std::cout<<"code length not supported yet!"<<std::endl;
//        }

        std::cout << "code length is "<<index_->codelength<<std::endl;
    }

    void IndexComponentInitHash::LoadCode32(char* filename, std::vector<std::vector<unsigned int>>& baseAll){
        if (index_->tablenum < 1){
            std::cout<<"Total hash table num error! "<<std::endl;
        }

        int actuallen = index_->codelength-index_->codelengthshift;

        unsigned int  maxValue = 1;
        for(int i=1;i<actuallen;i++){
            maxValue = maxValue << 1;
            maxValue ++;
        }


        std::stringstream ss;
        for(int j = 0; j < index_->tablenum; j++){

            ss << filename << "_" << j+1 ;
            std::string codeFile;
            ss >> codeFile;
            ss.clear();

            std::ifstream in(codeFile.c_str(), std::ios::binary);
            if(!in.is_open()){std::cout<<"open file " << filename <<" error"<< std::endl;return;}

            int codeNum;
            in.read((char*)&codeNum,4);
            if (codeNum != 1){
                std::cout<<"Codefile  "<< j << " error!"<<std::endl;
            }

            in.read((char*)&index_->codelength,4);
            //std::cout<<"codelength: "<<codelength<<std::endl;

            int num;
            in.read((char*)&num,4);
            //std::cout<<"ponit num: "<<num<<std::endl;

            std::vector<unsigned int> base;
            for(int i = 0; i < num; i++){
                unsigned int codetmp;
                in.read((char*)&codetmp,4);
                codetmp = codetmp >> index_->codelengthshift;
                if (codetmp > maxValue){
                    std::cout<<"codetmp: "<< codetmp <<std::endl;
                    std::cout<<"codelengthshift: "<<index_->codelengthshift<<std::endl;
                    std::cout<<"codefile  "<< codeFile << " error! Exceed maximum value"<<std::endl;
                    in.close();
                    return;
                }
                base.push_back(codetmp);
            }
            baseAll.push_back(base);

            in.close();
        }
    }

    void IndexComponentInitHash::LoadCode64(char* filename, std::vector<std::vector<unsigned long>>& baseAll){
        if (index_->tablenum < 1){
            std::cout<<"Total hash table num error! "<<std::endl;
        }

        int actuallen = index_->codelength-index_->codelengthshift;

        unsigned long  maxValue = 1;
        for(int i=1;i<actuallen;i++){
            maxValue = maxValue << 1;
            maxValue ++;
        }


        std::stringstream ss;
        for(int j = 0; j < index_->tablenum; j++){

            ss << filename << "_" << j+1 ;
            std::string codeFile;
            ss >> codeFile;
            ss.clear();

            std::ifstream in(codeFile.c_str(), std::ios::binary);
            if(!in.is_open()){std::cout<<"open file " << filename <<" error"<< std::endl;return;}

            int codeNum;
            in.read((char*)&codeNum,4);
            if (codeNum != 1){
                std::cout<<"Codefile  "<< j << " error!"<<std::endl;
            }

            in.read((char*)&index_->codelength,4);
            //std::cout<<"codelength: "<<codelength<<std::endl;

            int num;
            in.read((char*)&num,4);
            //std::cout<<"ponit num: "<<num<<std::endl;

            std::vector<unsigned long> base;
            for(int i = 0; i < num; i++){
                unsigned long codetmp;
                in.read((char*)&codetmp,8);
                codetmp = codetmp >> index_->codelengthshift;
                if (codetmp > maxValue){
                    std::cout<<"codetmp: "<< codetmp <<std::endl;
                    std::cout<<"codelengthshift: "<<index_->codelengthshift<<std::endl;
                    std::cout<<"codefile  "<< codeFile << " error! Exceed maximum value"<<std::endl;
                    in.close();
                    return;
                }

                base.push_back(codetmp);
            }
            baseAll.push_back(base);

            in.close();
        }
    }

    void IndexComponentInitHash::buildIndexImpl()
    {
        std::cout<<"HASHING building hashing table"<<std::endl;

        if (index_->codelength <= 32 ){
            index_->codelength = index_->codelength - index_->codelengthshift;
            BuildHashTable32(index_->upbits, index_->codelength-index_->upbits, index_->BaseCode ,index_->htb);
            generateMask32();
        }else if(index_->codelength <= 64 ){
            index_->codelength = index_->codelength - index_->codelengthshift;
            BuildHashTable64(index_->upbits, index_->codelength-index_->upbits, index_->BaseCode64 ,index_->htb64);
            generateMask64();
        }else{
            std::cout<<"code length not supported yet!"<<std::endl;
        }
    }

    void IndexComponentInitHash::generateMask32(){
        //i = 0 means the origin code
        index_->HammingBallMask.push_back(0);
        index_->HammingRadius.push_back(index_->HammingBallMask.size());

        if(index_->radius>0){
            //radius 1
            for(int i = 0; i < index_->codelength; i++){
                unsigned int mask = 1 << i;
                index_->HammingBallMask.push_back(mask);
            }
            index_->HammingRadius.push_back(index_->HammingBallMask.size());
        }

        if(index_->radius>1){
            //radius 2
            for(int i = 0; i < index_->codelength; i++){
                for(int j = i+1; j < index_->codelength; j++){
                    unsigned int mask = (1<<i) | (1<<j);
                    index_->HammingBallMask.push_back(mask);
                }
            }
            index_->HammingRadius.push_back(index_->HammingBallMask.size());
        }

        if(index_->radius>2){
            //radius 3
            for(int i = 0; i < index_->codelength; i++){
                for(int j = i+1; j < index_->codelength; j++){
                    for(int k = j+1; k < index_->codelength; k++){
                        unsigned int mask = (1<<i) | (1<<j) | (1<<k);
                        index_->HammingBallMask.push_back(mask);
                    }
                }
            }
            index_->HammingRadius.push_back(index_->HammingBallMask.size());
        }

        if(index_->radius>3){
            //radius 4
            for(int i = 0; i < index_->codelength; i++){
                for(int j = i+1; j < index_->codelength; j++){
                    for(int k = j+1; k < index_->codelength; k++){
                        for(int a = k+1; a < index_->codelength; a++){
                            unsigned int mask = (1<<i) | (1<<j) | (1<<k)| (1<<a);
                            index_->HammingBallMask.push_back(mask);
                        }
                    }
                }
            }
            index_->HammingRadius.push_back(index_->HammingBallMask.size());
        }

        if(index_->radius>4){
            //radius 5
            for(int i = 0; i < index_->codelength; i++){
                for(int j = i+1; j < index_->codelength; j++){
                    for(int k = j+1; k < index_->codelength; k++){
                        for(int a = k+1; a < index_->codelength; a++){
                            for(int b = a+1; b < index_->codelength; b++){
                                unsigned int mask = (1<<i) | (1<<j) | (1<<k)| (1<<a)| (1<<b);
                                index_->HammingBallMask.push_back(mask);
                            }
                        }
                    }
                }
            }
            index_->HammingRadius.push_back(index_->HammingBallMask.size());
        }

        if(index_->radius>5){
            //radius 6
            for(int i = 0; i < index_->codelength; i++){
                for(int j = i+1; j < index_->codelength; j++){
                    for(int k = j+1; k < index_->codelength; k++){
                        for(int a = k+1; a < index_->codelength; a++){
                            for(int b = a+1; b < index_->codelength; b++){
                                for(int c = b+1; c < index_->codelength; c++){
                                    unsigned int mask = (1<<i) | (1<<j) | (1<<k)| (1<<a)| (1<<b)| (1<<c);
                                    index_->HammingBallMask.push_back(mask);
                                }
                            }
                        }
                    }
                }
            }
            index_->HammingRadius.push_back(index_->HammingBallMask.size());
        }

        if(index_->radius>6){
            //radius 7
            for(int i = 0; i < index_->codelength; i++){
                for(int j = i+1; j < index_->codelength; j++){
                    for(int k = j+1; k < index_->codelength; k++){
                        for(int a = k+1; a < index_->codelength; a++){
                            for(int b = a+1; b < index_->codelength; b++){
                                for(int c = b+1; c < index_->codelength; c++){
                                    for(int d = c+1; d < index_->codelength; d++){
                                        unsigned int mask = (1<<i) | (1<<j) | (1<<k)| (1<<a)| (1<<b)| (1<<c)| (1<<d);
                                        index_->HammingBallMask.push_back(mask);
                                    }
                                }
                            }
                        }
                    }
                }
            }
            index_->HammingRadius.push_back(index_->HammingBallMask.size());
        }

        if(index_->radius>7){
            //radius 8
            for(int i = 0; i < index_->codelength; i++){
                for(int j = i+1; j < index_->codelength; j++){
                    for(int k = j+1; k < index_->codelength; k++){
                        for(int a = k+1; a < index_->codelength; a++){
                            for(int b = a+1; b < index_->codelength; b++){
                                for(int c = b+1; c < index_->codelength; c++){
                                    for(int d = c+1; d < index_->codelength; d++){
                                        for(int e = d+1; e < index_->codelength; e++){
                                            unsigned int mask = (1<<i) | (1<<j) | (1<<k)| (1<<a)| (1<<b)| (1<<c)| (1<<d)| (1<<e);
                                            index_->HammingBallMask.push_back(mask);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            index_->HammingRadius.push_back(index_->HammingBallMask.size());
        }

        if(index_->radius>8){
            //radius 9
            for(int i = 0; i < index_->codelength; i++){
                for(int j = i+1; j < index_->codelength; j++){
                    for(int k = j+1; k < index_->codelength; k++){
                        for(int a = k+1; a < index_->codelength; a++){
                            for(int b = a+1; b < index_->codelength; b++){
                                for(int c = b+1; c < index_->codelength; c++){
                                    for(int d = c+1; d < index_->codelength; d++){
                                        for(int e = d+1; e < index_->codelength; e++){
                                            for(int f = e+1; f < index_->codelength; f++){
                                                unsigned int mask = (1<<i) | (1<<j) | (1<<k)| (1<<a)| (1<<b)| (1<<c)| (1<<d)| (1<<e)| (1<<f);
                                                index_->HammingBallMask.push_back(mask);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            index_->HammingRadius.push_back(index_->HammingBallMask.size());
        }

        if(index_->radius>9){
            //radius 10
            for(int i = 0; i < index_->codelength; i++){
                for(int j = i+1; j < index_->codelength; j++){
                    for(int k = j+1; k < index_->codelength; k++){
                        for(int a = k+1; a < index_->codelength; a++){
                            for(int b = a+1; b < index_->codelength; b++){
                                for(int c = b+1; c < index_->codelength; c++){
                                    for(int d = c+1; d < index_->codelength; d++){
                                        for(int e = d+1; e < index_->codelength; e++){
                                            for(int f = e+1; f < index_->codelength; f++){
                                                for(int g = f+1; g < index_->codelength; g++){
                                                    unsigned int mask = (1<<i) | (1<<j) | (1<<k)| (1<<a)| (1<<b)| (1<<c)| (1<<d)| (1<<e)| (1<<f)| (1<<g);
                                                    index_->HammingBallMask.push_back(mask);
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            index_->HammingRadius.push_back(index_->HammingBallMask.size());
        }

        if(index_->radius>10){
            //radius 11
            for(int i = 0; i < index_->codelength; i++){
                for(int j = i+1; j < index_->codelength; j++){
                    for(int k = j+1; k < index_->codelength; k++){
                        for(int a = k+1; a < index_->codelength; a++){
                            for(int b = a+1; b < index_->codelength; b++){
                                for(int c = b+1; c < index_->codelength; c++){
                                    for(int d = c+1; d < index_->codelength; d++){
                                        for(int e = d+1; e < index_->codelength; e++){
                                            for(int f = e+1; f < index_->codelength; f++){
                                                for(int g = f+1; g < index_->codelength; g++){
                                                    for(int h = g+1; h < index_->codelength; h++){
                                                        unsigned int mask = (1<<i) | (1<<j) | (1<<k)| (1<<a)| (1<<b)| (1<<c)| (1<<d)| (1<<e)| (1<<f)| (1<<g)| (1<<h);
                                                        index_->HammingBallMask.push_back(mask);
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            index_->HammingRadius.push_back(index_->HammingBallMask.size());
        }

        if(index_->radius>11){
            //radius 12
            for(int i = 0; i < index_->codelength; i++){
                for(int j = i+1; j < index_->codelength; j++){
                    for(int k = j+1; k < index_->codelength; k++){
                        for(int a = k+1; a < index_->codelength; a++){
                            for(int b = a+1; b < index_->codelength; b++){
                                for(int c = b+1; c < index_->codelength; c++){
                                    for(int d = c+1; d < index_->codelength; d++){
                                        for(int e = d+1; e < index_->codelength; e++){
                                            for(int f = e+1; f < index_->codelength; f++){
                                                for(int g = f+1; g < index_->codelength; g++){
                                                    for(int h = g+1; h < index_->codelength; h++){
                                                        for(int l = h+1; h < index_->codelength; l++){
                                                            unsigned int mask = (1<<i) | (1<<j) | (1<<k)| (1<<a)| (1<<b)| (1<<c)| (1<<d)| (1<<e)| (1<<f)| (1<<g)| (1<<h)| (1<<l);
                                                            index_->HammingBallMask.push_back(mask);
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            index_->HammingRadius.push_back(index_->HammingBallMask.size());
        }

        if(index_->radius>12){
            //radius 13
            for(int i = 0; i < index_->codelength; i++){
                for(int j = i+1; j < index_->codelength; j++){
                    for(int k = j+1; k < index_->codelength; k++){
                        for(int a = k+1; a < index_->codelength; a++){
                            for(int b = a+1; b < index_->codelength; b++){
                                for(int c = b+1; c < index_->codelength; c++){
                                    for(int d = c+1; d < index_->codelength; d++){
                                        for(int e = d+1; e < index_->codelength; e++){
                                            for(int f = e+1; f < index_->codelength; f++){
                                                for(int g = f+1; g < index_->codelength; g++){
                                                    for(int h = g+1; h < index_->codelength; h++){
                                                        for(int l = h+1; h < index_->codelength; l++){
                                                            for(int m = l+1; m < index_->codelength; m++){
                                                                unsigned int mask = (1<<i) | (1<<j) | (1<<k)| (1<<a)| (1<<b)| (1<<c)| (1<<d)| (1<<e)| (1<<f)| (1<<g)| (1<<h)| (1<<l)| (1<<m);
                                                                index_->HammingBallMask.push_back(mask);
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            index_->HammingRadius.push_back(index_->HammingBallMask.size());
        }

    }

    void IndexComponentInitHash::generateMask64(){
        //i = 0 means the origin code
        index_->HammingBallMask64.push_back(0);
        index_->HammingRadius.push_back(index_->HammingBallMask64.size());

        unsigned long One = 1;
        if(index_->radius>0){
            //radius 1
            for(int i = 0; i < index_->codelength; i++){
                unsigned long mask = One << i;
                index_->HammingBallMask64.push_back(mask);
            }
            index_->HammingRadius.push_back(index_->HammingBallMask64.size());
        }

        if(index_->radius>1){
            //radius 2
            for(int i = 0; i < index_->codelength; i++){
                for(int j = i+1; j < index_->codelength; j++){
                    unsigned long mask = (One<<i) | (One<<j);
                    index_->HammingBallMask64.push_back(mask);
                }
            }
            index_->HammingRadius.push_back(index_->HammingBallMask64.size());
        }

        if(index_->radius>2){
            //radius 3
            for(int i = 0; i < index_->codelength; i++){
                for(int j = i+1; j < index_->codelength; j++){
                    for(int k = j+1; k < index_->codelength; k++){
                        unsigned long mask = (One<<i) | (One<<j) | (One<<k);
                        index_->HammingBallMask64.push_back(mask);
                    }
                }
            }
            index_->HammingRadius.push_back(index_->HammingBallMask64.size());
        }

        if(index_->radius>3){
            //radius 4
            for(int i = 0; i < index_->codelength; i++){
                for(int j = i+1; j < index_->codelength; j++){
                    for(int k = j+1; k < index_->codelength; k++){
                        for(int a = k+1; a < index_->codelength; a++){
                            unsigned long mask = (One<<i) | (One<<j) | (One<<k)| (One<<a);
                            index_->HammingBallMask64.push_back(mask);
                        }
                    }
                }
            }
            index_->HammingRadius.push_back(index_->HammingBallMask64.size());
        }

        if(index_->radius>4){
            //radius 5
            for(int i = 0; i < index_->codelength; i++){
                for(int j = i+1; j < index_->codelength; j++){
                    for(int k = j+1; k < index_->codelength; k++){
                        for(int a = k+1; a < index_->codelength; a++){
                            for(int b = a+1; b < index_->codelength; b++){
                                unsigned long mask = (One<<i) | (One<<j) | (One<<k)| (One<<a)| (One<<b);
                                index_->HammingBallMask64.push_back(mask);
                            }
                        }
                    }
                }
            }
            index_->HammingRadius.push_back(index_->HammingBallMask64.size());
        }

        if(index_->radius>5){
            //radius 6
            for(int i = 0; i < index_->codelength; i++){
                for(int j = i+1; j < index_->codelength; j++){
                    for(int k = j+1; k < index_->codelength; k++){
                        for(int a = k+1; a < index_->codelength; a++){
                            for(int b = a+1; b < index_->codelength; b++){
                                for(int c = b+1; c < index_->codelength; c++){
                                    unsigned long mask = (One<<i) | (One<<j) | (One<<k)| (One<<a)| (One<<b)| (One<<c);
                                    index_->HammingBallMask64.push_back(mask);
                                }
                            }
                        }
                    }
                }
            }
            index_->HammingRadius.push_back(index_->HammingBallMask64.size());
        }

        if(index_->radius>6){
            //radius 7
            for(int i = 0; i < index_->codelength; i++){
                for(int j = i+1; j < index_->codelength; j++){
                    for(int k = j+1; k < index_->codelength; k++){
                        for(int a = k+1; a < index_->codelength; a++){
                            for(int b = a+1; b < index_->codelength; b++){
                                for(int c = b+1; c < index_->codelength; c++){
                                    for(int d = c+1; d < index_->codelength; d++){
                                        unsigned long mask = (One<<i) | (One<<j) | (One<<k)| (One<<a)| (One<<b)| (One<<c)| (One<<d);
                                        index_->HammingBallMask64.push_back(mask);
                                    }
                                }
                            }
                        }
                    }
                }
            }
            index_->HammingRadius.push_back(index_->HammingBallMask64.size());
        }

        if(index_->radius>7){
            //radius 8
            for(int i = 0; i < index_->codelength; i++){
                for(int j = i+1; j < index_->codelength; j++){
                    for(int k = j+1; k < index_->codelength; k++){
                        for(int a = k+1; a < index_->codelength; a++){
                            for(int b = a+1; b < index_->codelength; b++){
                                for(int c = b+1; c < index_->codelength; c++){
                                    for(int d = c+1; d < index_->codelength; d++){
                                        for(int e = d+1; e < index_->codelength; e++){
                                            unsigned long mask = (One<<i) | (One<<j) | (One<<k)| (One<<a)| (One<<b)| (One<<c)| (One<<d)| (One<<e);
                                            index_->HammingBallMask64.push_back(mask);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            index_->HammingRadius.push_back(index_->HammingBallMask64.size());
        }

        if(index_->radius>8){
            //radius 9
            for(int i = 0; i < index_->codelength; i++){
                for(int j = i+1; j < index_->codelength; j++){
                    for(int k = j+1; k < index_->codelength; k++){
                        for(int a = k+1; a < index_->codelength; a++){
                            for(int b = a+1; b < index_->codelength; b++){
                                for(int c = b+1; c < index_->codelength; c++){
                                    for(int d = c+1; d < index_->codelength; d++){
                                        for(int e = d+1; e < index_->codelength; e++){
                                            for(int f = e+1; f < index_->codelength; f++){
                                                unsigned long mask = (One<<i) | (One<<j) | (One<<k)| (One<<a)| (One<<b)| (One<<c)| (One<<d)| (One<<e)| (One<<f);
                                                index_->HammingBallMask64.push_back(mask);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            index_->HammingRadius.push_back(index_->HammingBallMask64.size());
        }

        if(index_->radius>9){
            //radius 10
            for(int i = 0; i < index_->codelength; i++){
                for(int j = i+1; j < index_->codelength; j++){
                    for(int k = j+1; k < index_->codelength; k++){
                        for(int a = k+1; a < index_->codelength; a++){
                            for(int b = a+1; b < index_->codelength; b++){
                                for(int c = b+1; c < index_->codelength; c++){
                                    for(int d = c+1; d < index_->codelength; d++){
                                        for(int e = d+1; e < index_->codelength; e++){
                                            for(int f = e+1; f < index_->codelength; f++){
                                                for(int g = f+1; g < index_->codelength; g++){
                                                    unsigned long mask = (One<<i) | (One<<j) | (One<<k)| (One<<a)| (One<<b)| (One<<c)| (One<<d)| (One<<e)| (One<<f)| (One<<g);
                                                    index_->HammingBallMask64.push_back(mask);
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            index_->HammingRadius.push_back(index_->HammingBallMask64.size());
        }

        if(index_->radius>10){
            //radius 11
            for(int i = 0; i < index_->codelength; i++){
                for(int j = i+1; j < index_->codelength; j++){
                    for(int k = j+1; k < index_->codelength; k++){
                        for(int a = k+1; a < index_->codelength; a++){
                            for(int b = a+1; b < index_->codelength; b++){
                                for(int c = b+1; c < index_->codelength; c++){
                                    for(int d = c+1; d < index_->codelength; d++){
                                        for(int e = d+1; e < index_->codelength; e++){
                                            for(int f = e+1; f < index_->codelength; f++){
                                                for(int g = f+1; g < index_->codelength; g++){
                                                    for(int h = g+1; h < index_->codelength; h++){
                                                        unsigned long mask = (One<<i) | (One<<j) | (One<<k)| (One<<a)| (One<<b)| (One<<c)| (One<<d)| (One<<e)| (One<<f)| (One<<g)| (One<<h);
                                                        index_->HammingBallMask64.push_back(mask);
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            index_->HammingRadius.push_back(index_->HammingBallMask64.size());
        }

    }

    void IndexComponentInitHash::BuildHashTable32(int upbits, int lowbits, std::vector<Codes>& baseAll ,std::vector<HashTable>& tbAll){

        for(size_t h=0; h < baseAll.size(); h++){
            Codes& base = baseAll[h];

            HashTable tb;
            for(int i = 0; i < (1 << upbits); i++){
                HashBucket emptyBucket;
                tb.push_back(emptyBucket);
            }

            for(size_t i = 0; i < base.size(); i ++){
                unsigned int idx1 = base[i] >> lowbits;
                unsigned int idx2 = base[i] - (idx1 << lowbits);
                if(tb[idx1].find(idx2) != tb[idx1].end()){
                    tb[idx1][idx2].push_back(i);
                }else{
                    std::vector<unsigned int> v;
                    v.push_back(i);
                    tb[idx1].insert(make_pair(idx2,v));
                }
            }
            tbAll.push_back(tb);
        }
    }

    void IndexComponentInitHash::BuildHashTable64(int upbits, int lowbits, std::vector<Codes64>& baseAll ,std::vector<HashTable64>& tbAll){

        for(size_t h=0; h < baseAll.size(); h++){
            Codes64& base = baseAll[h];

            HashTable64 tb;
            for(int i = 0; i < (1 << upbits); i++){
                HashBucket64 emptyBucket;
                tb.push_back(emptyBucket);
            }

            for(size_t i = 0; i < base.size(); i ++){
                unsigned int idx1 = base[i] >> lowbits;
                unsigned long idx2 = base[i] - ((unsigned long)idx1 << lowbits);
                if(tb[idx1].find(idx2) != tb[idx1].end()){
                    tb[idx1][idx2].push_back(i);
                }else{
                    std::vector<unsigned int> v;
                    v.push_back(i);
                    tb[idx1].insert(make_pair(idx2,v));
                }
            }
            tbAll.push_back(tb);
        }
    }




}
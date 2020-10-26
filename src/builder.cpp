//
// Created by MurphySL on 2020/10/23.
//

#include "weavess/builder.h"
#include "weavess/component.h"
//#include "weavess/matplotlibcpp.h"

//namespace plt = matplotlibcpp;

namespace weavess {

    /**
     * 加载数据集及参数
     * @param data_file *_base.fvecs
     * @param query_file *_query.fvecs
     * @param ground_file *_groundtruth.ivecs
     * @param parameters 构建参数
     * @return 当前建造者指针
     */
    IndexBuilder *IndexBuilder::load(char *data_file, char *query_file, char *ground_file, Parameters &parameters) {
        auto *a = new ComponentLoad(final_index_);
        a->LoadInner(data_file, query_file, ground_file, parameters);

        e = std::chrono::high_resolution_clock::now();

        std::cout << "base data len : " << final_index_->getBaseLen() << std::endl;
        std::cout << "base data dim : " << final_index_->getBaseDim() << std::endl;
        std::cout << "query data len : " << final_index_->getQueryLen() << std::endl;
        std::cout << "query data dim : " << final_index_->getQueryDim() << std::endl;
        std::cout << "ground truth data len : " << final_index_->getGroundLen() << std::endl;
        std::cout << "ground truth data dim : " << final_index_->getGroundDim() << std::endl;
        std::cout << "=====================" << std::endl;

        std::cout << final_index_->getParam().toString() << std::endl;
        std::cout << "=====================" << std::endl;

        return this;
    }

    /**
    * 构建初始图
    * @param type 初始化类型
    * @param debug 是否输出图索引相关信息（开启将对性能产生一定影响）
    * @return 当前建造者指针
    */
    IndexBuilder *IndexBuilder::init(TYPE type) {
        ComponentInit *a = nullptr;

        if (type == INIT_NN_DESCENT) {
            std::cout << "__INIT : NN-Descent__" << std::endl;
            a = new ComponentInitNNDescent(final_index_);
        } else if (type == INIT_RAND) {
            std::cout << "__INIT : RAND__" << std::endl;
            a = new ComponentInitRand(final_index_);
        } else if (type == INIT_KDT) {
            std::cout << "__INIT : KDT__" << std::endl;
            a = new ComponentInitKDT(final_index_);
        } else if (type == INIT_IEH) {
            std::cout << "__INIT : IEH__" << std::endl;
            a = new ComponentInitIEH(final_index_);
        } else if (type == INIT_NSW) {
            std::cout << "__INIT : NSW__" << std::endl;
            a = new ComponentInitNSW(final_index_);
        } else if (type == INIT_HNSW) {
            std::cout << "__INIT : HNSW__" << std::endl;
            a = new ComponentInitHNSW(final_index_);
        } else if (type == INIT_ANNG) {
            std::cout << "__INIT : ANNG__" << std::endl;
            a = new ComponentInitANNG(final_index_);
        } else if (type == INIT_SPTAG_KDT) {
            final_index_->t.wtf();
        }

        else {
            std::cout << "__INIT : WRONG TYPE__" << std::endl;
            exit(-1);
        }

        a->InitInner();

        e = std::chrono::high_resolution_clock::now();

        std::cout << "__INIT FINISH__" << std::endl;

        return this;
    }

    /**
     * 构建改进图, 通常包含获取 入口点、获取候选点、裁边、增强连通性 四部分
     * @param type 改进类型
     * @param debug 是否输出图索引相关信息（开启将对性能产生一定影响）
     * @return 当前建造者指针
     */
    IndexBuilder *IndexBuilder::refine(TYPE type, bool debug) {
        ComponentRefine *a = nullptr;

        if (type == REFINE_NN_DESCENT) {
            std::cout << "__REFINE : KGRAPH__" << std::endl;
            a = new ComponentRefineNNDescent(final_index_);
        } else if (type == REFINE_NSG) {
            std::cout << "__REFINE : NSG__" << std::endl;
            a = new ComponentRefineNSG(final_index_);
        } else if (type == REFINE_SSG) {
            std::cout << "__REFINE : NSSG__" << std::endl;
            a = new ComponentRefineSSG(final_index_);
        } else if (type == REFINE_DPG) {
            std::cout << "__REFINE : DPG__" << std::endl;
            a = new ComponentRefineDPG(final_index_);
        } else if (type == REFINE_VAMANA) {
            std::cout << "__REFINE : VAMANA__" << std::endl;
            a = new ComponentRefineVAMANA(final_index_);
        } else if (type == REFINE_EFANNA) {
            std::cout << "__REFINE : EFANNA__" << std::endl;
            a = new ComponentRefineEFANNA(final_index_);
        } else {
            std::cerr << "__REFINE : WRONG TYPE__" << std::endl;
        }

        a->RefineInner();

        if (debug) {
            // degree
            std::unordered_map<unsigned, unsigned> degree;
            degree_info(degree);

            // 连通分量
            conn_info();
        }

        std::cout << "__REFINE : FINISH__" << std::endl;
        std::cout << "===================" << std::endl;

        e = std::chrono::high_resolution_clock::now();

        return this;
    }

    /**
     * 离线搜索
     * @param entry_type 入口点策略
     * @param route_type 路由策略
     * @return 当前建造者指针
     */
    IndexBuilder *IndexBuilder::search(TYPE entry_type, TYPE route_type) {
        std::cout << "__SEARCH__" << std::endl;

        unsigned K = 10;
        unsigned L_start = K;
        unsigned L_end = 500;
        unsigned experiment_num = 10;
        unsigned LI = (L_end - L_start) / experiment_num;

        final_index_->getParam().set<unsigned>("K_search", K);

        std::vector<Index::Neighbor> pool;
        std::vector<std::vector<unsigned>> res;

        // ENTRY
        ComponentSearchEntry *a = nullptr;
        if (entry_type == SEARCH_ENTRY_RAND) {
            std::cout << "__SEARCH ENTRY : RAND__" << std::endl;
            a = new ComponentSearchEntryRand(final_index_);
        } else if (entry_type == SEARCH_ENTRY_CENTROID) {
            std::cout << "__SEARCH ENTRY : CENTROID__" << std::endl;
            a = new ComponentSearchEntryCentroid(final_index_);
        } else if (entry_type == SEARCH_ENTRY_SUB_CENTROID) {
            std::cout << "__SEARCH ENTRY : SUB_CENTROID__" << std::endl;
            a = new ComponentSearchEntrySubCentroid(final_index_);
        } else if (entry_type == SEARCH_ENTRY_KDT) {
            std::cout << "__SEARCH ENTRY : KDT__" << std::endl;
            a = new ComponentSearchEntryKDT(final_index_);
        } else if (entry_type == SEARCH_ENTRY_NONE) {
            std::cout << "__SEARCH ENTRY : NONE__" << std::endl;
            a = new ComponentSearchEntryNone(final_index_);
        } else {
            std::cerr << "__SEARCH ENTRY : WRONG TYPE__" << std::endl;
            exit(-1);
        }

        // ROUTE
        ComponentSearchRoute *b = nullptr;
        if (route_type == ROUTER_GREEDY) {
            std::cout << "__ROUTER : GREEDY__" << std::endl;
            b = new ComponentSearchRouteGreedy(final_index_);
        } else if (route_type == ROUTER_NSW) {
            std::cout << "__ROUTER : NSW__" << std::endl;
            b = new ComponentSearchRouteNSW(final_index_);
        } else if (route_type == ROUTER_HNSW) {
            std::cout << "__ROUTER : NSW__" << std::endl;
            b = new ComponentSearchRouteHNSW(final_index_);
        } else {
            std::cerr << "__ROUTER : WRONG TYPE__" << std::endl;
            exit(-1);
        }

        for (unsigned L = L_start; L < L_end; L += LI) {
            std::cout << "SEARCH_L : " << L << std::endl;
            if (L < K) {
                std::cout << "search_L cannot be smaller than search_K! " << std::endl;
                exit(-1);
            }

            final_index_->getParam().set<unsigned>("L_search", L);

            auto s1 = std::chrono::high_resolution_clock::now();

            res.clear();
            res.resize(final_index_->getBaseLen());

            for (unsigned i = 0; i < final_index_->getQueryLen(); i++) {
                pool.clear();

                a->SearchEntryInner(i, pool);

                std::vector<unsigned> tmp(K);
                b->RouteInner(i, pool, tmp);

                res[i] = tmp;
            }

            auto e1 = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> diff = e1 - s1;
            std::cout << "search time: " << diff.count() << "\n";

            //float speedup = (float)(index_->n_ * query_num) / (float)distcount;
            std::cout << "DistCount: " << final_index_->getDistCount() << std::endl;
            //结果评估
            int cnt = 0;
            for (unsigned i = 0; i < final_index_->getGroundLen(); i++) {
                for (unsigned j = 0; j < K; j++) {
                    unsigned k = 0;
                    for (; k < K; k++) {
                        if (res[i][j] == final_index_->getGroundData()[i * final_index_->getGroundDim() + k])
                            break;
                    }
                    if (k == K)
                        cnt++;
                }
            }

            float acc = 1 - (float) cnt / (final_index_->getGroundLen() * K);
            std::cout << K << " NN accuracy: " << acc << std::endl;
        }

        e = std::chrono::high_resolution_clock::now();
        std::cout << "__SEARCH FINISH__" << std::endl;

        return this;
    }

    /**
     * 输出图索引出度、入度信息
     * @param degree 出度分布
     */
    void IndexBuilder::degree_info(std::unordered_map<unsigned, unsigned> &degree) {
        unsigned max = 0, min = 1e6;
        double avg = 0.0;
        for (size_t i = 0; i < final_index_->getBaseLen(); i++) {
            auto size = final_index_->getFinalGraph()[i].size();
            degree[size]++;
            max = max < size ? size : max;
            min = min > size ? size : min;
            avg += size;
        }
        avg /= final_index_->getBaseLen();
        printf("Degree Statistics: Max = %d, Min = %d, Avg = %lf\n", max, min, avg);
    }

    /**
     * 连通性分析
     */
    void IndexBuilder::conn_info() {
        unsigned root = 0;
        boost::dynamic_bitset<> flags{final_index_->getBaseLen(), 0};

        unsigned conn = 0;
        unsigned unlinked_cnt = 0;

        while (unlinked_cnt < final_index_->getBaseLen()) {
            conn++;
            DFS(flags, root, unlinked_cnt);
            // std::cout << unlinked_cnt << '\n';
            if (unlinked_cnt >= final_index_->getBaseLen()) break;
            findRoot(flags, root);
        }
        printf("Conn Statistics: conn = %d\n", conn);
    }

    void IndexBuilder::findRoot(boost::dynamic_bitset<> &flag, unsigned &root) {
        unsigned id = final_index_->getBaseLen();
        for (unsigned i = 0; i < final_index_->getBaseLen(); i++) {
            if (!flag[i]) {
                id = i;
                break;
            }
        }
        root = id;
    }

    void IndexBuilder::DFS(boost::dynamic_bitset<> &flag, unsigned root, unsigned &cnt) {
        unsigned tmp = root;
        std::stack<unsigned> s;
        s.push(root);
        if (!flag[root]) cnt++;
        flag[root] = true;
        while (!s.empty()) {
            unsigned next = final_index_->getBaseLen() + 1;
            for (unsigned i = 0; i < final_index_->getFinalGraph()[tmp].size(); i++) {
                if (!flag[final_index_->getFinalGraph()[tmp][i].id]) {
                    next = final_index_->getFinalGraph()[tmp][i].id;
                    break;
                }
            }
            // std::cout << next <<":"<<cnt <<":"<<tmp <<":"<<s.size()<< '\n';
            if (next == (final_index_->getBaseLen() + 1)) {
                s.pop();
                if (s.empty()) break;
                tmp = s.top();
                continue;
            }
            tmp = next;
            flag[tmp] = true;
            s.push(tmp);
            cnt++;
        }
    }


}

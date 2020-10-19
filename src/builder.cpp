//
// Created by MurphySL on 2020/9/14.
//

#include "weavess/builder.h"
#include "weavess/component.h"
#include "weavess/matplotlibcpp.h"

namespace plt = matplotlibcpp;

namespace weavess {

    IndexBuilder *IndexBuilder::draw() {

        for (int i = 0; i < final_index_->getBaseLen(); i++) {
            for (int j = 0; j < final_index_->getFinalGraph()[i][0].size(); j++) {
                //std::cout << (final_index_->getBaseData() + 2 * final_index_->getFinalGraph()[i][0][j])[0] << " ";
                //if(i == 20 ) {
                std::vector<float> x{(final_index_->getBaseData() + 2 * i)[0],
                                     (final_index_->getBaseData() + 2 * final_index_->getFinalGraph()[i][0][j])[0]};

                std::vector<float> y{(final_index_->getBaseData() + 2 * i)[1],
                                     (final_index_->getBaseData() + 2 * final_index_->getFinalGraph()[i][0][j])[1]};
                plt::plot(x, y, "grey");
                //}
            }
            //std::cout << std::endl;
        }

        std::vector<float> x;
        std::vector<float> y;
        for (int i = 0; i < final_index_->getBaseLen(); i++) {
            x.push_back((final_index_->getBaseData() + 2 * i)[0]);
            y.push_back((final_index_->getBaseData() + 2 * i)[1]);
        }
        plt::scatter(x, y);

        plt::show();

        return this;
    }

    void IndexBuilder::degree_info() {
        unsigned max = 0, min = 1e6, avg = 0;
        for (size_t i = 0; i < final_index_->getBaseLen(); i++) {
            auto size = final_index_->getFinalGraph()[i][0].size();
            max = max < size ? size : max;
            min = min > size ? size : min;
            avg += size;
        }
        avg /= 1.0 * final_index_->getBaseLen();
        printf("Degree Statistics: Max = %d, Min = %d, Avg = %d\n", max, min, avg);
    }

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
            findroot(flags, root);
        }
        printf("Conn Statistics: conn = %d\n", conn);
    }

    void IndexBuilder::findroot(boost::dynamic_bitset<> &flag, unsigned &root) {
        unsigned id = final_index_->getBaseLen();
        for (unsigned i = 0; i < final_index_->getBaseLen(); i++) {
            if (flag[i] == false) {
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
            for (unsigned i = 0; i < final_index_->getFinalGraph()[tmp][0].size(); i++) {
                if (flag[final_index_->getFinalGraph()[tmp][0][i]] == false) {
                    next = final_index_->getFinalGraph()[tmp][0][i];
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


    IndexBuilder *IndexBuilder::load(char *data_file, char *query_file, char *ground_file, Parameters &parameters) {
        std::cout << "__LOAD DATA__" << std::endl;

        auto *a = new ComponentLoad(final_index_);
        a->LoadInner(data_file, query_file, ground_file, parameters);

        e = std::chrono::high_resolution_clock::now();

        std::cout << "base data len : " << final_index_->getBaseLen() << std::endl;
        std::cout << "base data dim : " << final_index_->getBaseDim() << std::endl;
        std::cout << "query data len : " << final_index_->getQueryLen() << std::endl;
        std::cout << "query data dim : " << final_index_->getQueryDim() << std::endl;
        std::cout << "ground truth data len : " << final_index_->getGroundLen() << std::endl;
        std::cout << "ground truth data dim : " << final_index_->getGroundDim() << std::endl;

        std::cout << final_index_->getParam().toString() << std::endl;

        return this;
    }

    IndexBuilder *IndexBuilder::init(TYPE type, bool debug) {
        ComponentInit *a = nullptr;

        if (type == INIT_NN_DESCENT) {
            std::cout << "__INIT : NN-Descent__" << std::endl;
            a = new ComponentInitNNDescent(final_index_);
        } else if (type == INIT_KDT) {
            std::cout << "__INIT : KDT__" << std::endl;
            a = new ComponentInitKDT(final_index_);
        } else if (type == INIT_RANDOM) {
            std::cout << "__INIT : RANDOM__" << std::endl;
            a = new ComponentInitRandom(final_index_);
        } else if (type == INIT_HCNNG) {
            std::cout << "__INIT : HCNNG__" << std::endl;
            a = new ComponentInitHCNNG(final_index_);
        } else {
            std::cout << "__INIT : WRONG TYPE__" << std::endl;
        }

        a->InitInner();

        e = std::chrono::high_resolution_clock::now();

        if (debug) {
            // degree
            degree_info();

            // 连通分量
            conn_info();
        }

        std::cout << "__INIT FINISH__" << std::endl;

        return this;
    }

    IndexBuilder *IndexBuilder::save_graph(char *graph_file) {
        std::cout << "__SAVE__" << std::endl;

        auto *a = new ComponentSerializationCompactGraph(final_index_);

        a->SaveGraphInner(graph_file);

        std::cout << "__SAVE FINISH__" << std::endl;

        return this;
    }

    IndexBuilder *IndexBuilder::load_graph(char *graph_file) {
        std::cout << "__LOAD__" << std::endl;
        Index::CompactGraph().swap(final_index_->getFinalGraph());

        auto *a = new ComponentSerializationCompactGraph(final_index_);

        a->LoadGraphInner(graph_file);

        std::cout << final_index_->getFinalGraph().size() << std::endl;
        std::cout << final_index_->getFinalGraph()[0][0].size() << std::endl;

        std::cout << "__LOAD FINISH__ : " << final_index_->getFinalGraph().size() << " " <<
                  final_index_->getFinalGraph()[0][0].size() << std::endl;

        return this;
    }

    IndexBuilder *IndexBuilder::refine(TYPE type, bool debug) {
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
        } else if (type == REFINE_VAMANA) {
            std::cout << "__REFINE : VAMANA__" << std::endl;
            a = new ComponentRefineVAMANA(final_index_);

            // first pass α = 1
            final_index_->alpha = 1;
            a->RefineInner();

            // second pass α >= 1
            final_index_->alpha = final_index_->getParam().get<float>("alpha");

        } else if (type == REFINE_NN_DESCENT) {
            std::cout << "__REFINE : NN_DESCENT__" << std::endl;
            a = new ComponentRefineEFANNA(final_index_);
        } else if (type == REFINE_NSW) {
            std::cout << "__REFINE : NSW__" << std::endl;
            a = new ComponentRefineNSW(final_index_);
        } else if (type == REFINE_TEST) {
            std::cout << "__REFINE : TEST__" << std::endl;
            a = new ComponentRefineTest(final_index_);
        } else if(type == REFINE_ANNG) {
            std::cout << "__REFINE : ANNG__" << std::endl;
            a = new ComponentRefineANNG(final_index_);
        } else {
            std::cerr << "__REFINE : WRONG TYPE__" << std::endl;
        }

        a->RefineInner();
        if (debug) {
            // degree
            degree_info();

            // 连通分量
            conn_info();
        }

        std::cout << "__REFINE : FINISH__" << std::endl;

        e = std::chrono::high_resolution_clock::now();

        return this;
    }

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
        if (entry_type == ENTRY_RANDOM) {
            a = new ComponentSearchEntryRand(final_index_);
        } else if (entry_type == ENTRY_NSG_CENTROID) {
            a = new ComponentSearchEntryCentroid(final_index_);
        } else if (entry_type == SEARCH_ENTRY_NSSG_CENTROID) {
            a = new ComponentSearchEntrySubCentroid(final_index_);
        } else if (entry_type == SEARCH_ENTRY_RANDOM) {

        }

        // ROUTE
        ComponentSearchRoute *b = nullptr;
        if (route_type == ROUTER_GREEDY) {
            b = new ComponentSearchRouteGreedy(final_index_);
        }

        for (unsigned L = L_start; L <= L_end; L += LI) {
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
}


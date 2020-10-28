# WEAVESS

## Architecture

## Algorithms

|  Algo  |             TYPE             |        Init         |       Entry       |   Candidate   |        Prune         |     Conn     |    Search Entry    |    Search Route    |
|:------:|:----------------------------:| :------------------:| :---------------: | :-----------: | :-------------------:| :-----------:| :-----------------:|:------------------:|
| KGraph |                              |     NN-Descent      |       Query       | PROPAGATION 1 |        Naive         |              |      Random        |       Greedy       |
| NSG    |          Refinement          |     NN-Descent      |      Centroid     |     Greedy    |        NSG           | Reverse+DFS  |     Centroid       |       Greedy       |
| SSG    |          Refinement          |     NN-Descent      |       Query       | PROPAGATION 2 |        SSG           | Reverse+DFS  |    Sub Centroid    |       Greedy       |
| DPG    |          Refinement          |     NN-Descent      |       Query       | PROPAGATION 1 |        DPG           |    Reverse   |      Random        |       Greedy       |
| VAMANA |          Refinement          |       Random        |      Centroid     |     Greedy    |  HEURISTIC + VAMANA  |    Reverse   |     Centroid       |       Greedy       |
| EFANNA | Divide&Conquer + Refinement  |                     |      KD-tree      |   NN-Descent  |                      |              |      KD-tree       |       Greedy       |
| IEH    |      Hash + Refinement       |                     |        LSH        |               |                      |              |                    |                    |
| NSW    |          Increment           |                     |                   |               |                      |              |                    |                    |
| HNSW   |          Increment           |                     |                   |               |      HEURISTIC       |              |                    |                    |
| NGT    |          Increment           |       ANNG          |                   |               |        ONNG          |              |      DVPTree       |Greedy(Range Search)|
| SPTAG  |        Divide&Conquer        |                     | KD-tree / BK-tree |               |        RNG           |              |                    |                    |

### KGraph
                                                                      not need sort   need sort
load(dataset -> float*) -> init(graph_ -> final_graph_) -> entry() -> candidate() -> prune(final_graph_ -> cut_graph_) -> conn() -> final_graph_ -> search

#### Parameters

|  Dataset  |  K  |  L  | Iter |  S |  R  |
|:---------:|:---:|:---:|:----:|:--:|:---:|
| SIFT1M    | 50  | 70  |  10  | 10 | 50  |
| GIST1M    | 100 | 120 |  10  | 15 | 100 |
| Crawl     |  |  |    |  |  |
| GloVe-100 |  |  |    |  |  |
| uqv       |  |  |    |  |  |
| enron     |  |  |    |  |  |
| Crawl     |  |  |    |  |  |
| GloVe-100 |  |  |    |  |  |

### NSG

dataset -> load(float*) ->


## TODO

编译通过，验证图出度入度、连通分量，验证recall-qps

-[ x ] KGraph

-[ x ] NSG

-[ x ] NSSG

-[ x ] DPG

-[ x ] VAMANA

-[ x ] EFANNA

-[ ] IEH

-[ x ] NSW

-[ ] HNSW

-[ ] NGT

-[ ] HCNNG

-[ ] SPTAG

-[ ] FANNG


* 注意 flags 是否reset
* 增量式 search_L, search_K ？
* HNSW 召回率
* SPTAG init(KDT) -> refine(PBT ? )


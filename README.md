# WEAVESS

## Architecture

## Algorithms
       
|  Algo  |       TYPE       |          Init         |       Entry       |   Candidate   |     Prune    |     Conn     |    Search Entry    |    Search Route    |
|:------:|:----------------:| :--------------------:| :---------------: | :-----------: | :-----------:| :-----------:| :-----------------:|:------------------:|
| KGraph |    Refinement    |       NN-Descent      |       Query       | PROPAGATION 1 |    Naive     |              |      Random        |       Greedy       |
| FANNG  |    Refinement    |         KNNG          |       Query       | PROPAGATION 1 |     RNG      |              |      Random        |      Backtrack     |
| NSG    |    Refinement    |       NN-Descent      |      Centroid     |     Greedy    |     RNG      | Reverse+DFS  |     Centroid       |       Greedy       |
| SSG    |    Refinement    |       NN-Descent      |       Query       | PROPAGATION 2 |     SSG      | Reverse+DFS  |    Sub Centroid    |       Greedy       |
| DPG    |    Refinement    |       NN-Descent      |       Query       | PROPAGATION 1 |     DPG      |    Reverse   |      Random        |       Greedy       |
| VAMANA |    Refinement    |         Random        |      Centroid     |     Greedy    |    VAMANA    |    Reverse   |     Centroid       |       Greedy       |
| EFANNA |    Refinement    |                       |      KD-tree      |   NN-Descent  |              |              |      KD-tree       |       Greedy       |
| IEH    |    Refinement    |         KNNG          |                   |               |              |              |        LSH         |       Greedy       |
| NSW    |    Increment     |                       |     First Node    |               |              |              |                    |                    |
| HNSW   |    Increment     |                       |   Top Layer Node  |               |     RNG      |              |                    |                    |
| NGT    |    Increment     |         ANNG          |                   |               |     ONNG     |              |      DVPTree       |       Greedy       |
| SPTAG  |  Divide&Conquer  |                       | KD-tree / BK-tree |               |     RNG      |              |                    |                    |
| HCNNG  |  Divide&Conquer  |Hierarchical Clustering|                   |               |              |              |      KD-tree       |       Guided       |

## Parameters

### IEH

|  Name        |  Default  |  Description                                  |
|:------------:|:---------:|:---------------------------------------------:|
|  P           | 10        | number of top nearest candidates              |
|  K           | 50        | number of expansion                           |
|  S           | 3         | iteration number                              |

### VAMANA

|  Name        |  Default  |  Description                                  |
|:------------:|:---------:|:---------------------------------------------:|
|  R           | 70        | degree bound                                  |
|  L           | 125       | number of top nearest candidates              |
|  alpha       | 2         | distance threshold                            |
 

## TODO

- [x] KGraph

- [x] NSG

- [x] NSSG

- [x] DPG

- [x] EFANNA

- [x] IEH

- [x] VAMANA




- [x] NSW

- [x] HNSW



- [ ] NGT

- [x] SPTAG



- [x] HCNNG

- [x] FANNG

### TODO

* n2 引入third_party 方法
* FANNG 朴素 init 方法代价较高 : 考虑改进为论文中的改进方法
* VAMANA REFINE 速度慢
* 检查个算法candidate init 后排序情况
* efanna 算法问题
* FANNG RNG -> HNSW prune
* 检查 search getQueryData




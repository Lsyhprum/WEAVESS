# WEAVESS

WEAVESS is a frame for testing the major graph-based approximate nearest neighbor search (ANNS) algorithms.

## How to use

### Windows

* Prerequisites : C++11、CLion、MinGW-w64、boost 1.73.0

### Linux

* Prerequisites : 

## Algorithms

|  Algo  |  Init Framework |     Refine     | Connection   |Entry Access | Routing |
|:------:| :--------------:| :------------: | :----------: | :----------:| :-----: |
| KGraph |  NN-Descent     |                |              | Random      |  Greedy |
| IEH    |  **Hash**       |                |              | Random      |  Greedy |
| EFANNA |  **KDTree**     |  NN-Descent    |              | **KDTree**  |  Greedy |
| NSG    |  NN-Descent     |  **MRNG**      |**DFS**       | **Centroid**|  Greedy |
| NSSG   |  NN-Descent     |  **SSG**       |**DFS_expand**| Random      |  Greedy |
| DPG    |  NN-Descent     |  **DPG**       |**Reverse**   | Random      |  Greedy |
| NSW    |                 |                |              |             |         |
| HNSW   |                 |                |              |             |         |
| NGT    |                 |                |              |             |         |
| SPTAG  |                 |                |              |             |         |
| FANNG  |                 |                |              |             |         |
|Vamana  |  NN-Descent     |  **Vamana**    |              |             |         |
| HCNNG  |                 |                |              |             |         |



## ANNS Performance


## Building Parameters

### KGraph

* **K** : 'K' of K-NNG
* **L** : candidate pool size, larger is more accurate but slower, no smaller than K.
* **iter** : NN-Descent iteration times, iter usually < 30.
* **S** : number of neighbors in local join, larger is more accurate but slower.
* **R** : number of reverse neighbors, larger is more accurate but slower.

|  Dataset  |  K  |  L  | iter |  S |  R  |
|:---------:|:---:|:---:|:----:|:--:|:---:|
| SIFT1M    | 200 | 200 |  12  | 10 | 100 |
| GIST1M    | 400 | 400 |  12  | 15 | 100 |
| Crawl     | 400 | 420 |  12  | 15 | 100 |
| GloVe-100 | 400 | 420 |  12  | 20 | 200 |

### EFANNA

* **nTrees** : 'nTrees' is the number of trees used to build the graph (larger is more accurate but slower)
* **mLevel** : conquer-to-depth (smaller is more accurate but slower) 

* **K** : 'K' of K-NNG
* **L** : candidate pool size, larger is more accurate but slower, no smaller than K.
* **iter** : NN-Descent iteration times, iter usually = 4
* **S** : number of neighbors in local join, larger is more accurate but slower.
* **R** : number of reverse neighbors, larger is more accurate but slower.

|  Dataset  |  nTrees | mLevel |  K  |  L  | iter |  S |  R  |
|:---------:|:-------:|:------:|:---:|:---:|:----:|:--:|:---:|
| SIFT1M    |    8    |  8     | 200 | 200 | 4    | 10 | 100 |

### NSG

+ **L_nsg** : controls the quality of the NSG, the larger the better.
+ **R_nsg** : controls the index size of the graph, the best R is related to the intrinsic dimension of the dataset.
+ **C_nsg** : controls the maximum candidate pool size during NSG contruction.

| Dataset |  L_nsg |  R_nsg |  C_nsg  |
|:-------:|:--:|:--:|:---:|
|  SIFT1M | 40 | 50 | 500 |
|  GIST1M | 60 | 70 | 500 |


### NSSG

|  Dataset  |  L  |  R  | Angle |
|:---------:|:---:|:---:|:-----:|
| SIFT1M    | 100 | 50  |  60   |
| GIST1M    | 500 | 70  |  60   |
| Crawl     | 500 | 40  |  60   |
| GloVe-100 | 500 | 50  |  60   |


### DPG

* **L_dpg** : neighbors per data point, the value is half of KGraph.


## Search Parameters

+ `SEARCH_K` controls the number of result neighbors we want to query.
+ `search_L`: range from `search_K` to 2000, controls the quality of the search results, 
the larger the better but slower. The `SEARCH_L` cannot be samller than the `SEARCH_K`


## TODO

-[ ] KGraph

-[ ] IEH

-[ ] EFANNA

-[ ] DPG

-[ ] NSG

-[ ] NSSG

-[ ] NSW

-[ ] HNSW

-[ ] NGT

-[ ] SPTAG

-[ ] FANNG


* IEH 实现
* NSW 
* HNSW
* KGraph 修改
* EFANNA knn_graph 修改  参数缺失 内存不够
* 分离算法接口与 IndexBuilder 接口
* 与原算法实现进行比较
* SIMD 优化
* PruneInner，Link 公共代码合并
* coarse / eva 重构 —— search
* 检查数据结构是否清空

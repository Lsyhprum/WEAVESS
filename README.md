# WEAVESS

WEAVESS is a frame for testing the major graph-based approximate nearest neighbor search (ANNS) algorithms.

### How to use

### Building Parameters

#### KGraph

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

#### EFANNA



#### NSG

+ **L_nsg** : controls the quality of the NSG, the larger the better.
+ **R_nsg** : controls the index size of the graph, the best R is related to the intrinsic dimension of the dataset.
+ **C_nsg** : controls the maximum candidate pool size during NSG contruction.

| Dataset |  L_nsg |  R_nsg |  C_nsg  |
|:-------:|:--:|:--:|:---:|
|  SIFT1M | 40 | 50 | 500 |
|  GIST1M | 60 | 70 | 500 |


#### NSSG

|  Dataset  |  L  |  R  | Angle |
|:---------:|:---:|:---:|:-----:|
| SIFT1M    | 100 | 50  |  60   |
| GIST1M    | 500 | 70  |  60   |
| Crawl     | 500 | 40  |  60   |
| GloVe-100 | 500 | 50  |  60   |


### Search Parameters

+ `SEARCH_K` controls the number of result neighbors we want to query.
+ `search_L`: range from `search_K` to 2000, controls the quality of the search results, 
the larger the better but slower. The `SEARCH_L` cannot be samller than the `SEARCH_K`


### TODO

-[x] KGraph

-[ ] IEH

-[x] EFANNA

-[ ] DPG

-[x] NSG

-[x] NSSG

-[ ] NSW

-[ ] HNSW

-[ ] NGT

-[ ] SPTAG

-[ ] FANNG


* DPG conn 实现
* PruneInner，Link 公共代码合并
* IEH 实现
* coarse / eva 重构 —— search
* Eva 重构
* HNSW
* 分类算法接口与 IndexBuilder 接口
* 与原算法实现进行比较

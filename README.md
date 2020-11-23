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

### KGraph

### FANNG

|  Name        |  Default  |  Description                       |  sift1M |  gist |  glove-100 |  crawl |  audio |  msong |  uqv |  enron |
|:------------:|:---------:|:----------------------------------:|:-------:|:-----:|:----------:|:------:|:------:|:------:|:----:|:------:|
|  R           |           | degree bound                       |    70   |   10  |     10     |   90   |   90   |   50   |  10  |   30   |

### NSG

### SSG

### DPG

### VAMANA

|  Name        |  Default  |  Description                       |  sift1M |  gist |  glove-100 |  crawl |  audio |  msong |  uqv |  enron |   c_1   |  c_10 |  c_100  |  d_8   |  d_32  |  d_128 | n_10000 |n_100000|n_1000000 |  s_1  |    s_5     |  s_10  |
|:------------:|:---------:|:----------------------------------:|:-------:|:-----:|:----------:|:------:|:------:|:------:|:----:|:------:|:-------:|:-----:|:-------:|:------:|:------:|:------:|:-------:|:------:|:--------:|:-----:|:----------:|:------:|
|  R           | 70        | degree bound                       |    70   |  110  |    110     |    70  |   70   |   30   |  30  |   30   |   110   |   70  |   70    |   70   |   70   |   70   |   30    |   70   |     \    |   70  |     70     |   90   |
|  L           | 125       | number of top nearest candidates   |    80   |  120  |    140     |   100  |  100   |   40   |  60  |   40   |   140   |   90  |   80    |   80   |   90   |   80   |   40    |   80   |     \    |   90  |     90     |   100  |
|  alpha       | 2         | distance threshold                 |         |       |            |        |        |        |      |        |         |       |         |        |        |        |         |        |          |       |            |        |

### EFANNA

### IEH

|  Name        |  Default  |  Description                       |
|:------------:|:---------:|:----------------------------------:|
|  P           | 10        | number of top nearest candidates   |
|  K           | 50        | number of expansion                |
|  S           | 3         | iteration number                   |

### NSW

|  Name             |  Default  |  Description                             |  sift1M |  gist |  glove-100 |  crawl |  audio |  msong |  uqv |  enron |   c_1   |  c_10 |  c_100  |  d_8   |  d_32  |  d_128 | n_10000 |n_100000|n_1000000 |  s_1  |    s_5     |  s_10  |
|:-----------------:|:---------:|:----------------------------------------:|:-------:|:-----:|:----------:|:------:|:------:|:------:|:----:|:------:|:-------:|:-----:|:-------:|:------:|:------:|:------:|:-------:|:------:|:--------:|:-----:|:----------:|:------:|
|  max_m0           |  24       | max number of edges for nodes at level0  |    40   |   60  |     80     |    60  |   40   |   60   |  30  |   80   |   100   |   30  |   70    |    50  |   80   |   80   |   20    |   80   |   100    |   60  |     70     |    50  |
|  ef_construction  |  150      | number of top nearest candidates         |   300   |  200  |    100     |   400  |  800   |  300   | 400  |  600   |   500   |  100  |   400   |   500  |  100   |  1000  |  300    |  300   |   400    |  600  |    300     |  1000  |

### HNSW

### NGT

### SPTAG-KDT

### SPTAG_BKT

### HCNNG

|  Name        |  Default  |  Description                       |  sift1M |  gist |  glove-100 |  crawl |  audio |  msong |  uqv |  enron |   c_1   |  c_10 |  c_100  |  d_8   |  d_32  |  d_128 | n_10000 |n_100000|n_1000000 |  s_1  |    s_5     |  s_10  |
|:------------:|:---------:|:----------------------------------:|:-------:|:-----:|:----------:|:------:|:------:|:------:|:----:|:------:|:-------:|:-----:|:-------:|:------:|:------:|:------:|:-------:|:------:|:--------:|:-----:|:----------:|:------:|
|  num_cl      |           | num clusters                       |  45/90  | 30/30 |   60/100   |  35/70 |  40/60 | 55/100 | 20/80| 30/100 |  50/100 | 65/70 |  65/90  | 65/80  |  50/70 | 65/100 | 50/90   | 60/90  |  55/100  | 65/60 |  65/60     | 50/90  |
|  minsize_cl  |  sqrt(N)  | min size cluster                   |         |       |            |        |        |        |      |        |         |       |         |        |        |        |         |        |          |       |            |        |
 

## TODO

- [x] KGraph

- [x] NSG

- [x] NSSG

- [x] DPG

- [x] EFANNA

- [x] IEH

- [x] VAMANA

- [x] HCNNG

- [x] FANNG




- [x] NSW

- [x] HNSW

- [ ] NGT

- [x] SPTAG







### TODO

* n2 引入third_party 方法
* VAMANA REFINE 速度慢
* 检查个算法candidate init 后排序情况
* efanna 算法问题
* FANNG RNG -> HNSW prune
* 检查 search getQueryData




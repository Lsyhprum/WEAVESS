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
| HNSW   |    Increment     |                       |   Top Layer Node  |               |   HEURISTIC  |              |                    |                    |
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
 

## TODO

- [x] KGraph

- [x] NSG

- [x] NSSG

- [x] DPG

- [x] VAMANA

- [x] EFANNA

- [x] IEH

- [x] NSW

- [x] HNSW

- [ ] NGT

- [x] HCNNG

- [x] SPTAG

- [x] FANNG




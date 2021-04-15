## Machine learning based approaches

### Introduction

Our paper ([arXiv link](https://arxiv.org/abs/2101.12631), [PDF](https://arxiv.org/pdf/2101.12631.pdf))  included an evaluation and analysis of three Machine learning (ML) based methods. We would like to clarify that existing ML-based approaches vastly differ from the algorithms we focus on in this paper. In general, they can be viewed as some optimizations on graph-based algorithms (such as NSW and NSG). 

### Algorithms

we evaluate three representative ML-based approaches:

* [Learning to Route in Similarity Graphs](http://proceedings.mlr.press/v97/baranchuk19a.html) learns vertex representation on graph-based algorithms (for example, NSW) to provide better routing; 
* [Improving Approximate Nearest Neighbor Search through Learned Adaptive Early Termination](https://dl.acm.org/doi/10.1145/3318464.3380600) performs ANNS on HNSW through learned adaptive early termination;
* [Graph-based nearest neighbor search: From practice to theory](http://proceedings.mlr.press/v119/prokhorenkova20a.html) maps the dataset into a space of lower dimension while trying to preserve local geometry by ML, and then it can be combined with any graph-based algorithms such as HNSW or NSG.

their papers and the original codes are given in the following table.

|                          ALGORITHM                           |                            PAPER                             |                             CODE                             |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|            Learning to Route in Similarity Graphs            | [ICML'2019](http://proceedings.mlr.press/v97/baranchuk19a.html) |  [Python](https://github.com/dbaranchuk/learning-to-route)   |
| Improving Approximate Nearest Neighbor Search through Learned Adaptive Early Termination | [SIGMOD'2020](https://dl.acm.org/doi/10.1145/3318464.3380600) | [Python](https://github.com/efficient/faiss-learned-termination) |
| Graph-based nearest neighbor search: From practice to theory | [ICML'2020](http://proceedings.mlr.press/v119/prokhorenkova20a.html) |     [Python](https://github.com/Shekhale/gbnns_dim_red)      |


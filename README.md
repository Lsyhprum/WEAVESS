# A Comprehensive Survey and Experimental Comparison of Graph-based Approximate Nearest Neighbor Search

## Introduction

Approcimate Nearest Neighbor Search (ANNS) is a fundamental building block in various application domains. Recently, graph-based algorithms have emerged as a very effective choice to implement ANNS. Our paper provides a comprehensive comparative analysis and experimental evaluation of representative graph-based ANNS algorithms on carefully selected datasets of varying sizes and characteristics.

This project contains the code, dataset, optimal parameters, and other detailed information used for the experiments of our paper. It is worth noting that we reimplement all algorithms based on exactly the same design pattern, programming language (except for the hash table in IEH) and tricks, and experimental setup, which makes the comparison more fair. 

## Algorithms

we evaluate thirteen representative graph-based ANNS algorithms, and their papers and the original codes are given in the following table.

|   ALGO   |     PAPER     |   CODE   |
|:--------:|:------------:|:--------:|
|  KGraph  |  [WWW'2011](https://dl.acm.org/doi/abs/10.1145/1963405.1963487)  |  [C++/Python](https://github.com/aaalgo/kgraph)  |
|  FANNG   |  [CVPR'2016](https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Harwood_FANNG_Fast_Approximate_CVPR_2016_paper.html)  |   -   |
|  NSG        |    [VLDB'2019](http://www.vldb.org/pvldb/vol12/p461-fu.pdf)    | [C++](https://github.com/ZJULearning/nsg)      |
|  NSSG        |    [arXiv'2019](https://arxiv.org/abs/1907.06146)    |      [C++/Python](https://github.com/ZJULearning/ssg)      |
|  DPG        |    [TKDE'2019](https://ieeexplore.ieee.org/abstract/document/8681160)    | [C++](https://github.com/DBWangGroupUNSW/nns_benchmark/tree/master/algorithms/DPG) |
|  Vamana     |    [NeurIPS'2019](http://harsha-simhadri.org/pubs/DiskANN19.pdf)    |         -        |
|  EFANNA     |    [arXiv'2016](https://arxiv.org/abs/1609.07228)    | [C++/MATLAB](https://github.com/ZJULearning/ssg) |
|  IEH        |    [IEEE T CYBERNETICS'2014](https://ieeexplore.ieee.org/abstract/document/6734715/)    |        -      |
|  NSW        | [IS'2014](https://www.sciencedirect.com/science/article/abs/pii/S0306437913001300) | [C++/Python](https://github.com/kakao/n2) |
|  HNSW       | [TPAMI'2018](https://ieeexplore.ieee.org/abstract/document/8594636) | [C++/Python](https://github.com/kakao/n2) |
|  NGT-panng  | [SISAP'2016](https://link.springer.com/chapter/10.1007/978-3-319-46759-7_2) |         [C++/Python](https://github.com/yahoojapan/NGT)         |
|  NGT-onng  |    [arXiv'2018](https://arxiv.org/abs/1810.07355)    |         [C++/Python](https://github.com/yahoojapan/NGT)         |
|  SPTAG-KDT  |  [ACM MM'2012](https://dl.acm.org/doi/abs/10.1145/2393347.2393378); [CVPR'2012](https://ieeexplore.ieee.org/abstract/document/6247790); [TPAMI'2014](https://ieeexplore.ieee.org/abstract/document/6549106)  | [C++](https://github.com/microsoft/SPTAG) |
|  SPTAG-BKT  | [ACM MM'2012](https://dl.acm.org/doi/abs/10.1145/2393347.2393378); [CVPR'2012](https://ieeexplore.ieee.org/abstract/document/6247790); [TPAMI'2014](https://ieeexplore.ieee.org/abstract/document/6549106) | [C++](https://github.com/microsoft/SPTAG) |
|  HCNNG      |  [PR'2019](https://www.sciencedirect.com/science/article/abs/pii/S0031320319302730)  |-|

## Datasets

Our experiment involves eight [real-world datasets](https://github.com/Lsyhprum/WEAVESS/tree/master/dataset) popularly deployed by existing works. All datasets are pre-split into base data and query data and come with groundtruth data in the form of the top 20 or 100 neighbors. Additional twelve [synthetic datasets](https://github.com/Lsyhprum/WEAVESS/tree/master/dataset) are used to test the scalability of each algorithm to the performance of different datasets.

Note that, all base data and query data are converted to `fvecs` format, and groundtruth data is converted to `ivecs` format. Please refer [here](http://yael.gforge.inria.fr/file_format.html) for the description of `fvecs` and `ivecs` format. All datasets in this format can be downloaded from [here](https://github.com/Lsyhprum/WEAVESS/tree/master/dataset).

## Parameters

For the optimal parameters of each algorithm on all experimental datasets, see the [parameters](https://github.com/Lsyhprum/WEAVESS/tree/master/parameters) page.

## Usage

### Prerequisites

* GCC 4.9+ with OpenMP
* CMake 2.8+
* Boost 1.55+

### Compile on Linux

```shell
$ git clone https://github.com/Lsyhprum/WEAVESS.git
$ cd WEAVESS/
$ mkdir build && cd build/
$ cmake ..
$ make -j
```

### Build graph index

Before building index, you should set the root directory for the dataset in `WEAVESS/test/main.cpp` first. Then, you can run the following instructions for build graph index.

```shell
cd WEAVESS/build/test/
./main algorithm_name dataset_name build
```

With the index built, you can run the following commands to perform the search.

### Search via index

```shell
cd WEAVESS/build/test/
./main algorithm_name dataset_name search
```

## Experiment evaluation

Note that the default [`master` branch](https://github.com/Lsyhprum/WEAVESS/tree/master) is the evaluation of the overall performance of all algorithms, and the evaluation of a certain component needs to be carried out under the [`test` branch](https://github.com/Lsyhprum/WEAVESS/tree/test). For more details, please see our paper. 

## Acknowledgements

Thanks to everyone who provided references for this project. Special thanks to Dr. [Weijie Zhao](https://scholar.google.com/citations?user=c-gzOhwAAAAJ&hl=zh-CN&oi=sra), Dr. [Mingjie Li](https://scholar.google.com/citations?user=MoLSu5cAAAAJ&hl=zh-CN&oi=sra), and Dr. [Cong Fu](https://scholar.google.com/citations?user=Gvp9ErEAAAAJ&hl=zh-CN&oi=sra) for their assistance in the necessary implementation of this project.
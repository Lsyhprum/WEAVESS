# A Comprehensive Survey and Experimental Comparison of Graph-based Approximate Nearest Neighbor Search

## Introduction

Approcimate Nearest Neighbor Search (ANNS) is a fundamental building block in various application domains. Recently, graph-based algorithms have emerged as a very effective choice to implement ANNS. Our paper ([arXiv link](https://arxiv.org/abs/2101.12631), [PDF](https://arxiv.org/pdf/2101.12631.pdf)) provides a comprehensive comparative analysis and experimental evaluation of representative graph-based ANNS algorithms on carefully selected datasets of varying sizes and characteristics.

This project contains the code, dataset, optimal parameters, and other detailed information used for the experiments of our paper. It is worth noting that we reimplement all algorithms based on exactly the same design pattern, programming language (except for the hash table in IEH) and tricks, and experimental setup, which makes the comparison more fair. 

## Algorithms

we evaluate thirteen representative graph-based ANNS algorithms, and their papers and the original codes are given in the following table.

|   ALGORITHM   |     PAPER     |   CODE   |
|:--------:|:------------:|:--------:|
|  KGraph  |  [WWW'2011](https://dl.acm.org/doi/abs/10.1145/1963405.1963487)  |  [C++/Python](https://github.com/aaalgo/kgraph)  |
|  FANNG   |  [CVPR'2016](https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Harwood_FANNG_Fast_Approximate_CVPR_2016_paper.html)  |   -   |
|  NSG        |    [VLDB'2019](http://www.vldb.org/pvldb/vol12/p461-fu.pdf)    | [C++](https://github.com/ZJULearning/nsg)      |
|  NSSG        |    [TPAMI'2021](https://ieeexplore.ieee.org/abstract/document/9383170)    |      [C++/Python](https://github.com/ZJULearning/ssg)      |
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
| KDRG | [SIGKDD'2011](https://dl.acm.org/doi/10.1145/2020408.2020576) |-|

## Datasets

Our experiment involves eight [real-world datasets](https://github.com/Lsyhprum/WEAVESS/tree/dev/dataset) popularly deployed by existing works. All datasets are pre-split into base data and query data and come with groundtruth data in the form of the top 20 or 100 neighbors. Additional twelve [synthetic datasets](https://github.com/Lsyhprum/WEAVESS/tree/dev/dataset) are used to test the scalability of each algorithm to the performance of different datasets.

Note that, all base data and query data are converted to `fvecs` format, and groundtruth data is converted to `ivecs` format. Please refer [here](http://yael.gforge.inria.fr/file_format.html) for the description of `fvecs` and `ivecs` format. All datasets in this format can be downloaded from [here](https://github.com/Lsyhprum/WEAVESS/tree/dev/dataset).

## Parameters

For the optimal parameters of each algorithm on all experimental datasets, see the [parameters](https://github.com/Lsyhprum/WEAVESS/tree/dev/parameters) page.

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

### Index construction evaluation

Before building index, you should set the root directory for the dataset in `WEAVESS/test/main.cpp` first. Then, you can run the following instructions for build graph index.

```shell
cd WEAVESS/build/test/
./main algorithm_name dataset_name build
```

After the build is completed, the graph index will be written in the current folder in binary format (for index size). The index construction time can be viewed from the output log information. You can run the following command in current directory for getting other index information, such as average out-degree, graph quality, and the number of connected components.

```shell
./main algorithm_name dataset_name info
```

### Search performance
With the index built, you can run the following commands to perform the search. Related information about the search such as search time, distance evaluation times, candidate set size, average query path length, memory load can be obtained or calculated according to the output log information.

```shell
cd WEAVESS/build/test/
./main algorithm_name dataset_name search
```

### Components evaluation

Note that the default [`dev` branch](https://github.com/Lsyhprum/WEAVESS/tree/dev) is the evaluation of the overall performance of all algorithms, and the evaluation of a certain component needs to be carried out under the [`test` branch](https://github.com/Lsyhprum/WEAVESS/tree/test). For more details, please see [our paper](https://arxiv.org/pdf/2101.12631.pdf). 

### Machine learning-based optimizations

See [here](https://github.com/Lsyhprum/WEAVESS/tree/dev/ml) for details.

## Reference

Please cite our work in your publications if it helps your research:

```
@misc{wang2021comprehensive,
    title={A Comprehensive Survey and Experimental Comparison of Graph-Based Approximate Nearest Neighbor Search},
    author={Mengzhao Wang and Xiaoliang Xu and Qiang Yue and Yuxiang Wang},
    year={2021},
    eprint={2101.12631},
    archivePrefix={arXiv},
    primaryClass={cs.IR}
}
```



## Acknowledgements

Thanks to everyone who provided references for this project. Special thanks to Dr. [Weijie Zhao](https://scholar.google.com/citations?user=c-gzOhwAAAAJ&hl=zh-CN&oi=sra), Dr. [Mingjie Li](https://scholar.google.com/citations?user=MoLSu5cAAAAJ&hl=zh-CN&oi=sra), and Dr. [Cong Fu](https://scholar.google.com/citations?user=Gvp9ErEAAAAJ&hl=zh-CN&oi=sra) for their assistance in the necessary implementation of this project.
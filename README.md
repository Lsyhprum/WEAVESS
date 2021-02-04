# A Comprehensive Survey and Experimental Comparison of Graph-based Approximate Nearest Neighbor Search

## Usage

### Index Build

#### Sample Code

```cpp
const unsigned num_threads = parameters.get<unsigned>("n_threads");
std::string base_path = parameters.get<std::string>("base_path");
std::string query_path = parameters.get<std::string>("query_path");
std::string ground_path = parameters.get<std::string>("ground_path");
std::string graph_file = parameters.get<std::string>("graph_file");
	
auto *builder = new weavess::IndexBuilder(num_threads);

builder -> load(&base_path[0], &query_path[0], &ground_path[0], parameters)
        -> init(weavess::TYPE::INIT_RANDOM, false)
		-> refine(weavess::TYPE::REFINE_NN_DESCENT, false)
        -> save_graph(weavess::TYPE::INDEX_KGRAPH, &graph_file[0]);
```

#### Base Classes

* `ComponentLoad`: load input raw data for index build and algorithm parameters.
* `ComponentInit`:	initialization components for index construction.
* `ComponentPreEntry`: seed acquisition of the index construction.
* `ComponentRefine`: include candidate、prune、conn components.
* `ComponentCandidate`: candidate neighbor acquisition.
* `ComponentPrune`: neighbor selection.
* `ComponentConn`: connectivity assurance.

#### Operator

```
builder -> load()
		-> init(INIT_COMPONENT_TYPE)
		-> pre_entry(PRE_ENTRY__COMPONENT_TYPE)
		-> refine(REFINE_COMPONENT_TYPE)
		-> save_graph(SAVE_COMPONENT_TYPE);
```

### Index Search

#### Sample Code

```cpp
const unsigned num_threads = parameters.get<unsigned>("n_threads");
std::string base_path = parameters.get<std::string>("base_path");
std::string query_path = parameters.get<std::string>("query_path");
std::string ground_path = parameters.get<std::string>("ground_path");
std::string graph_file = parameters.get<std::string>("graph_file");
	
auto *builder = new weavess::IndexBuilder(num_threads);

builder -> load_graph(weavess::TYPE::INDEX_KGRAPH, &graph_file[0])
        -> search(weavess::TYPE::SEARCH_ENTRY_RAND, weavess::TYPE::ROUTER_GREEDY, weavess::TYPE::L_SEARCH_ASSIGN);
```

#### Base Classes

* `ComponentSearchEntry`: seed acquisition of the index search.
* `ComponentSearchRoute`: route strategies.

#### Operator

```
builder -> load()
		-> load_graph(LOAD_COMPONENT_TYPE)
		-> search(SEARCH_ENTRY_COMPONENT_TYPE, SEARCH_ROUTE_COMPONENT_TYPE);
```






















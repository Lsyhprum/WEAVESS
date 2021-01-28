#include <iostream>
#include <random>
#include <queue>
#include <algorithm>
#include <fstream>
#include <vector>
#include <chrono>
#include <unordered_set>
#include <cassert>

class DatasetGenerator {
public:

    DatasetGenerator(const std::string &dir, const std::string &prefix) : dir_(dir), prefix_(prefix),
                                                                          base_path(ROOT + dir + "/" + prefix +
                                                                                    "_base.fvecs"),
                                                                          query_path(ROOT + dir + "/" + prefix +
                                                                                     "_query.fvecs"),
                                                                          ground_path(ROOT + dir + "/" + prefix +
                                                                                      "_groundtruth.ivecs") {}


    void loadAllData() {
        load_data(&base_path[0], base_data, base_num, base_dim);

        load_data(&query_path[0], query_data, query_num, query_dim);

        load_data(&ground_path[0], ground_data, ground_num, ground_dim);
    }

    void genGroundTruth() {
        load_data(&base_path[0], base_data, base_num, base_dim);

        load_data(&query_path[0], query_data, query_num, query_dim);

        gen_ground_truth(ground_path, GROUND_TRUTH_DIM, base_data, query_data, base_num, query_num);
    }

    void genInner() {
        inner_query_path = ROOT + dir_ + "/" + prefix_ + "_inner_query.fvecs";
        std::cout << inner_query_path << std::endl;
        inner_ground_path = ROOT + dir_ + "/" + prefix_ + "_inner_groundtruth.ivecs";
        std::cout << inner_ground_path << std::endl;
        std::cout << "SAVE TO NEW QUERY DATASET" << std::endl;
        auto *inner_query_data = new float[query_num * query_dim];
        for (int i = 0; i < query_num; i++) {
            for (int j = 0; j < base_dim; j++) {
                inner_query_data[i * base_dim + j] = base_data[i * base_dim + j];
            }
        }
        save_data(&inner_query_path[0], inner_query_data, query_num, base_dim);

        gen_ground_truth(inner_ground_path, ground_dim, base_data, inner_query_data, base_num, query_num);
    }

    void genSample(unsigned sample_base_num, unsigned sample_query_num) {
        sample_base_path = ROOT + dir_ + "/" + prefix_ + "_sample_base.fvecs";
        std::cout << sample_base_path << std::endl;
        sample_query_path = ROOT + dir_ + "/" + prefix_ + "_sample_query.fvecs";
        std::cout << sample_query_path << std::endl;
        sample_ground_path = ROOT + dir_ + "/" + prefix_ + "_sample_groundtruth.ivecs";
        std::cout << sample_ground_path << std::endl;

        std::mt19937 rng(rand());
        std::vector<unsigned> rand_index;
        gen_random(rand_index, rng, base_num, sample_base_num);
        //std::cout << rand_index.size() << std::endl;
        std::unordered_set<unsigned> set(rand_index.begin(), rand_index.end());

        std::ifstream in(base_path, std::ios::binary);
        if (!in.is_open()) {
            std::cerr << "open file error" << std::endl;
            exit(-1);
        }
        float *sample_base_data = new float[sample_base_num * base_dim];

        in.seekg(0, std::ios::beg);
        auto *tmp = new float[base_dim];
        unsigned k = 0;
        for (size_t i = 0; i < base_num; i++) {
            in.seekg(4, std::ios::cur);
            if (set.find(i) != set.end()) {
                in.read((char *) (sample_base_data + k * base_dim), base_dim * sizeof(float));
                k++;
            } else {
                in.read((char *) tmp, base_dim * sizeof(float));
            }
        }
        in.close();

        save_data(&sample_base_path[0], sample_base_data, sample_base_num, base_dim);

        set.clear();
        std::vector<unsigned>().swap(rand_index);
        delete[] tmp;
        std::cout << "gen sample base success" << std::endl;


        save_data(&sample_query_path[0], query_data, sample_query_num, query_dim);

        std::cout << "gen sample query success" << std::endl;


        gen_ground_truth(sample_ground_path, ground_dim, sample_base_data, query_data, sample_base_num,
                         sample_query_num);

        std::cout << "gen sample ground truth success" << std::endl;


        sample_test(sample_base_num, sample_query_num);

        delete[] sample_base_data;
    }

private:
    const std::string ROOT = "F:/ANNS/DATASET/";
    const std::string dir_;
    const std::string prefix_;

    const std::string base_path;
    const std::string query_path;
    const std::string ground_path;

    std::string inner_query_path;
    std::string inner_ground_path;

    std::string sample_base_path;
    std::string sample_query_path;
    std::string sample_ground_path;

    const unsigned GROUND_TRUTH_DIM = 100;

    float *base_data = nullptr;
    unsigned base_num{};
    unsigned base_dim{};

    float *query_data = nullptr;
    unsigned query_num{};
    unsigned query_dim{};

    unsigned *ground_data = nullptr;
    unsigned ground_num{};
    unsigned ground_dim{};

    template<typename T>
    inline void load_data(const char *filename, T *&data, unsigned &num, unsigned &dim) {
        std::cout << "file : " << filename << std::endl;

        std::ifstream in(filename, std::ios::binary);
        if (!in.is_open()) {
            std::cerr << "open file error" << std::endl;
            exit(-1);
        }
        in.read((char *) &dim, 4);
        in.seekg(0, std::ios::end);
        std::ios::pos_type ss = in.tellg();
        auto f_size = (size_t) ss;
        num = (unsigned) (f_size / (dim + 1) / 4);
        data = new T[num * dim];

        in.seekg(0, std::ios::beg);
        for (size_t i = 0; i < num; i++) {
            in.seekg(4, std::ios::cur);
            in.read((char *) (data + i * dim), dim * sizeof(T));
        }
        in.close();

        std::cout << "num : " << num << std::endl;
        std::cout << "dim : " << dim << std::endl;
    }

    template<typename T>
    void save_data(const char *filename, const T *dataset, unsigned n, unsigned d) {
        FILE *ofp = fopen(filename, "wb");

        for (int i = 0; i < n; i++) {
            fwrite(&d, 4, 1, ofp);
            fflush(ofp);
            fwrite(&dataset[i * d], sizeof(T), d, ofp);
            fflush(ofp);

            //val += (4 + sizeof(T) * d);
        }

        fclose(ofp);
    }

    void gen_random(std::vector<unsigned> &index, std::mt19937 &rng, unsigned N, unsigned num) {
        for (unsigned i = 0; i < num; ++i)
            index.push_back(rng() % (N - num));

        std::sort(index.begin(), index.end());

        for (unsigned i = 1; i < num; ++i)
            if (index[i] <= index[i - 1])
                index[i] = index[i - 1] + 1;

        unsigned off = rng() % N;

        std::unordered_set<unsigned> set;
        for (unsigned i = 0; i < num; ++i) {
            index[i] = (index[i] + off) % N;
            if (set.find(index[i]) != set.end())
                std::cout << "gen random index failed" << std::endl;
            set.insert(index[i]);
            //std::cout << index[i] << std::endl;
        }
    }

    template<typename T>
    T compare(const T *a, const T *b, unsigned length) {
        T result = 0;

        float diff0, diff1, diff2, diff3;
        const T *last = a + length;
        const T *unroll_group = last - 3;

        /* Process 4 items with each loop for efficiency. */
        while (a < unroll_group) {
            diff0 = a[0] - b[0];
            diff1 = a[1] - b[1];
            diff2 = a[2] - b[2];
            diff3 = a[3] - b[3];
            result += diff0 * diff0 + diff1 * diff1 + diff2 * diff2 + diff3 * diff3;
            a += 4;
            b += 4;
        }
        /* Process last 0-3 pixels.  Not needed for standard vector lengths. */
        while (a < last) {
            diff0 = *a++ - *b++;
            result += diff0 * diff0;
        }

        return result;
    }

    void gen_ground_truth(std::string ground_path, unsigned ground_dim, float *base_data, float *query_data,
                          unsigned base_num, unsigned query_num) {

        std::chrono::high_resolution_clock::time_point s = std::chrono::high_resolution_clock::now();
        unsigned *groundtruth = new unsigned[query_num * ground_dim];

#pragma omp parallel for
        for (int i = 0; i < query_num; i++) {
            //std::cout << "save : " << i << std::endl;
            std::priority_queue<std::pair<float, unsigned>, std::vector<std::pair<float, unsigned>>, cmp> dist;
            for (int j = 0; j < base_num; j++) {
                float d = compare(query_data + query_dim * i, base_data + base_dim * j, base_dim);
                dist.push(std::pair<float, unsigned>(d, j));
            }

            for (int j = 0; j < ground_dim; j++) {
                std::pair<float, unsigned> p = dist.top();
                dist.pop();
                groundtruth[i * ground_dim + j] = p.second;
//              std::cout << p.first << std::endl;
//              std::cout << p.second << std::endl;
            }
//          std::cout << std::endl;
        }
        std::chrono::high_resolution_clock::time_point e = std::chrono::high_resolution_clock::now();

        auto time = e - s;
        // brute force    : 75205034500
        // parallel       : 34023507100
        // priority_queue : 19180940100
        // total          : 1514844795500
        std::cout << "time : " << time.count() << std::endl;
        save_data<unsigned>(&ground_path[0], groundtruth, query_num, ground_dim);
    }

    struct cmp {
        template<typename T, typename U>
        bool operator()(T const &left, U const &right) {
            if (left.first > right.first) return true;
            return false;
        }
    };

    static bool test_cmp(const std::pair<float, unsigned> a, const std::pair<float, unsigned> b) {
        return a.first < b.first;
    }

    void sample_test(unsigned sample_base_num, unsigned sample_query_num) {
        unsigned dim{};
        unsigned num{};

        std::ifstream in(sample_base_path, std::ios::binary);
        if (!in.is_open()) {
            std::cerr << "open file error" << std::endl;
            exit(-1);
        }
        in.read((char *) &dim, 4);
        in.seekg(0, std::ios::end);
        std::ios::pos_type ss = in.tellg();
        auto f_size = (size_t) ss;
        num = (unsigned) (f_size / (dim + 1) / 4);
        float *sample_base_data = new float[num * dim];

        in.seekg(0, std::ios::beg);
        for (size_t i = 0; i < num; i++) {
            in.seekg(4, std::ios::cur);
            in.read((char *) (sample_base_data + i * dim), dim * sizeof(float));
        }
        in.close();

        assert(sample_base_num == num);
        assert(base_dim == dim);
        std::cout << "test sample base num success" << std::endl;

        std::ifstream in2(sample_query_path, std::ios::binary);
        if (!in2.is_open()) {
            std::cerr << "open file error" << std::endl;
            exit(-1);
        }
        in2.read((char *) &dim, 4);
        in2.seekg(0, std::ios::end);
        ss = in2.tellg();
        f_size = (size_t) ss;
        num = (unsigned) (f_size / (dim + 1) / 4);
        float *sample_query_data = new float[num * dim];

        in2.seekg(0, std::ios::beg);
        for (size_t i = 0; i < num; i++) {
            in2.seekg(4, std::ios::cur);
            in2.read((char *) (sample_query_data + i * dim), dim * sizeof(float));
        }
        in2.close();

        assert(sample_query_num == num);
        assert(query_dim == dim);
        std::cout << "test  sample query num success" << std::endl;

        std::ifstream in3(sample_ground_path, std::ios::binary);
        if (!in3.is_open()) {
            std::cerr << "open file error" << std::endl;
            exit(-1);
        }
        in3.read((char *) &dim, 4);
        in3.seekg(0, std::ios::end);
        ss = in3.tellg();
        f_size = (size_t) ss;
        num = (unsigned) (f_size / (dim + 1) / 4);
        float *sample_ground_data = new float[num * dim];

        in3.seekg(0, std::ios::beg);
        for (size_t i = 0; i < num; i++) {
            in3.seekg(4, std::ios::cur);
            in3.read((char *) (sample_ground_data + i * dim), dim * sizeof(float));
        }
        in3.close();

        assert(sample_query_num == num);
        assert(ground_dim == dim);
        std::cout << "test sample ground num success" << std::endl;

        std::vector<unsigned> index;
        std::mt19937 rng(rand());
        gen_random(index, rng, query_num, 10);

        for (int i : index) {
            std::vector<std::pair<float, unsigned>> dist;
            for (int j = 0; j < base_num; j++) {
                float d = compare(query_data + query_dim * i, base_data + base_dim * j, base_dim);
                dist.emplace_back(d, j);
            }
            std::sort(dist.begin(), dist.end(), test_cmp);

            for (int j = 0; j < ground_dim; j++) {
                if(ground_data[i * ground_dim + j] != dist[j].second){
                    float d = compare(query_data + query_dim * i, base_data + base_dim * ground_data[i * ground_dim + j], base_dim);
                    if(d != dist[j].first) {
                        std::cout << "some wrong has happend" << std::endl;
                        return;
                    }
                }
            }
        }
        std::cout << "test sample dataset success" << std::endl;
    }

};


int main() {
    std::vector<std::string> dataset = {"sift1M", "gist", "glove-100", "crawl", "audio", "msong", "uqv", "enron"};
    std::vector<std::string> prefix = {"sift", "gist", "glove-100", "crawl", "audio", "msong", "uqv", "enron"};

    for (int i = 0; i < dataset.size(); i++) {
        DatasetGenerator gen(dataset[i], prefix[i]);
        gen.loadAllData();

        //gen.genSample(10000, 100);
    }

    return 0;
}

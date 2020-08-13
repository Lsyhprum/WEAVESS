//
// Created by Murph on 2020/8/12.
//

#ifndef WEAVESS_NEIGHBOR_H
#define WEAVESS_NEIGHBOR_H

#include <cstddef>
#include <vector>
#include <mutex>
#include <random>

namespace weavess {

    struct SimpleNeighbor {
        unsigned id;
        float distance;

        SimpleNeighbor() = default;

        SimpleNeighbor(unsigned id, float distance) : id{id}, distance{distance} {}

        inline bool operator<(const SimpleNeighbor &other) const {
            return distance < other.distance;
        }
    };


    struct Neighbor {
        unsigned id;
        float distance;
        bool flag;

        Neighbor() = default;

        Neighbor(unsigned id, float distance, bool f) : id{id}, distance{distance}, flag(f) {}

        inline bool operator<(const Neighbor &other) const {
            return distance < other.distance;
        }
    };

    typedef std::lock_guard<std::mutex> LockGuard;

    struct nhood {
        std::mutex lock;
        std::vector<Neighbor> pool;
        unsigned M;

        std::vector<unsigned> nn_old;
        std::vector<unsigned> nn_new;
        std::vector<unsigned> rnn_old;
        std::vector<unsigned> rnn_new;

        nhood() {}

        nhood(unsigned l, unsigned s, std::mt19937 &rng, unsigned N) {
            M = s;
            nn_new.resize(s * 2);
            GenRandom(rng, &nn_new[0], (unsigned) nn_new.size(), N);
            nn_new.reserve(s * 2);
            pool.reserve(l + 1);
        }

        nhood(const nhood &other) {
            M = other.M;
            std::copy(other.nn_new.begin(), other.nn_new.end(), std::back_inserter(nn_new));
            nn_new.reserve(other.nn_new.capacity());
            pool.reserve(other.pool.capacity());
        }

        void InsertIntopool(Neighbor *addr, unsigned K, Neighbor nn) {
            // find the location to insert
            int left = 0, right = K - 1;
            if (addr[left].distance > nn.distance) {
                memmove((char *) &addr[left + 1], &addr[left], K * sizeof(Neighbor));
                addr[left] = nn;
                return;
            }
            if (addr[right].distance < nn.distance) {
                addr[K] = nn;
                return;
            }
            while (left < right - 1) {
                int mid = (left + right) / 2;
                if (addr[mid].distance > nn.distance)right = mid;
                else left = mid;
            }
            //check equal ID

            while (left > 0) {
                if (addr[left].distance < nn.distance) break;
                // if (addr[left].id == nn.id) return ;
                left--;
            }
            // if(addr[left].id == nn.id||addr[right].id==nn.id)return ;
            memmove((char *) &addr[right + 1], &addr[right], (K - right) * sizeof(Neighbor));
            addr[right] = nn;
            return;
        }

        void insert(unsigned id, float dist) {
            LockGuard guard(lock);
            if (dist > pool.back().distance) return;
            for (unsigned i = 0; i < pool.size(); i++) {
                if (id == pool[i].id)return;
            }
            bool is_insert = true;
            // for (unsigned i = 0; i < pool.size(); i++) {
            //   if  (dist < pool[i].distance) {
            //     is_insert = true;
            //     break;
            //   }else {
            //     float tmp = dist  > pool[i].distance ? dist - pool[i].distance : pool[i].distance - dist;
            //     // tmp *= 10.0;
            //     // std::cout << "tmp: " << tmp << " " << "dist: " << dist << "\n";
            //     if (dist > tmp) {
            //       break;
            //     }
            //   }
            // }
            unsigned L = pool.capacity() - 1;
            unsigned poolL = pool.size();
            if (is_insert) {
                if (poolL <= L) {
                    pool.resize(poolL + 1);
                    InsertIntopool(pool.data(), poolL, Neighbor(id, dist, true));
                } else {
                    InsertIntopool(pool.data(), L, Neighbor(id, dist, true));
                }
            }
            // pool.reserve(L + 1);
            // if (pool.size() > L) {pool.resize(L);}
        }

        template<typename C>
        void join(C callback) const {
            for (unsigned const i: nn_new) {
                for (unsigned const j: nn_new) {
                    if (i < j) {
                        callback(i, j);
                    }
                }
                for (unsigned j: nn_old) {
                    callback(i, j);
                }
            }
        }
    };

    struct LockNeighbor {
        std::mutex lock;
        std::vector<Neighbor> pool;
    };

    static inline int InsertIntoPool(Neighbor *addr, unsigned K, Neighbor nn) {
        // find the location to insert
        int left = 0, right = K - 1;
        if (addr[left].distance > nn.distance) {
            memmove((char *) &addr[left + 1], &addr[left], K * sizeof(Neighbor));
            addr[left] = nn;
            return left;
        }
        if (addr[right].distance < nn.distance) {
            addr[K] = nn;
            return K;
        }
        while (left < right - 1) {
            int mid = (left + right) / 2;
            if (addr[mid].distance > nn.distance)right = mid;
            else left = mid;
        }
        //check equal ID

        while (left > 0) {
            if (addr[left].distance < nn.distance) break;
            if (addr[left].id == nn.id) return K + 1;
            left--;
        }
        if (addr[left].id == nn.id || addr[right].id == nn.id)return K + 1;
        memmove((char *) &addr[right + 1], &addr[right], (K - right) * sizeof(Neighbor));
        addr[right] = nn;
        return right;
    }
}


#endif //WEAVESS_NEIGHBOR_H

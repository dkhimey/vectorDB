#include <iostream>
#include <random>
#include <queue>
#include <algorithm>
#include <vector>
#include <set>
#include <functional>
#include "external/hnswlib/hnswlib/hnswlib.h"
#include "external/KaHIP/interface/kaHIP_interface.h"

#define KMEANS_EPOCHS 10

class Pyramid {
private:
    float* X;
    size_t max_elements;
    size_t dim;
    size_t w_partitions;

    size_t ef_construction;
    hnswlib::SpaceInterface<float>* space;
    std::function<float(float*, float*)> computeDistance;
    bool normalize;

    std::vector<int> partitions;
    std::vector<float*>* X_partitions;

    hnswlib::HierarchicalNSW<float>* meta_HNSW;
    hnswlib::HierarchicalNSW<float>** sub_HNSWs;
    
    std::mt19937 gen;


    void getSample(float* sample, size_t nPrime) {

        std::vector<int> indices(max_elements);
        std::iota(indices.begin(), indices.end(), 0);

        // Random sampling of indeces without replacement
        std::vector<int> sampled_indices;
        std::sample(indices.begin(), indices.end(), std::back_inserter(sampled_indices),
                    nPrime, gen);

        // copy vectors into sample
        for (int i = 0; i < nPrime; ++i) {
            int idx = sampled_indices[i] * dim;
            std::copy(X + idx, X + idx + dim, sample + i * dim);
        }
    }

    float computeEucleadianDistance(float* a, float* b) {
        float dist = 0.0f;
        for (size_t i = 0; i < dim; i++) {
            dist += (a[i] - b[i]) * (a[i] - b[i]);
        }
        return dist;
    }

    float computeInnerProductDistance(float* a, float* b) {
        float dot = 0.0f;
        for (size_t i = 0; i < dim; i++) {
            dot += a[i] * b[i];
        }
        return 1.0 - dot;
    }

    float computeCosineSimilarity(float* a, float* b) {
        float dot = 0.0f, normA = 0.0f, normB = 0.0f;
        for (size_t i = 0; i < dim; i++) {
            dot += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }
        return dot / (std::sqrt(normA) * std::sqrt(normB)); 
    }

    void normalizeVec(float* data, float* norm_array) {
        float norm = 0.0f;
        for (int i = 0; i < dim; i++)
            norm += data[i] * data[i];
        norm = 1.0f / (sqrtf(norm) + 1e-30f);
        for (int i = 0; i < dim; i++)
            norm_array[i] = data[i] * norm;
    }

    void printVector(float* v, size_t d) {
        for (size_t i = 0; i < d; i++) {
            std::cout << v[i] << " ";
        }
        std::cout << std::endl;
    }

    bool compareVector(float* v1, float* v2, size_t d) {
        for (size_t i = 0; i < d; i++) {
            if (v1[i] != v2[i]) return false;
        }
        return true;
    }

    std::vector<float*> alsh(float* vec, size_t r) {
        // TODO: can probably pass a pointer and push directly to existing list
        std::vector<float*> result;

        return result;
    }

    void kmeans(float* sample, size_t nPrime, size_t m_centers, float* centers) {
        // Initialize centers to the first m vectors from sample
        std::copy(sample, sample + m_centers * dim, centers);

        for (int iter = 0; iter < KMEANS_EPOCHS; iter++) {
            // New centers initialized to 0
            std::vector<float> newCenters(m_centers * dim, 0.0f);
            std::vector<int> counts(m_centers, 0);

            for (size_t i = 0; i < nPrime; i++) {
                float* point = sample + i*dim;

                size_t bestCenter = 0;
                // distance beteween point and centers[0]
                float bestDist = computeDistance(point, centers);
                for (size_t c = 1; c < m_centers; c++) {
                    float dist = computeDistance(point, centers + c*dim);
                    if (dist < bestDist) {
                        bestDist = dist;
                        bestCenter = c;
                    }
                }

                float* newCenter = newCenters.data() + bestCenter*dim;
                for (size_t d = 0; d < dim; d++) {
                    newCenter[d] += point[d];
                }
                counts[bestCenter]++;
            }

            // update centers
            for (size_t c = 0; c < m_centers; c++) {
                float* center = centers + c*dim;
                float* newCenter = newCenters.data() + c*dim;
                if (counts[c] > 0) {
                    for (size_t d = 0; d < dim; d++) {
                        center[d] = newCenter[d] / counts[c];
                    }
                    if (normalize) normalizeVec(center, center);
                }
            }
        }
    }

public:
    Pyramid(float* X, size_t max_elements, size_t dim, size_t w_partitions, size_t ef_construction, bool mips = false) : 
            X(X), max_elements(max_elements), dim(dim), 
            w_partitions(w_partitions), ef_construction(ef_construction),
            gen(42) {
            // gen(std::random_device{}()) {
                sub_HNSWs = new hnswlib::HierarchicalNSW<float>*[w_partitions];
                X_partitions = new std::vector<float*>[w_partitions];

                if (mips) {
                    space = new hnswlib::InnerProductSpace(dim);
                    normalize = true;
                    computeDistance = [this](float* a, float* b) { return computeInnerProductDistance(a, b); };
                } else {
                    space = new hnswlib::L2Space(dim);
                    normalize = false;
                    computeDistance = [this](float* a, float* b) { return computeEucleadianDistance(a, b); };
                }
            }

    ~Pyramid() {
        delete meta_HNSW;
        for (int i = 0; i < w_partitions; i++){
            delete sub_HNSWs[i];
        }
        delete[] sub_HNSWs;
        delete[] X_partitions;
        delete space;
    }

    void buildPyramid(size_t nPrime,
                       size_t M_meta,
                       size_t m_centers,
                       bool mips = false,
                       bool check_recalls = false) {
        // 1. sample n' items from dataset
        float* sample = new float[nPrime * dim];
        getSample(sample, nPrime);

        // 2. run k-means with m centers
        float* centers = new float[m_centers * dim];

        if (mips) {
            float* norm_sample = new float[nPrime * dim];
            for (size_t i = 0; i < nPrime; i++) {
                normalizeVec(sample + i * dim, norm_sample + i * dim);
            }
            kmeans(norm_sample, nPrime, m_centers, centers);
            delete[] norm_sample;
        }
        else kmeans(sample, nPrime, m_centers, centers);

        delete[] sample;

        // 3. build meta-HNSW on the centers
        meta_HNSW = new hnswlib::HierarchicalNSW<float>(space, m_centers, M_meta, ef_construction);
        for (int i = 0; i < m_centers; i++) {
            meta_HNSW->addPoint(centers + i * dim, i);
        }
        
        if (check_recalls) {
            // check the HNSW recall
            float correct = 0;
            for (int i = 0; i < m_centers; i++) {
                std::priority_queue<std::pair<float, hnswlib::labeltype>> result = meta_HNSW->searchKnn(centers + i * dim, 1);
                hnswlib::labeltype label = result.top().second;
                if (label == i) correct++;
            }
            float recall = correct / m_centers;
            std::cout << "Recall: " << correct << "/" << m_centers << " = " << recall << "\n";
            printf("    Levels: %i\n", meta_HNSW->maxlevel_);
        }

        delete[] centers;

        // 4. partition the bottom layer into w partitions
        std::vector<int> xadj;
        std::vector<int> adjncy;
        int pos = 0;
        for (int i = 0; i < m_centers; i++) {
            xadj.push_back(pos);

            auto bottom = meta_HNSW->get_linklist_at_level(i, 0); 
            int nlinks = meta_HNSW->getListCount(bottom);
            hnswlib::tableint *links = (hnswlib::tableint *) (bottom + 1);
            for (int j = 0; j < nlinks; j++) {
                hnswlib::tableint link = links[j];
                adjncy.push_back(link);
                pos++;
            }
        }
        xadj.push_back(pos);

        int edge_cut = 0; // TODO
        int m_centers_int = (int) m_centers;
        int w_partitions_int = (int) w_partitions;
        double imbalance = 0.03; // TODO
        partitions = std::vector<int>(m_centers, -1);
        // run Karlsuhe partitioning algorithm
        kaffpa(&m_centers_int, nullptr, xadj.data(), nullptr, adjncy.data(), 
               &w_partitions_int, &imbalance, true, gen(), FAST, &edge_cut, partitions.data());   
               
        // 5. assign each vector to partition
        float* norm_element = new float;
        for (size_t i = 0; i < max_elements; i++) {
            std::priority_queue<std::pair<float, hnswlib::labeltype>> result;
            if (normalize) {
                normalizeVec(X + i * dim, norm_element);
                result = meta_HNSW->searchKnn(norm_element, 1);
            } else {
                result = meta_HNSW->searchKnn(X + i * dim, 1);
            }
            hnswlib::labeltype label = result.top().second;
            X_partitions[partitions[label]].push_back(X + i * dim); //WARNING: adding non-normed vec!!
        }
        delete norm_element;

        // 5.5. MIPs ONLY 
        //      for each vector in partition i, 
        //      find the top r MIPs neighbors in dataset, 
        //      add neighbors to X_i
        if (mips) {
            for (size_t i = 0; i < w_partitions; i++) {
                int n = X_partitions[i].size();
                for (int j = 0; j < n; j++) {
                    // find top r neighbors
                    int r = w_partitions;
                    for (float* neighbor: alsh(X_partitions[i][j], r)) {
                        X_partitions[i].push_back(neighbor);
                    }
                }
            }
        }

        printf("MIPS success here\n");
        if (mips) return;

        // 6. build sub_HNSW on each partition X_i
        size_t M_sub = M_meta; // TODO 
        for (size_t i = 0; i < w_partitions; i++) {
            sub_HNSWs[i] = new hnswlib::HierarchicalNSW<float>(space, X_partitions[i].size(), M_sub, ef_construction);
            for (int j = 0; j < X_partitions[i].size(); j++) {
                sub_HNSWs[i]->addPoint(X_partitions[i][j], j);
            }
        }

        if (check_recalls) {
            for (size_t i = 0; i < w_partitions; i++) {
                float correct = 0;
                for (int j = 0; j < X_partitions[i].size(); j++) {
                    std::priority_queue<std::pair<float, hnswlib::labeltype>> result = sub_HNSWs[i]->searchKnn(X_partitions[i][j], 1);
                    hnswlib::labeltype label = result.top().second;
                    if (label == j) correct++;
                }
                float recall = correct / X_partitions[i].size();
                std::cout << "Recall p =" << i << ": " << correct << "/" << X_partitions[i].size() << " = " << recall << "\n";
                printf("    Partition Levels: %i\n", sub_HNSWs[i]->maxlevel_);
            }
        }
    }

    void searchPyramid(float* element, int n_meta, int n_sub) {
        std::cout << "Looking for: ";
        printVector(element, dim);
        std::priority_queue<std::pair<float, hnswlib::labeltype>> partition_result = meta_HNSW->searchKnn(element, n_meta);

        std::set<int> searched;
        while (!partition_result.empty()) {
            hnswlib::labeltype part = partition_result.top().second;
            partition_result.pop();

            if (searched.count(partitions[part]) > 0) continue;
            searched.insert(partitions[part]);

            std::priority_queue<std::pair<float, hnswlib::labeltype>> sub_result = sub_HNSWs[partitions[part]]->searchKnn(element, n_sub);
            while (!sub_result.empty()) {
                hnswlib::labeltype label = sub_result.top().second;
                float dist = sub_result.top().first;
                if (dist == 0){
                    std::cout << "partition - " << partitions[part] << " match: ";
                    printVector(X_partitions[partitions[part]][label], dim);
                }
                sub_result.pop();
            }
        }
    }

    void testPyramid() {
        // cross level search
        float multi_correct = 0;
        for (size_t i = 0; i < max_elements; i++) {
            bool found = false;
            std::priority_queue<std::pair<float, hnswlib::labeltype>> partition_result = meta_HNSW->searchKnn(X + i * dim, 3);
            while (!partition_result.empty()) {
                hnswlib::labeltype label = partition_result.top().second;
                partition_result.pop();
                std::priority_queue<std::pair<float, hnswlib::labeltype>> sub_result = sub_HNSWs[partitions[label]]->searchKnn(X + i * dim, 100);
                while (!sub_result.empty()) {
                    hnswlib::labeltype label = sub_result.top().second;
                    float dist = sub_result.top().first;
                    if (dist == 0){
                        found = true;
                    }
                    sub_result.pop();
                }
            }
            if (found) multi_correct++;
        }
        std::cout << "Multi Recall: " << multi_correct << "/" << max_elements << " = " << multi_correct/max_elements << "\n";
    }
};

int main(){

    size_t dim = 128;               // Dimension of the elements
    size_t max_elements = 10000;   // Maximum number of elements, should be known beforehand
    size_t M_meta = 4;             // Tightly connected with internal dimensionality of the data
                                // strongly affects the memory consumption
    size_t ef_construction = 200;  // Controls index search speed/build speed tradeoff
    size_t m_centers = 50; // meta-HNSW size (number of centers)
    size_t w_partitions = 10; // number of sub-HNSWs (number of partitions in the bottom layer of the meta-HNSW)
    size_t nPrime = max_elements/10; // number to sample from full dataset when choosing centers

    // Generate random data
    std::mt19937 rng;
    rng.seed(42);
    // rng.seed(std::random_device{}());
    std::uniform_real_distribution<> distrib_real;
    float* X = new float[max_elements * dim];
    for (int i = 0; i < max_elements * dim; i++) {
        X[i] = distrib_real(rng);
    }

    Pyramid p(X, max_elements, dim, w_partitions, ef_construction);
    p.buildPyramid(nPrime, M_meta, m_centers, false, true);
    // float* element = new float[dim];
    // std::copy(X, X + dim, element);
    // p.searchPyramid(X, 5, 10);

    p.testPyramid();
    delete[] X;
}
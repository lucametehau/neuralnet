#include "neuralnet.h"
#include "data.h"
#include <cstring>
#include <thread>
#include <unordered_map>
#include <random>
#include <mutex>

using namespace std;

const int DATASET_SIZE = (int)6e8;

mt19937_64 gen(0xBEEF);
uniform_int_distribution <uint64_t> rng;

namespace chessTraining {

    int ind = 0;

    void readDataset(FILE* bin_file, Dataset& dataset, int dataSize) {
        double start = clock();

        for (int id = 0; id < dataSize; id++) {
            NetInput inp;
            float score;
            if (!fread(&inp, sizeof(NetInput), 1, bin_file))
                break;

            

            fread(&score, sizeof(float), 1, bin_file);

            if (id % 100000 == 0)
                cout << "Position #" << id << ", score = " << score << ", input king WHITE " << int(inp.kingSq[WHITE]) << ", time passed " << (clock() - start) / CLOCKS_PER_SEC << "s\n";

            dataset.input[dataset.nr] = inp;
            dataset.output[dataset.nr] = score;
            dataset.nr++;
        }
    }
};

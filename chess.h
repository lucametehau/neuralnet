#include "neuralnet.h"
#include "data.h"
#include <cstring>
#include <thread>
#include <unordered_map>
#include <random>
#include <mutex>

using namespace std;

const int DATASET_SIZE = (int)1e9;

mt19937_64 gen(0xBEEF);
uniform_int_distribution <uint64_t> rng;

namespace table {
    const int MB = (1 << 20);

    uint64_t entries;
    vector <uint64_t> hashTable;

    uint64_t hashKey[12][64];

    void init() {
        srand(time(0));
        for (int i = 0; i < 12; i++) {
            for (int j = 0; j < 64; j++)
                hashKey[i][j] = rng(gen);
        }
    }

    uint64_t hashInput(NetInput input) {
        uint64_t h = 0;
        vector <uint16_t> input_v/* = input.toInput()*/;

        for (auto& i : input_v) {
            h ^= hashKey[i >> 8][i & 63];
        }

        return h;
    }

    void init(uint64_t sz) {
        entries = sz * MB;

        hashTable.resize(entries);
    }

    bool seen(uint64_t h) {
        if (hashTable[h & (entries - 1)] == h)
            return 1;

        hashTable[h & (entries - 1)] = h;
        return 0;
    }
}

namespace chessTraining {
    int fileInd;

    void readDataset(Dataset &dataset, int dataSize, string path) {
        ifstream in(path);

        int positions = 0;

        fileInd++;

        char stm;
        string fen, line;

        for (int id = 0; id < dataSize && getline(in, line); id++) {
            if (in.eof() || dataset.nr >= DATASET_SIZE) {
                //cout << id << "\n";
                break;
            }

            int p = line.find(" "), p2;

            fen = line.substr(0, p + 2);

            stm = line[p + 1];

            p = line.find("[") + 1;
            p2 = line.find("]");

            string res = line.substr(p, p2 - p);
            float gameRes;

            if (res == "0")
                gameRes = 0;
            else if (res == "0.5")
                gameRes = 0.5;
            else
                gameRes = 1;

            int evalNr = 0, sign = 1;

            p = p2 + 2;

            if (line[p] == '-')
                sign = -1, p++;

            while (p < (int)line.size())
                evalNr = evalNr * 10 + line[p++] - '0';

            evalNr *= sign;

            if (stm == 'b') {
                //evalNr *= -1;
                gameRes = 1.0 - gameRes;
            }


            float eval = 1.0 / (1.0 + exp(-1.0 * evalNr * SIGMOID_SCALE));
            float score = EVAL * eval + GAME_RES * gameRes;

            NetInput inp = fenToInput(fen);

            if (__builtin_popcountll(inp.occ) == 3) {
                //cout << fen << " " << evalNr << " " << gameRes << '\n';
            }

            //cout << line << " " << stm << " " << gameRes << " " << eval << "\n";

            //uint64_t h = table::hashInput(inp);

            if (true) {
                positions++;

                dataset.input[dataset.nr] = inp;
                dataset.output[dataset.nr] = score;
                dataset.nr++;
            }
        }
    }

    void readMultipleDatasets(Dataset &dataset, int dataSize, string path, int nrFiles) {
        vector <string> paths(nrFiles);

        //nrThreads = 1;

        for (int i = 0; i < nrFiles; i++) {
            paths[i] = path;
            if (i < 10)
                paths[i] += char(i + '0');
            else
                paths[i] += char(i / 10 + '0'), paths[i] += char(i % 10 + '0');
            paths[i] += ".txt"; /// assuming all files have this format
        }

        float startTime = clock();
        int temp = dataset.nr;

        cout << "Reading files from: " << path << "\n";
        for (int t = 0; t < nrFiles; t++) {
            readDataset(dataset, dataSize, paths[t]);
            cout << "Done with file #" << t << "\n";
        }

        cout << (clock() - startTime) / CLOCKS_PER_SEC << " seconds for loading "
            << (int)dataset.nr - temp << " files\n";
    }
}

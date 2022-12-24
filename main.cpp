#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <random>
#include <ctime>
#include <cassert>
#include <queue>
#include "chess.h"

using namespace std;

struct Position {
    double eval;
    double res;
};

double sig(double val, double K) {
    return 1.0 / (1.0 + exp(-K * val));
}

double calcError(vector <Position> & positions, double K) {
    double error = 0;

#pragma omp parallel for schedule(auto) num_threads(8) reduction(+ : error)
    for (int i = 0; i < (int)positions.size(); i++) {
        error += pow(positions[i].res - sig(positions[i].eval, K), 2);
    }

    cout << error << "\n";

    return error / positions.size();
}

double findBestK(vector <string> paths) {

    NN.load("Clover_3_3_550mil_e38.nn");

    vector <Position> positions;
    Position pos;

    for (auto& path : paths) {

        ifstream stream;

        stream.open(path, ifstream::in);

        if (!stream.is_open())
            continue;

        string line;
        int nrPos = 0;

        while (getline(stream, line)) {
            int p = line.find(" "), p2;

            string fen = line.substr(0, p + 2);
            char stm = line[p + 1];

            p = line.find("[") + 1;
            p2 = line.find("]");

            string res = line.substr(p, p2 - p);
            double gameRes;

            if (res == "0")
                gameRes = 0;
            else if (res == "0.5")
                gameRes = 0.5;
            else
                gameRes = 1;

            GoodNetInput input_v;
            NetInput inp = fenToInput(fen);
            OutputValues o;
            setInput(input_v, inp);


            pos.res = (stm == 'b' ? 1.0 - gameRes : gameRes);

            pos.eval = int(NN.inverseSigmoid(NN.feedForward(input_v, o)));

            positions.push_back(pos);

            nrPos++;


            if (nrPos % 1000000 == 0) {
                cout << nrPos << " loaded\n";
                
                if (nrPos)
                    cout << fen << " " << stm << " " << positions.back().eval << " " << positions.back().res << "\n";
                
            }
        }
    }

    double K = 0;
    double mn = numeric_limits <double>::max();

    cout.precision(15);

    for (int i = 0; i <= 10; i++) {
        cout << "iteration " << i + 1 << "\n";
        double unit = pow(10, -i), range = 10.0 * unit, r = K + range, best = K;

        for (double curr = max(0.0, K - range); curr <= r; curr += unit) {
            double error = calcError(positions, curr);

            cout << "curr K = " << curr << ", error = " << error << '\n';

            if (error < mn) {
                mn = error;
                best = curr;
            }
        }

        K = best;
    }
    return K;
}

Dataset dataset(DATASET_SIZE);
bool find_K = false;

int main() {
    
    if (find_K) {
        vector <string> paths;

        for (int i = 0; i < 16; i++) {
            string path = "C:\\Users\\Luca\\Desktop\\CloverData\\CloverData_3_3_v1_";

            if (i > 9)
                path += char(i / 10 + '0'), path += char(i % 10 + '0');
            else
                path += char(i + '0');

            path += ".txt";
            paths.push_back(path);
        }
        for (int i = 0; i < 16; i++) {
            string path = "C:\\Users\\Luca\\Desktop\\CloverData\\CloverData_3_2_v3_";

            if (i > 9)
                path += char(i / 10 + '0'), path += char(i % 10 + '0');
            else
                path += char(i + '0');

            path += ".txt";
            paths.push_back(path);
        }
        for (int i = 0; i < 16; i++) {
            string path = "C:\\Users\\Luca\\Desktop\\CloverData\\CloverData_3_2_v2_";

            if (i > 9)
                path += char(i / 10 + '0'), path += char(i % 10 + '0');
            else
                path += char(i + '0');

            path += ".txt";
            paths.push_back(path);
        }
        for (int i = 0; i < 16; i++) {
            string path = "C:\\Users\\Luca\\Desktop\\CloverData\\CloverData_3_2_v1_";

            if (i > 9)
                path += char(i / 10 + '0'), path += char(i % 10 + '0');
            else
                path += char(i + '0');

            path += ".txt";
            paths.push_back(path);
        }
        double K = findBestK(paths);

        cout << K << "\n";
        return 0;
    }
    

    vector <NetInput> input;
    vector <float> output;

    //table::init(128);

    int dataSize, batchSize, nrEpochs, nrThreads;
    float split;

    cin >> dataSize >> batchSize >> nrEpochs >> nrThreads >> split;

    const string path = "C:\\Users\\Luca\\Desktop\\CloverData\\CloverData.bin";
    FILE* bin_file = fopen(path.c_str(), "rb");

    chessTraining::readDataset(bin_file, dataset, dataSize);

    {
        //chessTraining::readMultipleDatasets(input, output, dataSize, "C:\\Users\\Luca\\source\\repos\\CloverEngine\\CloverData", 16);
        
        /*chessTraining::readMultipleDatasets(dataset, dataSize, "C:\\Users\\Luca\\source\\repos\\CloverEngine\\CloverData_d9_", 16);
        chessTraining::readMultipleDatasets(dataset, dataSize, "C:\\Users\\Luca\\source\\repos\\CloverEngine\\CloverData_d9_2_", 16);
        chessTraining::readMultipleDatasets(dataset, dataSize, "C:\\Users\\Luca\\source\\repos\\CloverEngine\\CloerData_d9_3_", 16);
        chessTraining::readMultipleDatasets(dataset, dataSize, "C:\\Users\\Luca\\source\\repos\\CloverEngine\\CloverData_d9_4", 16);*/
        
        //chessTraining::readMultipleDatasets(dataset, dataSize, "C:\\Users\\Luca\\source\\repos\\CloverEngine\\Clover.3.0_data_d9", 16);
        //chessTraining::readMultipleDatasets(dataset, dataSize, "C:\\Users\\Luca\\source\\repos\\CloverEngine\\Clover3.0_data_d9_2_", 16);
        //chessTraining::readMultipleDatasets(dataset, dataSize, "C:\\Users\\Luca\\source\\repos\\CloverEngine\\Clover3.0_data_d9_3_", 16);
        //chessTraining::readMultipleDatasets(dataset, dataSize, "C:\\Users\\Luca\\source\\repos\\CloverEngine\\Clover_3_0_data_d9_4_", 16);
        //chessTraining::readMultipleDatasets(dataset, dataSize, "C:\\Users\\Luca\\source\\repos\\CloverEngine\\Clover_3_0_data_d9_5_", 16);
        
        //chessTraining::readMultipleDatasets(dataset, dataSize, "C:\\Users\\Luca\\Desktop\\CloverData\\Clover_3_1_data_d9_", 16);
        //chessTraining::readMultipleDatasets(dataset, dataSize, "C:\\Users\\Luca\\Desktop\\CloverData\\Clover_3_1_d10_", 16); // it's actually depth 9 oops
        //chessTraining::readMultipleDatasets(dataset, dataSize, "C:\\Users\\Luca\\Desktop\\CloverData\\Clover_3_1_d9_v3_", 16);
    }

    cout << sizeof(Network) << "\n";

    runTraining(dataset, dataset.nr, batchSize, nrEpochs, nrThreads, split,
        "Clover_3_2_410mil_e15_kingside.nn", "d.nn", false, true);
    return 0;
}

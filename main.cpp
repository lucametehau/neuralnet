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

int main() {
  vector <LayerInfo> topology;

  topology.push_back({768, NO_ACTIV});
  topology.push_back({512, RELU});
  topology.push_back({1, SIGMOID});

  vector <NetInput> input;
  vector <float> output;

  table::init();
  table::init(128);

  int dataSize, batchSize, nrEpochs, nrThreads;
  float split;

  cin >> dataSize >> batchSize >> nrEpochs >> nrThreads >> split;

  {
    //chessTraining::readMultipleDatasets(input, output, dataSize, "C:\\Users\\Luca\\source\\repos\\CloverEngine\\CloverData", 16);
    chessTraining::readMultipleDatasets(input, output, dataSize, "C:\\Users\\Luca\\source\\repos\\CloverEngine\\CloverData_d9_", 16);
    chessTraining::readMultipleDatasets(input, output, dataSize, "C:\\Users\\Luca\\source\\repos\\CloverEngine\\CloverData_d9_2_", 16);
    chessTraining::readMultipleDatasets(input, output, dataSize, "C:\\Users\\Luca\\source\\repos\\CloverEngine\\CloerData_d9_3_", 16);
    chessTraining::readMultipleDatasets(input, output, dataSize, "C:\\Users\\Luca\\source\\repos\\CloverEngine\\CloverData_d9_4", 16);
  }

  runTraining(topology, input, output, (int)input.size(), batchSize, nrEpochs, nrThreads, split,
              "a.nn", "b.nn", false, true);
  return 0;
}

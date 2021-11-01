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
  topology.push_back({256, RELU});
  topology.push_back({1, SIGMOID});

  vector <NetInput> input;
  vector <float> output;

  table::init();
  table::init(128);

  int dataSize, batchSize, nrEpochs, nrThreads;
  float split;

  cin >> dataSize >> batchSize >> nrEpochs >> nrThreads >> split;

  {
    chessTraining::readMultipleDatasets(input, output, dataSize, "C:\\Users\\LMM\\Desktop\\CloverEngine-2.4\\CloverEngine-2.4\\CloverData11", 8);
    chessTraining::readMultipleDatasets(input, output, dataSize, "C:\\Users\\LMM\\Desktop\\CloverEngine-2.4\\CloverEngine-2.4\\CloverData.12", 8);
    chessTraining::readMultipleDatasets(input, output, dataSize, "C:\\Users\\LMM\\Desktop\\CloverEngine-2.4\\CloverEngine-2.4\\CloverData13", 8);
    /*chessTraining::readMultipleDatasets(input, output, dataSize, "C:\\Users\\LMM\\Desktop\\probleme_info\\chess\\CloverData3", 8);
    chessTraining::readMultipleDatasets(input, output, dataSize, "C:\\Users\\LMM\\Desktop\\probleme_info\\chess\\CloverData4", 8);
    chessTraining::readMultipleDatasets(input, output, dataSize, "C:\\Users\\LMM\\Desktop\\probleme_info\\chess\\CloverData5", 8);
    chessTraining::readMultipleDatasets(input, output, dataSize, "C:\\Users\\LMM\\Desktop\\probleme_info\\chess\\CloverData6", 8);
    chessTraining::readMultipleDatasets(input, output, dataSize, "C:\\Users\\LMM\\Desktop\\probleme_info\\chess\\CloverData7", 8);
    chessTraining::readMultipleDatasets(input, output, dataSize, "C:\\Users\\LMM\\Desktop\\probleme_info\\chess\\CloverData8", 8);
    chessTraining::readMultipleDatasets(input, output, dataSize, "C:\\Users\\LMM\\Desktop\\probleme_info\\chess\\CloverData9", 8);*/
  }

  runTraining(topology, input, output, (int)input.size(), batchSize, nrEpochs, nrThreads, split,
              "a.nn", "b.nn", false, true);
  return 0;
}

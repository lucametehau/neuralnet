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
  vector <double> output;

  table::init();
  table::init(128);

  int dataSize, batchSize, nrEpochs, nrThreads;
  double split;

  cin >> dataSize >> batchSize >> nrEpochs >> nrThreads >> split;

  {
    chessTraining::readMultipleDatasets(input, output, dataSize, "C:\\Users\\LMM\\Desktop\\CloverEngine-2.4\\CloverEngine-2.4\\CloverData11", 8);
    chessTraining::readMultipleDatasets(input, output, dataSize, "C:\\Users\\LMM\\Desktop\\CloverEngine-2.4\\CloverEngine-2.4\\CloverData.12", 8);
    chessTraining::readMultipleDatasets(input, output, dataSize, "C:\\Users\\LMM\\Desktop\\probleme_info\\chess\\CloverData3", 8);
    chessTraining::readMultipleDatasets(input, output, dataSize, "C:\\Users\\LMM\\Desktop\\probleme_info\\chess\\CloverData4", 8);
    chessTraining::readMultipleDatasets(input, output, dataSize, "C:\\Users\\LMM\\Desktop\\probleme_info\\chess\\CloverData5", 8);
    chessTraining::readMultipleDatasets(input, output, dataSize, "C:\\Users\\LMM\\Desktop\\probleme_info\\chess\\CloverData6", 8);
    chessTraining::readMultipleDatasets(input, output, dataSize, "C:\\Users\\LMM\\Desktop\\probleme_info\\chess\\CloverData7", 8);
    chessTraining::readMultipleDatasets(input, output, dataSize, "C:\\Users\\LMM\\Desktop\\probleme_info\\chess\\CloverData8", 8);
    chessTraining::readMultipleDatasets(input, output, dataSize, "C:\\Users\\LMM\\Desktop\\probleme_info\\chess\\CloverData9", 8);

    /*chessTraining::readDataset(input, output, dataSize, "C:\\Users\\LMM\\Desktop\\probleme_info\\chess\\CloverData3.0.txt");
    chessTraining::readDataset(input, output, dataSize, "C:\\Users\\LMM\\Desktop\\probleme_info\\chess\\CloverData3.1.txt");
    chessTraining::readDataset(input, output, dataSize, "C:\\Users\\LMM\\Desktop\\probleme_info\\chess\\CloverData3.2.txt");
    chessTraining::readDataset(input, output, dataSize, "C:\\Users\\LMM\\Desktop\\probleme_info\\chess\\CloverData3.3.txt");
    chessTraining::readDataset(input, output, dataSize, "C:\\Users\\LMM\\Desktop\\probleme_info\\chess\\CloverData3.4.txt");
    chessTraining::readDataset(input, output, dataSize, "C:\\Users\\LMM\\Desktop\\probleme_info\\chess\\CloverData3.5.txt");
    chessTraining::readDataset(input, output, dataSize, "C:\\Users\\LMM\\Desktop\\probleme_info\\chess\\CloverData3.6.txt");
    chessTraining::readDataset(input, output, dataSize, "C:\\Users\\LMM\\Desktop\\probleme_info\\chess\\CloverData3.7.txt");

    chessTraining::readDataset(input, output, dataSize, "C:\\Users\\LMM\\Desktop\\probleme_info\\chess\\CloverData4.0.txt");
    chessTraining::readDataset(input, output, dataSize, "C:\\Users\\LMM\\Desktop\\probleme_info\\chess\\CloverData4.1.txt");
    chessTraining::readDataset(input, output, dataSize, "C:\\Users\\LMM\\Desktop\\probleme_info\\chess\\CloverData4.2.txt");
    chessTraining::readDataset(input, output, dataSize, "C:\\Users\\LMM\\Desktop\\probleme_info\\chess\\CloverData4.3.txt");
    chessTraining::readDataset(input, output, dataSize, "C:\\Users\\LMM\\Desktop\\probleme_info\\chess\\CloverData4.4.txt");
    chessTraining::readDataset(input, output, dataSize, "C:\\Users\\LMM\\Desktop\\probleme_info\\chess\\CloverData4.5.txt");
    chessTraining::readDataset(input, output, dataSize, "C:\\Users\\LMM\\Desktop\\probleme_info\\chess\\CloverData4.6.txt");
    chessTraining::readDataset(input, output, dataSize, "C:\\Users\\LMM\\Desktop\\probleme_info\\chess\\CloverData4.7.txt");

    chessTraining::readDataset(input, output, dataSize, "C:\\Users\\LMM\\Desktop\\probleme_info\\chess\\CloverData5.0.txt");
    chessTraining::readDataset(input, output, dataSize, "C:\\Users\\LMM\\Desktop\\probleme_info\\chess\\CloverData5.1.txt");
    chessTraining::readDataset(input, output, dataSize, "C:\\Users\\LMM\\Desktop\\probleme_info\\chess\\CloverData5.2.txt");
    chessTraining::readDataset(input, output, dataSize, "C:\\Users\\LMM\\Desktop\\probleme_info\\chess\\CloverData5.3.txt");
    chessTraining::readDataset(input, output, dataSize, "C:\\Users\\LMM\\Desktop\\probleme_info\\chess\\CloverData5.4.txt");
    chessTraining::readDataset(input, output, dataSize, "C:\\Users\\LMM\\Desktop\\probleme_info\\chess\\CloverData5.5.txt");
    chessTraining::readDataset(input, output, dataSize, "C:\\Users\\LMM\\Desktop\\probleme_info\\chess\\CloverData5.6.txt");
    chessTraining::readDataset(input, output, dataSize, "C:\\Users\\LMM\\Desktop\\probleme_info\\chess\\CloverData5.7.txt");

    chessTraining::readDataset(input, output, dataSize, "C:\\Users\\LMM\\Desktop\\probleme_info\\chess\\CloverData6.0.txt");
    chessTraining::readDataset(input, output, dataSize, "C:\\Users\\LMM\\Desktop\\probleme_info\\chess\\CloverData6.1.txt");
    chessTraining::readDataset(input, output, dataSize, "C:\\Users\\LMM\\Desktop\\probleme_info\\chess\\CloverData6.2.txt");
    chessTraining::readDataset(input, output, dataSize, "C:\\Users\\LMM\\Desktop\\probleme_info\\chess\\CloverData6.3.txt");
    chessTraining::readDataset(input, output, dataSize, "C:\\Users\\LMM\\Desktop\\probleme_info\\chess\\CloverData6.4.txt");
    chessTraining::readDataset(input, output, dataSize, "C:\\Users\\LMM\\Desktop\\probleme_info\\chess\\CloverData6.5.txt");
    chessTraining::readDataset(input, output, dataSize, "C:\\Users\\LMM\\Desktop\\probleme_info\\chess\\CloverData6.6.txt");
    chessTraining::readDataset(input, output, dataSize, "C:\\Users\\LMM\\Desktop\\probleme_info\\chess\\CloverData6.7.txt");

    chessTraining::readDataset(input, output, dataSize, "C:\\Users\\LMM\\Desktop\\probleme_info\\chess\\CloverData7.0.txt");
    chessTraining::readDataset(input, output, dataSize, "C:\\Users\\LMM\\Desktop\\probleme_info\\chess\\CloverData7.1.txt");
    chessTraining::readDataset(input, output, dataSize, "C:\\Users\\LMM\\Desktop\\probleme_info\\chess\\CloverData7.2.txt");
    chessTraining::readDataset(input, output, dataSize, "C:\\Users\\LMM\\Desktop\\probleme_info\\chess\\CloverData7.3.txt");
    chessTraining::readDataset(input, output, dataSize, "C:\\Users\\LMM\\Desktop\\probleme_info\\chess\\CloverData7.4.txt");
    chessTraining::readDataset(input, output, dataSize, "C:\\Users\\LMM\\Desktop\\probleme_info\\chess\\CloverData7.5.txt");
    chessTraining::readDataset(input, output, dataSize, "C:\\Users\\LMM\\Desktop\\probleme_info\\chess\\CloverData7.6.txt");
    chessTraining::readDataset(input, output, dataSize, "C:\\Users\\LMM\\Desktop\\probleme_info\\chess\\CloverData7.7.txt");

    chessTraining::readDataset(input, output, dataSize, "C:\\Users\\LMM\\Desktop\\probleme_info\\chess\\CloverData8.0.txt");
    chessTraining::readDataset(input, output, dataSize, "C:\\Users\\LMM\\Desktop\\probleme_info\\chess\\CloverData8.1.txt");
    chessTraining::readDataset(input, output, dataSize, "C:\\Users\\LMM\\Desktop\\probleme_info\\chess\\CloverData8.2.txt");
    chessTraining::readDataset(input, output, dataSize, "C:\\Users\\LMM\\Desktop\\probleme_info\\chess\\CloverData8.3.txt");
    chessTraining::readDataset(input, output, dataSize, "C:\\Users\\LMM\\Desktop\\probleme_info\\chess\\CloverData8.4.txt");
    chessTraining::readDataset(input, output, dataSize, "C:\\Users\\LMM\\Desktop\\probleme_info\\chess\\CloverData8.5.txt");
    chessTraining::readDataset(input, output, dataSize, "C:\\Users\\LMM\\Desktop\\probleme_info\\chess\\CloverData8.6.txt");
    chessTraining::readDataset(input, output, dataSize, "C:\\Users\\LMM\\Desktop\\probleme_info\\chess\\CloverData8.7.txt");

    chessTraining::readDataset(input, output, dataSize, "C:\\Users\\LMM\\Desktop\\probleme_info\\chess\\CloverData9.0.txt");
    chessTraining::readDataset(input, output, dataSize, "C:\\Users\\LMM\\Desktop\\probleme_info\\chess\\CloverData9.1.txt");
    chessTraining::readDataset(input, output, dataSize, "C:\\Users\\LMM\\Desktop\\probleme_info\\chess\\CloverData9.2.txt");
    chessTraining::readDataset(input, output, dataSize, "C:\\Users\\LMM\\Desktop\\probleme_info\\chess\\CloverData9.3.txt");
    chessTraining::readDataset(input, output, dataSize, "C:\\Users\\LMM\\Desktop\\probleme_info\\chess\\CloverData9.4.txt");
    chessTraining::readDataset(input, output, dataSize, "C:\\Users\\LMM\\Desktop\\probleme_info\\chess\\CloverData9.5.txt");
    chessTraining::readDataset(input, output, dataSize, "C:\\Users\\LMM\\Desktop\\probleme_info\\chess\\CloverData9.6.txt");
    chessTraining::readDataset(input, output, dataSize, "C:\\Users\\LMM\\Desktop\\probleme_info\\chess\\CloverData9.7.txt");

    chessTraining::readDataset(input, output, dataSize, "C:\\Users\\LMM\\Desktop\\CloverEngine-2.4\\CloverEngine-2.4\\CloverData.12.0.txt");
    chessTraining::readDataset(input, output, dataSize, "C:\\Users\\LMM\\Desktop\\CloverEngine-2.4\\CloverEngine-2.4\\CloverData.12.1.txt");
    chessTraining::readDataset(input, output, dataSize, "C:\\Users\\LMM\\Desktop\\CloverEngine-2.4\\CloverEngine-2.4\\CloverData.12.2.txt");
    chessTraining::readDataset(input, output, dataSize, "C:\\Users\\LMM\\Desktop\\CloverEngine-2.4\\CloverEngine-2.4\\CloverData.12.3.txt");
    chessTraining::readDataset(input, output, dataSize, "C:\\Users\\LMM\\Desktop\\CloverEngine-2.4\\CloverEngine-2.4\\CloverData.12.4.txt");
    chessTraining::readDataset(input, output, dataSize, "C:\\Users\\LMM\\Desktop\\CloverEngine-2.4\\CloverEngine-2.4\\CloverData.12.5.txt");
    chessTraining::readDataset(input, output, dataSize, "C:\\Users\\LMM\\Desktop\\CloverEngine-2.4\\CloverEngine-2.4\\CloverData.12.6.txt");
    chessTraining::readDataset(input, output, dataSize, "C:\\Users\\LMM\\Desktop\\CloverEngine-2.4\\CloverEngine-2.4\\CloverData.12.7.txt");*/
  }

  runTraining(topology, input, output, (int)input.size(), batchSize, nrEpochs, nrThreads, split,
              "a.nn", "b.nn", false, true);
  return 0;
}

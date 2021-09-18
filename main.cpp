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

  int dataSize = 10000, batchSize = 8;
  int epochs = 100;
  double split = 0.1; /// how big is the validation dataset

  chessTraining::readDataset(input, output, dataSize, "C:\\Users\\LMM\\Desktop\\probleme_info\\chess\\CloverData3.0.txt", true);
  chessTraining::readDataset(input, output, dataSize, "C:\\Users\\LMM\\Desktop\\probleme_info\\chess\\CloverData3.1.txt", true);
  chessTraining::readDataset(input, output, dataSize, "C:\\Users\\LMM\\Desktop\\probleme_info\\chess\\CloverData3.2.txt", true);
  chessTraining::readDataset(input, output, dataSize, "C:\\Users\\LMM\\Desktop\\probleme_info\\chess\\CloverData3.3.txt", true);
  chessTraining::readDataset(input, output, dataSize, "C:\\Users\\LMM\\Desktop\\probleme_info\\chess\\CloverData3.4.txt", true);
  chessTraining::readDataset(input, output, dataSize, "C:\\Users\\LMM\\Desktop\\probleme_info\\chess\\CloverData3.5.txt", true);
  chessTraining::readDataset(input, output, dataSize, "C:\\Users\\LMM\\Desktop\\probleme_info\\chess\\CloverData3.6.txt", true);
  chessTraining::readDataset(input, output, dataSize, "C:\\Users\\LMM\\Desktop\\probleme_info\\chess\\CloverData3.7.txt", true);

  chessTraining::readDataset(input, output, dataSize, "C:\\Users\\LMM\\Desktop\\probleme_info\\chess\\CloverData4.0.txt", true);
  chessTraining::readDataset(input, output, dataSize, "C:\\Users\\LMM\\Desktop\\probleme_info\\chess\\CloverData4.1.txt", true);
  chessTraining::readDataset(input, output, dataSize, "C:\\Users\\LMM\\Desktop\\probleme_info\\chess\\CloverData4.2.txt", true);
  chessTraining::readDataset(input, output, dataSize, "C:\\Users\\LMM\\Desktop\\probleme_info\\chess\\CloverData4.3.txt", true);
  chessTraining::readDataset(input, output, dataSize, "C:\\Users\\LMM\\Desktop\\probleme_info\\chess\\CloverData4.4.txt", true);
  chessTraining::readDataset(input, output, dataSize, "C:\\Users\\LMM\\Desktop\\probleme_info\\chess\\CloverData4.5.txt", true);
  chessTraining::readDataset(input, output, dataSize, "C:\\Users\\LMM\\Desktop\\probleme_info\\chess\\CloverData4.6.txt", true);
  chessTraining::readDataset(input, output, dataSize, "C:\\Users\\LMM\\Desktop\\probleme_info\\chess\\CloverData4.7.txt", true);

  chessTraining::readDataset(input, output, dataSize, "C:\\Users\\LMM\\Desktop\\probleme_info\\chess\\CloverData5.0.txt", true);
  chessTraining::readDataset(input, output, dataSize, "C:\\Users\\LMM\\Desktop\\probleme_info\\chess\\CloverData5.1.txt", true);
  chessTraining::readDataset(input, output, dataSize, "C:\\Users\\LMM\\Desktop\\probleme_info\\chess\\CloverData5.2.txt", true);
  chessTraining::readDataset(input, output, dataSize, "C:\\Users\\LMM\\Desktop\\probleme_info\\chess\\CloverData5.3.txt", true);
  chessTraining::readDataset(input, output, dataSize, "C:\\Users\\LMM\\Desktop\\probleme_info\\chess\\CloverData5.4.txt", true);
  chessTraining::readDataset(input, output, dataSize, "C:\\Users\\LMM\\Desktop\\probleme_info\\chess\\CloverData5.5.txt", true);
  chessTraining::readDataset(input, output, dataSize, "C:\\Users\\LMM\\Desktop\\probleme_info\\chess\\CloverData5.6.txt", true);
  chessTraining::readDataset(input, output, dataSize, "C:\\Users\\LMM\\Desktop\\probleme_info\\chess\\CloverData5.7.txt", true);

  chessTraining::readDataset(input, output, dataSize, "C:\\Users\\LMM\\Desktop\\probleme_info\\chess\\CloverData6.0.txt", true);
  chessTraining::readDataset(input, output, dataSize, "C:\\Users\\LMM\\Desktop\\probleme_info\\chess\\CloverData6.1.txt", true);
  chessTraining::readDataset(input, output, dataSize, "C:\\Users\\LMM\\Desktop\\probleme_info\\chess\\CloverData6.2.txt", true);
  chessTraining::readDataset(input, output, dataSize, "C:\\Users\\LMM\\Desktop\\probleme_info\\chess\\CloverData6.3.txt", true);
  chessTraining::readDataset(input, output, dataSize, "C:\\Users\\LMM\\Desktop\\probleme_info\\chess\\CloverData6.4.txt", true);
  chessTraining::readDataset(input, output, dataSize, "C:\\Users\\LMM\\Desktop\\probleme_info\\chess\\CloverData6.5.txt", true);
  chessTraining::readDataset(input, output, dataSize, "C:\\Users\\LMM\\Desktop\\probleme_info\\chess\\CloverData6.6.txt", true);
  chessTraining::readDataset(input, output, dataSize, "C:\\Users\\LMM\\Desktop\\probleme_info\\chess\\CloverData6.7.txt", true);

  chessTraining::readDataset(input, output, dataSize, "C:\\Users\\LMM\\Desktop\\probleme_info\\chess\\CloverData7.0.txt", true);
  chessTraining::readDataset(input, output, dataSize, "C:\\Users\\LMM\\Desktop\\probleme_info\\chess\\CloverData7.1.txt", true);
  chessTraining::readDataset(input, output, dataSize, "C:\\Users\\LMM\\Desktop\\probleme_info\\chess\\CloverData7.2.txt", true);
  chessTraining::readDataset(input, output, dataSize, "C:\\Users\\LMM\\Desktop\\probleme_info\\chess\\CloverData7.3.txt", true);
  chessTraining::readDataset(input, output, dataSize, "C:\\Users\\LMM\\Desktop\\probleme_info\\chess\\CloverData7.4.txt", true);
  chessTraining::readDataset(input, output, dataSize, "C:\\Users\\LMM\\Desktop\\probleme_info\\chess\\CloverData7.5.txt", true);
  chessTraining::readDataset(input, output, dataSize, "C:\\Users\\LMM\\Desktop\\probleme_info\\chess\\CloverData7.6.txt", true);
  chessTraining::readDataset(input, output, dataSize, "C:\\Users\\LMM\\Desktop\\probleme_info\\chess\\CloverData7.7.txt", true);

  chessTraining::readDataset(input, output, dataSize, "C:\\Users\\LMM\\Desktop\\probleme_info\\chess\\CloverData8.0.txt", true);
  chessTraining::readDataset(input, output, dataSize, "C:\\Users\\LMM\\Desktop\\probleme_info\\chess\\CloverData8.1.txt", true);
  chessTraining::readDataset(input, output, dataSize, "C:\\Users\\LMM\\Desktop\\probleme_info\\chess\\CloverData8.2.txt", true);
  chessTraining::readDataset(input, output, dataSize, "C:\\Users\\LMM\\Desktop\\probleme_info\\chess\\CloverData8.3.txt", true);
  chessTraining::readDataset(input, output, dataSize, "C:\\Users\\LMM\\Desktop\\probleme_info\\chess\\CloverData8.4.txt", true);
  chessTraining::readDataset(input, output, dataSize, "C:\\Users\\LMM\\Desktop\\probleme_info\\chess\\CloverData8.5.txt", true);
  chessTraining::readDataset(input, output, dataSize, "C:\\Users\\LMM\\Desktop\\probleme_info\\chess\\CloverData8.6.txt", true);
  chessTraining::readDataset(input, output, dataSize, "C:\\Users\\LMM\\Desktop\\probleme_info\\chess\\CloverData8.7.txt", true);

  int nrInputs = (int)input.size();

  cout << nrInputs << " positions\n";

  /// shuffle training data

  /*vector <int> v(nrInputs);

  for(int i = 0; i < nrInputs; i++)
    v[i] = i;

  random_shuffle(v.begin(), v.end());

  for(int i = 0; i < nrInputs; i++) {
    swap(input[i], input[v[i]]);
    swap(output[i], output[v[i]]);
  }*/

  /*for(int i = 0; i < 100; i++) {
    cout << "Position #" << i << "/100" << "\n";
    for(auto &j : input[i].ind)
      cout << j << " ";
    cout << "\n" << output[i] << "\n";
  }*/

  runTraining(topology, input, output, (int)input.size(), batchSize, epochs, split, "XD.nn", false);
  return 0;
}

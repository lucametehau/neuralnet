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

double inverseSigmoid(double val) {
  return log(val / (1.0 - val)) / SIGMOID_SCALE;
}

int main() {
  vector <LayerInfo> topology;

  topology.push_back({768, NO_ACTIV});
  topology.push_back({256, RELU});
  topology.push_back({1, SIGMOID});

  vector <NetInput> input;
  vector <double> output;

  int dataSize = 2900000;
  int epochs = 10000;
  double LR = 1;
  double split = 0.05;

  Network NN(topology);

  NN.load("Clover_40mil_d8.nn");

  char s[105];

  while(1) {
    cin.getline(s, 100);

    cout << s << "\n";
    string fen = "";
    int ind = 0;
    char realFen[105];

    while(s[ind] != ' ')
      fen += s[ind++];

    strcpy(realFen, fen.c_str());

    cout << realFen << "\n";

    NetInput input = chessTraining::fenToInput(realFen);
    double ans = NN.calc(input);
    cout << ans << "\n";
    cout << inverseSigmoid(ans) << "\n" << 1.0 / (1.0 + exp(-inverseSigmoid(ans) * SIGMOID_SCALE)) << "\n";
  }

  chessTraining::readDataset(input, output, dataSize, "C:\\Users\\LMM\\Desktop\\probleme_info\\chess\\CloverData0.txt", true);
  chessTraining::readDataset(input, output, dataSize, "C:\\Users\\LMM\\Desktop\\probleme_info\\chess\\CloverData1.txt", true);
  chessTraining::readDataset(input, output, dataSize, "C:\\Users\\LMM\\Desktop\\probleme_info\\chess\\CloverData2.txt", true);
  chessTraining::readDataset(input, output, dataSize, "C:\\Users\\LMM\\Desktop\\probleme_info\\chess\\CloverData3.txt", true);
  chessTraining::readDataset(input, output, dataSize, "C:\\Users\\LMM\\Desktop\\probleme_info\\chess\\CloverData4.txt", true);
  chessTraining::readDataset(input, output, dataSize, "C:\\Users\\LMM\\Desktop\\probleme_info\\chess\\CloverData5.txt", true);
  chessTraining::readDataset(input, output, dataSize, "C:\\Users\\LMM\\Desktop\\probleme_info\\chess\\CloverData6.txt", true);
  chessTraining::readDataset(input, output, dataSize, "C:\\Users\\LMM\\Desktop\\probleme_info\\chess\\CloverData7.txt", true);

  chessTraining::readDataset(input, output, dataSize, "C:\\Users\\LMM\\Desktop\\probleme_info\\chess\\CloverData2.0.txt", true);
  chessTraining::readDataset(input, output, dataSize, "C:\\Users\\LMM\\Desktop\\probleme_info\\chess\\CloverData2.1.txt", true);
  chessTraining::readDataset(input, output, dataSize, "C:\\Users\\LMM\\Desktop\\probleme_info\\chess\\CloverData2.2.txt", true);
  chessTraining::readDataset(input, output, dataSize, "C:\\Users\\LMM\\Desktop\\probleme_info\\chess\\CloverData2.3.txt", true);
  chessTraining::readDataset(input, output, dataSize, "C:\\Users\\LMM\\Desktop\\probleme_info\\chess\\CloverData2.4.txt", true);
  chessTraining::readDataset(input, output, dataSize, "C:\\Users\\LMM\\Desktop\\probleme_info\\chess\\CloverData2.5.txt", true);
  chessTraining::readDataset(input, output, dataSize, "C:\\Users\\LMM\\Desktop\\probleme_info\\chess\\CloverData2.6.txt", true);
  chessTraining::readDataset(input, output, dataSize, "C:\\Users\\LMM\\Desktop\\probleme_info\\chess\\CloverData2.7.txt", true);

  cout << input.size() << " positions\n";

  runTraining(topology, input, output, (int)input.size(), epochs, LR, split, "Clover_40mil_d8.nn", false);
  return 0;
}

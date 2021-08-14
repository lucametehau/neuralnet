#include "neuralnet.h"

namespace chessTraining {

};

void chessNN() {
  vector <int> topology;

  topology.push_back(768);
  topology.push_back(512);
  topology.push_back(1);

  vector <vector <double>> input, output;

  chessTraining::readDataset(input, output, )
}

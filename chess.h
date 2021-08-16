#include "neuralnet.h"
#include <cstring>

using namespace std;

namespace chessTraining {

  int cod(char c) {
    bool color = 0;
    if('A' <= c && c <= 'Z') {
      color = 1;
      c += 32;
    }

    int val = 0;

    switch(c) {
    case 'p':
      val = 0;
      break;
    case 'n':
      val = 1;
      break;
    case 'b':
      val = 2;
      break;
    case 'r':
      val = 3;
      break;
    case 'q':
      val = 4;
      break;
    case 'k':
      val = 5;
      break;
    default:
      assert(0);
    }

    return 6 * color + val;
  }

  vector <double> fenToInput(char fen[]) {
    vector <double> input;
    uint64_t bb[12];
    int ind = 0;
    int lg = strlen(fen);

    for(int i = 0; i < 12; i++)
      bb[i] = 0;

    for(int i = 7; i >= 0; i--) {
      int j = 0;
      while(ind < lg && fen[ind] != '/') {
        int sq = 8 * i + j;
        if(fen[ind] < '0' || '9' < fen[ind]) {
          int piece = cod(fen[ind]);

          bb[piece] |= (1ULL << sq);
          j++;
        } else {
          int nr = fen[ind] - '0';
          while(nr)
            j++, sq++, nr--;
        }
        ind++;
      }
      ind++;
    }

    for(int j = 0; j < 12; j++) {
      for(int k = 0; k < 64; k++) {
        input.push_back((bb[j] >> k) & 1);
      }
    }

    return input;
  }

  void readDataset(vector <vector <double>> &input, vector <vector <double>> &output, int dataSize, string path) {
    freopen(path.c_str(), "r", stdin);

    double y;

    char fen[105], a[15], c;

    cout << "Loading data...\n";

    for(int id = 0; id < dataSize; id++) {
      vector <double> inp;
      vector <double> outp;

      if(dataSize > 1000 && id % (dataSize / 1000) == 0) {
        cout << "Index " << id << "/" << dataSize << "\n";
      }

      scanf("%s", fen);

      scanf("%s", a);

      scanf("%s", a);
      scanf("%s", a);
      scanf("%s", a);
      scanf("%s", a);

      inp = fenToInput(fen);

      scanf("%c", &c);
      scanf("%c", &c);
      scanf("%lf", &y);
      scanf("%c", &c);

      outp.push_back(y);

      input.push_back(inp);
      output.push_back(outp);
    }
  }

}

void chessNN() {
  vector <LayerInfo> topology;

  topology.push_back({768, NONE});
  topology.push_back({256, RELU});
  topology.push_back({1, SIGMOID});

  vector <vector <double>> input, output;

  int dataSize = 1000000, batchSize = 128;
  int epochs = 10000;
  double LR = 1;
  double split = 0.1;

  chessTraining::readDataset(input, output, dataSize, "D:\\Downloads\\lichess-big3-resolved\\lichess-big3-resolved.book");

  runTraining(topology, input, output, dataSize, batchSize, epochs, LR, split, "chess.nn", false);
}

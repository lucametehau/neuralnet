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

  NetInput fenToInput(char fen[]) {
    NetInput ans;
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
        if((bb[j] >> k) & 1) {
          ans.ind.push_back(64 * j + k);
        }
      }
    }

    return ans;
  }

  void readDataset(vector <NetInput> &input, vector <vector <float>> &output, int dataSize, string path) {
    freopen(path.c_str(), "r", stdin);

    float y;

    char fen[105], a[15], c;
    NetInput inp;
    vector <float> outp;

    cout << "Loading data...\n";

    for(int id = 0; id < dataSize; id++) {

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
      scanf("%f", &y);
      scanf("%c", &c);

      outp.clear();
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

  vector <NetInput> input;
  vector <vector <float>> output;

  int dataSize = 7100000, batchSize = 2500;
  int epochs = 10000;
  double LR = 1e-1;
  double split = 0.1;

  chessTraining::readDataset(input, output, dataSize, "D:\\Downloads\\lichess-big3-resolved\\lichess-big3-resolved.book");

  runTraining(topology, input, output, dataSize, batchSize, epochs, LR, split, "temp.nn", false);
}

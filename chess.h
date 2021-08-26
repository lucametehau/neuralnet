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
      cout << c << "\n";
      assert(0);
    }

    return 6 * color + val;
  }

  NetInput fenToInput(char fen[]) {
    NetInput ans;
    int ind = 0;

    for(int i = 7; i >= 0; i--) {
      int j = 0;
      while(j < 8 && fen[ind] != '/') {
        int sq = 8 * i + j;
        if(fen[ind] < '0' || '9' < fen[ind]) {
          int piece = cod(fen[ind]);

          ans.ind.push_back(64 * piece + sq);
          j++;
        } else {
          int nr = fen[ind] - '0';
          j += nr;
          sq += nr;
        }
        ind++;
      }
      ind++;
    }

    return ans;
  }

  void readDataset(vector <NetInput> &input, vector <double> &output, int dataSize, string path, bool ownData) {
    freopen(path.c_str(), "r", stdin);

    double gameRes, eval;

    char fen[105], a[15], stm, c;

    cout << "Loading data...\n";

    for(int id = 0; id < dataSize; id++) {

      if(dataSize > 1000 && id % (dataSize / 100) == 0) {
        //cout << "Index " << id << "/" << dataSize << "\n";
      }

      if(feof(stdin)) {
        cout << id << "\n";
        break;
      }

      scanf("%s", fen);

      scanf("%c", &c);
      scanf("%c", &stm);

      scanf("%s", a); scanf("%s", a); scanf("%s", a); scanf("%s", a);

      scanf("%c", &c); scanf("%c", &c);
      scanf("%lf", &gameRes);
      scanf("%c", &c);

      if(ownData) {
        scanf("%lf", &eval);
        if(stm == 'b')
          eval *= -1;
      } else {
        eval = 1505;
      }

      /// use 40% game result, 60% evaluation

      double score = (id < dataSize * 0.4 || eval == 1505 ?
                     gameRes : 1.0 / (1.0 + exp(-eval * SIGMOID_SCALE)));

      input.push_back(fenToInput(fen));
      output.push_back(score);
    }

    cout << "Done loading data from file " << path << "\n";
  }

}

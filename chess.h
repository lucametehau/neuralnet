#include "neuralnet.h"
#include <cstring>
#include <thread>

using namespace std;

namespace chessTraining {
  void readDataset(vector <NetInput> &input, vector <double> &output, int dataSize, string path) {
    freopen(path.c_str(), "r", stdin);

    double gameRes, eval;

    char fen[105], a[15], stm, c;

    for(int id = 0; id < dataSize; id++) {
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

      scanf("%lf", &eval);
      if(stm == 'b')
        eval *= -1;

      eval = 1.0 / (1.0 + exp(-eval * SIGMOID_SCALE));

      /// use 50% game result, 50% evaluation

      double score = eval * 0.5 + gameRes * 0.5;

      input.push_back(fenToInput(fen));
      output.push_back(score);
    }

    //cout << input.size() << " positions for file " << path << "\n";
  }

}

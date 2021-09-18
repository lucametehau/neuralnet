#include "neuralnet.h"
#include <cstring>
#include <thread>

using namespace std;

namespace chessTraining {
  void readDataset(vector <NetInput> &input, vector <double> &output, int dataSize, string path, bool ownData) {
    freopen(path.c_str(), "r", stdin);

    double gameRes, eval;

    char fen[105], a[15], stm, c;

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

    //cout << input.size() << " positions for file " << path << "\n";
  }

}

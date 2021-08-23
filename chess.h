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
    int lg = strlen(fen);

    for(int i = 7; i >= 0; i--) {
      int j = 0;
      while(ind < lg && fen[ind] != '/') {
        int sq = 8 * i + j;
        if(fen[ind] < '0' || '9' < fen[ind]) {
          int piece = cod(fen[ind]);

          ans.ind.push_back(64 * piece + sq);
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

    return ans;
  }

  void readDataset(vector <NetInput> &input, vector <double> &output, int dataSize, string path, bool ownData) {
    freopen(path.c_str(), "r", stdin);

    double gameRes, eval;

    char fen[105], a[15], stm, c;
    NetInput inp;
    vector <FenOutput> outp;

    cout << "Loading data...\n";

    for(int id = 0; id < dataSize; id++) {

      if(dataSize > 1000 && id % (dataSize / 100) == 0) {
        cout << "Index " << id << "/" << dataSize << "\n";
      }

      if(feof(stdin)) {
        cout << id << "\n";
        break;
      }

      scanf("%s", fen);

      inp = fenToInput(fen);

      scanf("%c", &c);
      scanf("%c", &stm);

      scanf("%s", a); scanf("%s", a); scanf("%s", a); scanf("%s", a);

      scanf("%c", &c);
      scanf("%c", &c);
      scanf("%lf", &gameRes);
      scanf("%c", &c);

      if(ownData) {
        scanf("%lf", &eval);
        if(stm == 'b')
          eval *= -1;
      } else {
        eval = 1505;
      }

      double score = (id < dataSize / 2 || eval == 1505 ?
                     gameRes : 1.0 / (1.0 + exp(-eval * SIGMOID_SCALE)));

      input.push_back(inp);
      output.push_back(score);
    }

    cout << "Done loading data from file " << path << "\n";
  }

}

double inverseSigmoid(double val) {
  return log(val / (1.0 - val)) / SIGMOID_SCALE;
}

void chessNN() {
  vector <LayerInfo> topology;

  topology.push_back({768, NONE});
  topology.push_back({256, RELU});
  topology.push_back({1, SIGMOID});

  vector <NetInput> input;
  vector <double> output;

  int dataSize = 2800000, batchSize = 16834;
  int epochs = 10000;
  double LR = 1;
  double split = 0.05;

  Network NN(topology);

  NN.load("chess.nn");

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

  runTraining(topology, input, output, (int)input.size(), batchSize, epochs, LR, split, "chess.nn", false);
}

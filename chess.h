#include "neuralnet.h"
#include <cstring>
#include <thread>
#include <unordered_map>
#include <random>
#include <mutex>

using namespace std;

const int MOD1 = (int)1e9 + 3;
const int MOD2 = (int)1e9 + 7;

mt19937_64 gen(0xBEEF);
uniform_int_distribution <uint64_t> rng;

mutex M;

namespace chessTraining {
  unordered_map <uint64_t, bool> seen;
  int fileInd;

  uint64_t hashKey[12][64];

  void init() {
    srand(time(0));
    for(int i = 0; i < 12; i++) {
      for(int j = 0; j < 64; j++)
        hashKey[i][j] = rng(gen);
    }
  }

  uint64_t hashInput(NetInput input) {
    uint64_t h = 0;

    for(auto &i : input.ind) {
      h ^= hashKey[i >> 8][i & 63];
    }

    return h;
  }

  void readDataset(vector <NetInput> &input, vector <double> &output, int dataSize, string path) {
    ifstream in(path);

    double gameRes, eval;
    int positions = 0;

    fileInd++;

    char fen[105], a[15], stm, c;

    for(int id = 0; id < dataSize; id++) {
      if(in.eof()) {
        cout << id << "\n";
        break;
      }

      in >> fen >> stm >> a >> a >> a >> a >> c >> gameRes >> c >> eval;

      /*M.lock();
      cout << fen << " " << stm << " " << gameRes << " " << eval << "\n";
      M.unlock();*/

      if(stm == 'b')
        eval *= -1;

      eval = 1.0 / (1.0 + exp(-eval * SIGMOID_SCALE));

      /// use 50% game result, 50% evaluation

      double score = eval * 0.5 + gameRes * 0.5;

      NetInput inp = fenToInput(fen);

      M.lock();
      //cout << id << " " << path << "\n";
      uint64_t h = hashInput(inp);

      if(seen.find(h) != seen.end()) {
        /*if(49 <= fileInd && fileInd <= 56)
          cout << fen << "already seen in file index " << seen[h] << "\n";*/
        //cout << id << "\n" << seen[h].first << " " << seen[h].second << "\n" << p.first << " " << p.second << "\n" << fen << " was already seen\n";
        M.unlock();
        continue;
      }

      seen[h] = 1;

      M.unlock();

      positions++;

      input.push_back(inp);
      output.push_back(score);
    }
  }

  void readMultipleDatasets(vector <NetInput> &input, vector <double> &output, int dataSize, string path, int nrThreads) {
    vector <vector <NetInput>> inputs(nrThreads);
    vector <vector <double>> outputs(nrThreads);
    vector <thread> threads(nrThreads);
    vector <string> paths(nrThreads);

    for(int i = 0; i < nrThreads; i++) {
      paths[i] = path + ".";
      paths[i] += char(i + '0');
      paths[i] += ".txt"; /// assuming all files have this format
    }

    int id = 0;

    for(auto &t : threads) {
      t = thread{ readDataset, ref(inputs[id]), ref(outputs[id]), dataSize, paths[id] };
      id++;
    }

    for(auto &t : threads)
      t.join();

    int temp = (int)input.size();

    for(int t = 0; t < nrThreads; t++) {
      for(int i = 0; i < (int)inputs[t].size(); i++) {
        input.push_back(inputs[t][i]);
        output.push_back(outputs[t][i]);
      }
    }

    cout << (int)input.size() - temp << " positions for files at " << path << "\n";
  }
}

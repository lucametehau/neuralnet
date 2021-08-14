#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <random>
#include <ctime>
#include <cassert>
#include <queue>
#include "neuralnet.h"
#pragma GCC optimize("Ofast")

using namespace std;

namespace training {
  /// change these values to adjust training data size
  const int inputSize = 36;
  const int outputSize = 1;

  /// given a 6 by 6 matrix
  /// it returns if there is a path
  /// from the first to the last column
  /// through a 1
  bool solve(vector <int> &v) {
    bool mat[8][8], seen[8][8];
    static int dx[] = {-1, 0, 1, 0};
    static int dy[] = {0, 1, 0, -1};

    for(int i = 0; i < 36; i++)
      mat[i / 6][i % 6] = v[i];

    for(int i = 0; i < 6; i++) {
      for(int j = 0; j < 6; j++)
        seen[i][j] = 0;
    }

    queue <pair <int, int>> q;

    for(int i = 0; i < 6; i++) {
      if(!mat[i][0]) {
        q.push({i, 0});
        seen[i][0] = 1;
      }
    }

    while(!q.empty()) {
      pair <int, int> p = q.front();
      q.pop();

      for(int i = 0; i < 4; i++) {
        int x = p.first + dx[i], y = p.second + dy[i];

        if(0 <= x && x < 6 && 0 <= y && y < 6 && !seen[x][y] && !mat[x][y]) {
          q.push({x, y});
          seen[x][y] = 1;
        }
      }
    }

    bool path = 0;

    for(int i = 0; i < 6; i++)
      path |= seen[i][5];

    return path;
  }

  void createDataset(int size, string path) {
    ofstream out (path);

    cout << "Creating data...\n";

    int correct = 0;

    for(int i = 0; i < size; i++) {
      if(i % (size / 100) == 0)
        cout << "Index " << i << " / " << size << "\n";

      vector <int> inp, outp;
      int x;
      for(int j = 0; j < inputSize; j++) {
        x = tools::bin(tools::gen);
        inp.push_back(x);
        out << x << " ";
      }

      x = solve(inp);
      outp.push_back(x);
      out << x << " ";

      if(x)
        correct++;

      out << "\n";
    }

    cout << correct << " out of " << size << "\n";
  }

  void readDataset(vector <vector <double>> &input, vector <vector <double>> &output, int size, string path) {
    freopen(path.c_str(), "r", stdin);
    double x;

    cout << "Loading data...\n";

    for(int i = 0; i < size; i++) {
      if(i % (size / 100) == 0)
        cout << "Index " << i << " / " << size << "\n";

      vector <double> inp, outp;

      for(int j = 0; j < inputSize; j++) {
        scanf("%lf", &x);
        inp.push_back(x);
      }

      for(int j = 0; j < outputSize; j++) {
        scanf("%lf", &x);
        outp.push_back(x);
      }

      input.push_back(inp);
      output.push_back(outp);
    }
  }
}

void test() {
  bool create = false; /// change this to true if you want a fresh new dataset
  int dataSize = 1000000;
  int epochs = 100000;
  double LR = 0.01;
  double split = 0.1;
  int trainSize = dataSize * (1.0 - split);

  if(create) {
    training::createDataset(dataSize, "training.txt");
    return;
  }

  vector <int> topology;

  topology.push_back(36);
  topology.push_back(12);
  //topology.push_back(6);
  topology.push_back(1);

  Network NN(topology);

  //NN.load("mazesolver_relu.nn");

  vector <vector <double>> input, output;

  training::readDataset(input, output, dataSize, "training.txt");

  /*int acc = 0;

  for(int i = trainSize; i < dataSize; i++) {
    vector <double> ans = NN.calc(input[i]);

    acc += (round(ans[0]) == output[i][0]);

    if(i % 100 == 0) {
      cout << "Index : " << i << "\n";

      cout << "Input : ";
      for(auto &j : input[i])
        cout << j << " ";
      cout << "\n";

      cout << "Output: ";
      for(auto &j : output[i])
        cout << j << " ";
      cout << "\n";

      cout << "Got   : ";
      for(auto &j : ans)
        cout << j << " ";
      cout << "\n";
    }
  }

  cout << "Accuracy: " << 100.0 * acc / (dataSize - trainSize) << "%\n";

  return;*/

  for(int epoch = 1; epoch <= epochs; epoch++) {
    cout << "----------------------------------------- Epoch " << epoch << "/" << epochs << " -----------------------------------------\n";

    double tStart = clock();

    for(int i = 0; i < trainSize; i++) {
      NN.train(input[i], output[i], LR);
    }

    double error = 0;

    for(int i = trainSize; i < dataSize; i++) {
      vector <double> ans = NN.calc(input[i]);

      double delta = (ans[0] - output[i][0]);
      error += delta * delta;
    }

    double tEnd = clock();

    cout << "Raw error : " << error << "\n";

    cout << "Error     : " << error / (2 * (dataSize - trainSize)) << "\n";

    cout << "Time taken: " << (tEnd - tStart) / CLOCKS_PER_SEC << " s\n";

    NN.save("mazesolver_relu.nn");
  }
}

int main() {
  test();
  return 0;
}

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <random>
#include <ctime>
#include <cassert>
#include <queue>
#pragma GCC optimize("Ofast")

using namespace std;

namespace tools {
  mt19937_64 gen(time(0));
  uniform_real_distribution <double> rng(0, 1);
  uniform_int_distribution <int> bin(0, 1);

  vector <double> createRandomArray(int length) {
    vector <double> v;

    for(int i = 0; i < length; i++)
      v.push_back(rng(gen));

    return v;
  }
}

class Layer {
public:
  Layer(int numNeurons, int prevNumNeurons) {
    size = numNeurons;
    bias = tools::createRandomArray(numNeurons);
    output.resize(numNeurons);
    outputDerivative.resize(numNeurons);
    error.resize(numNeurons);
    weights.resize(numNeurons);

    if(prevNumNeurons) {
      for(int i = 0; i < numNeurons; i++) {
        weights[i].resize(prevNumNeurons);
        weights[i] = tools::createRandomArray(prevNumNeurons);
      }
    }
  }

  int size;
  vector <double> bias, output, error, outputDerivative;
  vector <vector <double>> weights;
};

class Network {
public:

  Network(vector <int> &topology) {
    for(int i = 0; i < (int)topology.size(); i++) {
      layers.push_back(Layer(topology[i], (i > 0 ? topology[i - 1] : 0)));
    }
  }

  double activationFunction(double x) {
    return 1.0 / (1.0 + exp(-x));
  }

  double activationFunctionDerivative(double x) {
    return activationFunction(x) * (1 - activationFunction(x));
  }

  vector <double> calc(vector <double> &input) { /// feed forward
    //cout << layers.size() << " " << layers[0].size << " " << input.size() << "\n";
    layers[0].output = input;

    //cout << "a\n";

    for(int l = 1; l < (int)layers.size(); l++) {
      for(int n = 0; n < layers[l].size; n++) {
        double sum = layers[l].bias[n];

        for(int prevN = 0; prevN < layers[l - 1].size; prevN++) {
          sum += layers[l - 1].output[prevN] * layers[l].weights[n][prevN];
        }

        layers[l].output[n] = activationFunction(sum);
        layers[l].outputDerivative[n] = activationFunctionDerivative(sum);
      }
    }

    //cout << "a\n";

    return layers.back().output;
  }

  void backProp(vector <double> &target) {
    /// for output neurons
    for(int n = 0; n < layers.back().size; n++) {
      layers.back().error[n] = (layers.back().output[n] - target[n]) * layers.back().outputDerivative[n];
    }

    /// for hidden layers
    for(int l = (int)layers.size() - 2; l > 0; l--) {
      for(int n = 0; n < layers[l].size; n++) {
        double sum = 0;
        for(int nextN = 0; nextN < layers[l + 1].size; nextN++)
          sum += layers[l + 1].weights[nextN][n] * layers[l + 1].error[nextN];

        layers[l].error[n] = sum * layers[l].outputDerivative[n];
      }
    }
  }

  void updateWeights(double LR) {
    for(int l = 1; l < (int)layers.size(); l++) {
      for(int n = 0; n < layers[l].size; n++) {
        for(int prevN = 0; prevN < layers[l - 1].size; prevN++) {
          double delta = -LR * layers[l - 1].output[prevN] * layers[l].error[n];
          layers[l].weights[n][prevN] += delta;
        }

        /// update bias
        /// no more layers[l - 1].output[prevN], because bias isn't connected to previous neurons
        double delta = -LR * layers[l].error[n];

        layers[l].bias[n] += delta;
      }
    }
  }

  void train(vector <double> &input, vector <double> &target, double LR) {
    if((int)input.size() != layers[0].size) {
      cout << "Wrong input size!\n";
      assert(0);
    }

    if((int)target.size() != layers.back().size) {
      cout << "Wrong output size!\n";
      assert(0);
    }

    calc(input);
    backProp(target);
    updateWeights(LR);
  }

  void write(vector <double> &v, ofstream &out) {
    for(auto &i : v)
      out << i << " ";
    out << "\n";
  }

  vector <double> read(int lg, ifstream &in) {
    vector <double> v;
    double x;
    for(int i = 0; i < lg; i++)
      in >> x, v.push_back(x);
    return v;
  }

  void save(string path) {
    ofstream out (path);
    int cnt = layers.size();

    vector <double> v;

    out << cnt << "\n";

    for(int i = 0; i < (int)layers.size(); i++) {
      write(layers[i].bias, out);
      write(layers[i].output, out);
      write(layers[i].outputDerivative, out);
      write(layers[i].error, out);

      for(int j = 0; j < layers[i].size && i; j++) {
        write(layers[i].weights[j], out);
      }
    }
  }

  void load(string path) {
    ifstream in (path);
    int cnt = layers.size(), cnt2 = 0;

    in >> cnt2;

    if(cnt2 != cnt) {
      cout << "Can't load network!\n";
      cout << "Expected " << cnt << ", got " << cnt2 << "\n";
      assert(0);
    }

    for(int i = 0; i < (int)layers.size(); i++) {
      layers[i].bias = read(layers[i].size, in);
      layers[i].output = read(layers[i].size, in);
      layers[i].outputDerivative = read(layers[i].size, in);
      layers[i].error = read(layers[i].size, in);

      for(int j = 0; j < layers[i].size && i; j++) {
        layers[i].weights[j] = read(layers[i - 1].size, in);
      }
    }
  }

  vector <Layer> layers;
};

namespace training {
  /// change these values to adjust training data size
  const int inputSize = 36;
  const int outputSize = 1;

  /// given a 7 by 7 matrix
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
  double LR = 0.1;

  if(create) {
    training::createDataset(dataSize, "training.txt");
    return;
  }

  vector <int> topology;

  topology.push_back(36);
  topology.push_back(12);
  topology.push_back(6);
  topology.push_back(1);

  Network NN(topology), NN2(topology);

  /// 0.90034 0.00163233 0.00353823 0.00444064 0.786639 0.752504 0.414985 0.290204 0.97584 0.248327 0.659422 0.613187

  //NN2.save("mazesolver.nn");
  NN.load("mazesolver.nn");

  //return;

  for(auto &l : NN.layers)
    cout << l.size << "\n";

  double split = 0.1;
  int trainSize = dataSize * split;
  vector <vector <double>> input, output;

  training::readDataset(input, output, dataSize, "training.txt");

  for(int epoch = 1; epoch <= epochs; epoch++) {
    cout << "----------------------------------------- Epoch " << epoch << "/" << epochs << " -----------------------------------------\n";
    for(int i = 0; i < trainSize; i++) {
      NN.train(input[i], output[i], LR);
    }

    double error = 0;

    for(int i = trainSize; i < dataSize; i++) {
      vector <double> ans = NN.calc(input[i]);

      double delta = (ans[0] - output[i][0]);
      error += delta * delta;
    }

    cout << "Total error: " << error / (2 * (dataSize - trainSize)) << "\n";

    NN.save("mazesolver.nn");
  }

  /*for(int i = trainSize; i < dataSize; i++) {
    vector <double> ans = NN.calc(input[i]);

    if(i % 10 == 0) {
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
  }*/
}

int main() {
  test();
  return 0;
}

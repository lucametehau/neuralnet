#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <random>
#include <ctime>
#include <cassert>
#pragma GCC optimize("Ofast")

using namespace std;

mt19937_64 gen(time(0));
uniform_real_distribution <double> rng(0, 1);

namespace tools {
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
      for(int i = 0; i < numNeurons; i++)
        weights[i] = tools::createRandomArray(prevNumNeurons);
    }
  }

  int size;
  vector <double> bias, output, error, outputDerivative;
  vector <vector <double>> weights;
};

class Network {
public:

  Network(vector <int> &topology) {

    netSize = (int)topology.size();

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
    layers[0].output = input;

    for(int l = 1; l < netSize; l++) {
      for(int n = 0; n < layers[l].size; n++) {
        double sum = layers[l].bias[n];

        for(int prevN = 0; prevN < layers[l - 1].size; prevN++) {
          sum += layers[l - 1].output[prevN] * layers[l].weights[n][prevN];
        }

        layers[l].output[n] = activationFunction(sum);
        layers[l].outputDerivative[n] = activationFunctionDerivative(sum);
      }
    }

    return layers.back().output;
  }

  void backProp(vector <double> &target) {
    /// for output neurons
    for(int n = 0; n < layers.back().size; n++) {
      layers.back().error[n] = (layers.back().output[n] - target[n]) * layers.back().outputDerivative[n];
    }

    /// for hidden layers
    for(int l = netSize - 2; l > 0; l--) {
      for(int n = 0; n < layers[l].size; n++) {
        double sum = 0;
        for(int nextN = 0; nextN < layers[l + 1].size; nextN++)
          sum += layers[l + 1].weights[nextN][n] * layers[l + 1].error[nextN];

        layers[l].error[n] = sum * layers[l].outputDerivative[n];
      }
    }
  }

  void updateWeights(double LR) {
    for(int l = 1; l < netSize; l++) {
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

  int netSize;
  vector <Layer> layers;
};

/// use this to adjust training data size

namespace Training {
  const int inputSize = 10;
  const int outputSize = 1;

  void createDataset(int size, string path) {
    ofstream out (path);

    for(int i = 0; i < size; i++) {
      vector <double> inp, outp;
      int cnt[2], x;
      cnt[0] = cnt[1] = 0;

      for(int j = 0; j < inputSize; j++) {
        x = rand() % 2;
        inp.push_back(x);
        out << x << " ";
        cnt[x]++;
      }
      x = (cnt[1] > cnt[0]);
      outp.push_back(x);
      out << x << " ";
      out << "\n";
    }
  }

  void readDataset(vector <vector <double>> &input, vector <vector <double>> &output, int size, string path) {
    ifstream in (path);
    double x;

    for(int i = 0; i < size; i++) {
      vector <double> inp, outp;
      for(int j = 0; j < inputSize; j++) {
        in >> x;
        inp.push_back(x);
      }

      for(int j = 0; j < outputSize; j++) {
        in >> x;
        outp.push_back(x);
      }

      input.push_back(inp);
      output.push_back(outp);
    }
  }
}

void test() {
  bool create = false;
  int dataSize = 100000;
  int epochs = 1000;
  double LR = 0.1;

  if(create) {
    Training::createDataset(dataSize, "training.txt");
    return;
  }

  vector <int> topology;

  topology.push_back(10);
  topology.push_back(5);
  topology.push_back(1);

  Network NN(topology);

  double split = 0.1;
  int trainSize = dataSize * split;
  vector <vector <double>> input, output;

  Training::readDataset(input, output, dataSize, "training.txt");

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

    if(epoch % 50 == 0)
      LR /= 2;

    cout << "Total error: " << error / (2 * (dataSize - trainSize)) << "\n";
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

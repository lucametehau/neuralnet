#pragma once
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include <random>

using namespace std;

const double SIGMOID_SCALE = 0.005;

const int NONE       = 0;
const int SIGMOID    = 1;
const int RELU       = 2;
const int LEAKY_RELU = 3;

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

class LayerInfo {
public:
  int size;
  int activationType;
};

class Layer {
public:
  Layer(LayerInfo _info, int prevNumNeurons) {
    info = _info;

    int numNeurons = _info.size;

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

  LayerInfo info;
  vector <double> bias, output, error, outputDerivative;
  vector <vector <double>> weights;
};

class Network {
public:

  Network(vector <LayerInfo> &topology) {
    for(int i = 0; i < (int)topology.size(); i++) {
      layers.push_back(Layer(topology[i], (i > 0 ? topology[i - 1].size : 0)));
    }
  }

  double activationFunction(double x, int type) {
    if(type == SIGMOID)
      return 1.0 / (1.0 + exp(-SIGMOID_SCALE * x));

    if(x > 0)
      return x;

    return (type == RELU ? 0 : 0.05 * x);
  }

  double activationFunctionDerivative(double x, int type) {
    if(type == SIGMOID) {
      double value = activationFunction(x, type);
      return value * (1 - value) * SIGMOID_SCALE;
    }

    if(x > 0)
      return 1;

    return (type == RELU ? 0 : 0.05);
  }

  vector <double> calc(vector <double> &input) { /// feed forward
    //cout << layers.size() << " " << layers[0].size << " " << input.size() << "\n";
    layers[0].output = input;

    //cout << "a\n";

    for(int l = 1; l < (int)layers.size(); l++) {
      for(int n = 0; n < layers[l].info.size; n++) {
        double sum = layers[l].bias[n];

        for(int prevN = 0; prevN < layers[l - 1].info.size; prevN++) {
          sum += layers[l - 1].output[prevN] * layers[l].weights[n][prevN];
        }

        //cout << l << " " << n << " " << layers[l].bias[n] << " " << sum << "\n";

        layers[l].output[n] = activationFunction(sum, layers[l].info.activationType);
        layers[l].outputDerivative[n] = activationFunctionDerivative(sum, layers[l].info.activationType);
      }

      //cout << "---------------------------\n";
    }

    //cout << "a\n";

    return layers.back().output;
  }

  void backProp(vector <double> &target) {
    /// for output neurons
    for(int n = 0; n < layers.back().info.size; n++) {
      layers.back().error[n] = (layers.back().output[n] - target[n]) * layers.back().outputDerivative[n];
    }

    /// for hidden layers
    for(int l = (int)layers.size() - 2; l > 0; l--) {
      for(int n = 0; n < layers[l].info.size; n++) {
        double sum = 0;
        for(int nextN = 0; nextN < layers[l + 1].info.size; nextN++)
          sum += layers[l + 1].weights[nextN][n] * layers[l + 1].error[nextN];

        layers[l].error[n] = sum * layers[l].outputDerivative[n];
      }
    }
  }

  void updateWeights(double LR) {
    for(int l = 1; l < (int)layers.size(); l++) {
      for(int n = 0; n < layers[l].info.size; n++) {
        for(int prevN = 0; prevN < layers[l - 1].info.size; prevN++) {
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
    if((int)input.size() != layers[0].info.size) {
      cout << "Wrong input size!\n";
      assert(0);
    }

    if((int)target.size() != layers.back().info.size) {
      cout << "Wrong output size!\n";
      assert(0);
    }

    calc(input);
    backProp(target);
    updateWeights(LR);
  }

  double calcError(vector <vector <double>> &input, vector <vector <double>> &output, int trainSize, int dataSize) {
    double error = 0;

    double tStart = clock();

    for(int i = trainSize; i < dataSize; i++) {
      vector <double> ans = calc(input[i]);

      double delta = (ans[0] - output[i][0]);
      error += delta * delta;

      if(i % 1000 == 0) {
        cout << "index " << i << "\n";
        cout << ans[0] << " " << output[i][0] << "\n\n";
      }
    }

    double tEnd = clock();

    cout << "Raw error : " << error << "\n";

    cout << "Error     : " << error / (2 * (dataSize - trainSize)) << "\n";

    cout << "Time taken: " << (tEnd - tStart) / CLOCKS_PER_SEC << " s\n";

    return error;
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

      for(int j = 0; j < layers[i].info.size && i; j++) {
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
      int sz = layers[i].info.size;
      layers[i].bias = read(sz, in);
      layers[i].output = read(sz, in);
      layers[i].outputDerivative = read(sz, in);
      layers[i].error = read(sz, in);

      for(int j = 0; j < sz && i; j++) {
        layers[i].weights[j] = read(layers[i - 1].info.size, in);
      }
    }
  }

  vector <Layer> layers;
};

/// runs training on given input, output, dataSize, epochs, LR
/// splits data into training and testing according to split
/// saves NN to "savePath"
/// can load NN from "savePath"

void runTraining(vector <LayerInfo> &topology, vector <vector <double>> &input, vector <vector <double>> &output,
                 int dataSize, int batchSize, int epochs, double LR, double split, string savePath, bool load) {

  int trainSize = dataSize * (1.0 - split);
  double minError = 1e10;

  Network NN(topology);

  if(load) {
    NN.load(savePath);

    NN.calcError(input, output, trainSize, dataSize);

    //return;
  }

  /*for(int i = 0; i < 100000; i++) {
    NN.train(input[0], output[0], LR);
    NN.train(input[2], output[2], LR);

    vector <double> ans = NN.calc(input[0]);

    cout << i << " : " << ans[0] << " " << output[0][0] << ", ";

    ans = NN.calc(input[2]);

    cout << ans[0] << " " << output[2][0] << "\n";
  }

  return;*/

  for(int epoch = 1; epoch <= epochs; epoch++) {
    cout << "----------------------------------------- Epoch " << epoch << "/" << epochs << " -----------------------------------------\n";

    double tStart = clock();

    for(int i = 0; i < trainSize; i++) {
      /*NN.calc(input[i]);
      NN.backProp(output[i]);*/
      NN.train(input[i], output[i], LR);

      /*if(i % batchSize == 0 || i == trainSize - 1) {
        NN.updateWeights(LR);
        cout << "Training index " << i << " / " << trainSize << "\n";

        //NN.calcError(input, output, trainSize, dataSize);
      }*/
    }

    double tEnd = clock();

    double error = NN.calcError(input, output, trainSize, dataSize);

    cout << "Time taken for epoch: " << (tEnd - tStart) / CLOCKS_PER_SEC << "s\n";

    if(error < minError) {
      cout << "Network saved!\n";
      minError = error;
      NN.save(savePath);
    }
  }
}

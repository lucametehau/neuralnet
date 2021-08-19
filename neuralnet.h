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
  uniform_real_distribution <float> rng(0, 1);
  uniform_int_distribution <int> bin(0, 1);

  vector <float> createRandomArray(int length) {
    vector <float> v;
    float k = sqrtf(2.0 / length);

    for(int i = 0; i < length; i++)
      v.push_back(rng(gen) * k);

    return v;
  }
}

struct LayerInfo {
  int size;
  int activationType;
};

struct NetInput {
  vector <short> ind;
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
  vector <float> bias, output, error, outputDerivative;
  vector <vector <float>> weights;
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

  vector <float> calc(NetInput &input) { /// feed forward
    for(int i = 0; i < layers[0].info.size; i++)
      layers[0].output[i] = 0;

    for(auto &i : input.ind)
      layers[0].output[i] = 1;
    float sum;

    for(int n = 0; n < layers[1].info.size; n++) {
      sum = layers[1].bias[n];

      /// when feeding forward to first layer
      /// we don't have to go through the input
      /// values that are 0 (which is the output
      /// of the input layer)

      for(auto &prevN : input.ind) {
        sum += layers[0].output[prevN] * layers[1].weights[n][prevN];
      }

      layers[1].output[n] = activationFunction(sum, layers[1].info.activationType);
      layers[1].outputDerivative[n] = activationFunctionDerivative(sum, layers[1].info.activationType);
    }

    for(int l = 2; l < (int)layers.size(); l++) {
      for(int n = 0; n < layers[l].info.size; n++) {
        sum = layers[l].bias[n];

        for(int prevN = 0; prevN < layers[l - 1].info.size; prevN++) {
          sum += layers[l - 1].output[prevN] * layers[l].weights[n][prevN];
        }

        layers[l].output[n] = activationFunction(sum, layers[l].info.activationType);
        layers[l].outputDerivative[n] = activationFunctionDerivative(sum, layers[l].info.activationType);
      }
    }

    return layers.back().output;
  }

  void backProp(vector <float> &target) {
    /// for output neurons
    for(int n = 0; n < layers.back().info.size; n++) {
      layers.back().error[n] = (layers.back().output[n] - target[n]) * layers.back().outputDerivative[n];
    }

    /// for hidden layers
    for(int l = (int)layers.size() - 2; l > 0; l--) {
      for(int n = 0; n < layers[l].info.size; n++) {
        float sum = 0;
        for(int nextN = 0; nextN < layers[l + 1].info.size; nextN++)
          sum += layers[l + 1].weights[nextN][n] * layers[l + 1].error[nextN];

        layers[l].error[n] = sum * layers[l].outputDerivative[n];
      }
    }
  }

  void updateWeights(double LR) {
    for(int l = 1; l < (int)layers.size(); l++) {
      for(int n = 0; n < layers[l].info.size; n++) {
        float delta = -LR * layers[l].error[n];

        for(int prevN = 0; prevN < layers[l - 1].info.size; prevN++) {
          layers[l].weights[n][prevN] += delta * layers[l - 1].output[prevN];
        }

        layers[l].bias[n] += delta;
      }
    }
  }

  void train(NetInput &input, vector <float> &target, double LR) {
    calc(input);
    backProp(target);
    updateWeights(LR);
  }

  double calcError(vector <NetInput> &input, vector <vector <float>> &output, int trainSize, int dataSize) {
    double error = 0;

    double tStart = clock();

    for(int i = trainSize; i < dataSize; i++) {
      vector <float> ans = calc(input[i]);

      double delta = (ans[0] - output[i][0]);
      error += delta * delta;
    }

    double tEnd = clock();

    cout << "Raw error : " << error << "\n";

    cout << "Error     : " << error / (dataSize - trainSize) << "\n";

    cout << "Time taken: " << (tEnd - tStart) / CLOCKS_PER_SEC << " s\n";

    return error;
  }

  void write(vector <float> &v, ofstream &out) {
    for(auto &i : v)
      out << i << " ";
    out << "\n";
  }

  vector <float> read(int lg, ifstream &in) {
    vector <float> v;
    float x;
    for(int i = 0; i < lg; i++)
      in >> x, v.push_back(x);
    return v;
  }

  void save(string path) {
    ofstream out (path);
    int cnt = layers.size();

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

void runTraining(vector <LayerInfo> &topology, vector <NetInput> &input, vector <vector <float>> &output,
                 int dataSize, int batchSize, int epochs, double LR, double split, string savePath, bool load) {

  int trainSize = dataSize * (1.0 - split);
  double minError = 1e10;

  Network NN(topology);

  if(load) {
    NN.load(savePath);

    NN.calcError(input, output, trainSize, dataSize);

    //return;
  }

  for(int epoch = 1; epoch <= epochs; epoch++) {
    cout << "----------------------------------------- Epoch " << epoch << "/" << epochs << " -----------------------------------------\n";

    double tStart = clock();

    for(int i = 0; i < trainSize; i++) {
      NN.calc(input[i]);
      NN.backProp(output[i]);
      //NN.train(input[i], output[i], LR);

      if(i % batchSize == 0 || i == trainSize - 1) {
        NN.updateWeights(LR);
        //cout << "Training index " << i << " / " << trainSize << "\n";

        //NN.calcError(input, output, trainSize, dataSize);
      }
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

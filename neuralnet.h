#pragma once
#include <iomanip>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include <random>

using namespace std;

const double Momentum = 0.3;

const double SIGMOID_SCALE = 0.0075;

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
    double k = sqrtf(2.0 / length);

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

struct FenOutput {
  double result;
  double eval;
};

class Layer {
public:
  Layer(LayerInfo _info, int prevNumNeurons) {
    info = _info;

    int numNeurons = _info.size;

    bias = tools::createRandomArray(numNeurons);
    deltaBias.resize(numNeurons);

    output.resize(numNeurons);
    outputDerivative.resize(numNeurons);
    error.resize(numNeurons);

    weights.resize(numNeurons);
    deltaWeights.resize(numNeurons);

    if(prevNumNeurons) {
      for(int i = 0; i < numNeurons; i++) {
        deltaWeights[i].resize(prevNumNeurons);
        weights[i] = tools::createRandomArray(prevNumNeurons);
      }
    }
  }

  LayerInfo info;
  vector <double> bias, deltaBias;
  vector <double> output, error, outputDerivative;
  vector <vector <double>> weights, deltaWeights;
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

  double calc(NetInput &input) { /// feed forward
    for(int i = 0; i < layers[0].info.size; i++)
      layers[0].output[i] = 0;

    for(auto &i : input.ind)
      layers[0].output[i] = 1;
    double sum;

    for(int n = 0; n < layers[1].info.size; n++) {
      sum = layers[1].bias[n];

      /// when feeding forward to first layer
      /// we don't have to go through the input
      /// values that are 0 (which is the output
      /// of the input layer)

      for(auto &prevN : input.ind) {
        /// layers[0].output[prevN] = 1
        sum += layers[1].weights[n][prevN];
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

    return layers.back().output[0];
  }

  void removeInput(int ind) {
    double sum;

    /// layer 1 uses relu

    layers[0].output[ind] = 0;

    for(int n = 0; n < layers[1].info.size; n++) {
      layers[1].output[n] -= layers[1].weights[n][ind];
    }

    for(int l = 2; l < (int)layers.size(); l++) {
      for(int n = 0; n < layers[l].info.size; n++) {
        sum = layers[l].bias[n];

        for(int prevN = 0; prevN < layers[l - 1].info.size; prevN++) {
          sum += layers[l - 1].output[prevN] * layers[l].weights[n][prevN];
        }

        layers[l].output[n] = activationFunction(sum, layers[l].info.activationType);
      }
    }
  }

  void addInput(int ind) {
    double sum;

    /// layer 1 uses relu

    layers[0].output[ind] = 1;

    for(int n = 0; n < layers[1].info.size; n++) {
      layers[1].output[n] += layers[1].weights[n][ind];
    }

    for(int l = 2; l < (int)layers.size(); l++) {
      for(int n = 0; n < layers[l].info.size; n++) {
        sum = layers[l].bias[n];

        for(int prevN = 0; prevN < layers[l - 1].info.size; prevN++) {
          sum += layers[l - 1].output[prevN] * layers[l].weights[n][prevN];
        }

        layers[l].output[n] = activationFunction(sum, layers[l].info.activationType);
      }
    }
  }

  void checkRemoval() {
    NetInput input;
    input.ind.push_back(1);
    double ans = calc(input);
    input.ind.push_back(2);
    calc(input);
    removeInput(2);
    input.ind.pop_back();
    double ans2 = calc(input);
    cout << ans2 << " " << ans << "\n";

    assert(abs(ans2 - ans) < 1e-9);
  }

  void backProp(double &target) {
    /// for output neurons
    layers.back().error[0] += (layers.back().output[0] - target) * layers.back().outputDerivative[0];

    //cout << layers.back().error[0] << "\n";

    /// for hidden layers
    for(int l = (int)layers.size() - 2; l > 0; l--) {
      for(int n = 0; n < layers[l].info.size; n++) {
        double sum = 0;
        for(int nextN = 0; nextN < layers[l + 1].info.size; nextN++)
          sum += layers[l + 1].weights[nextN][n] * layers[l + 1].error[nextN];

        layers[l].error[n] += sum * layers[l].outputDerivative[n];

        //cout << layers[l].error[n] << " ";
      }
    }
    //cout << "\n";
  }

  void updateWeights(NetInput &input, double LR) {
    for(int n = 0; n < layers[1].info.size; n++) {
      double delta = -LR * layers[1].error[n];

      for(auto &prevN : input.ind) {
        /// layers[0].output[prevN] = 1
        double newDelta = delta + Momentum * layers[1].deltaWeights[n][prevN];
        layers[1].weights[n][prevN] += newDelta;
        layers[1].deltaWeights[n][prevN] = newDelta;
      }

      double newDelta = delta + Momentum * layers[1].deltaBias[n];
      layers[1].bias[n] += newDelta;
      layers[1].deltaBias[n] = newDelta;

      layers[1].error[n] = 0;
    }

    for(int l = 2; l < (int)layers.size(); l++) {
      for(int n = 0; n < layers[l].info.size; n++) {
        double delta = -LR * layers[l].error[n];

        for(int prevN = 0; prevN < layers[l - 1].info.size; prevN++) {
          double newDelta = delta * layers[l - 1].output[prevN] + Momentum * layers[l].deltaWeights[n][prevN];
          layers[l].weights[n][prevN] += newDelta;
          layers[l].deltaWeights[n][prevN] = newDelta;
        }

        double newDelta = delta + Momentum * layers[l].deltaBias[n];
        layers[l].bias[n] += newDelta;
        layers[l].deltaBias[n] = newDelta;

        layers[l].error[n] = 0;
      }
    }
  }

  void train(NetInput &input, double &target, double LR) {
    calc(input);
    backProp(target);
    updateWeights(input, LR);
  }

  double calcError(vector <NetInput> &input, vector <double> &output, int l, int r) {
    double error = 0;

    for(int i = l; i < r; i++) {
      double ans = calc(input[i]);

      double delta = (ans - output[i]);

      //if(i % 10 == 0)
      //cout << i << " : " << ans[0] << " " << output[i] << "\n";

      error += delta * delta;
    }

    return error / (r - l);
  }

  void gradCheck(NetInput &input, double &target) {
    for(int l = 1; l < (int)layers.size(); l++) {
      for(int n = 0; n < layers[l].info.size; n++) {
        for(int prevN = 0; prevN < layers[l - 1].info.size; prevN++) {
          double delta = 0.001;
          double ans = calc(input);

          double E1 = (ans - target) * (ans - target);

          layers[l].weights[n][prevN] += delta;

          //cout << fixed << setprecision(10) << " ans " << ans << "\n";

          ans = calc(input);

          //cout << fixed << setprecision(10) << " ans " << ans << "\n";

          layers[l].weights[n][prevN] -= delta;

          double E2 = (ans - target) * (ans - target), d = (E2 - E1) / (2.0 * delta);
          //cout << E1 << " " << E2 << "\n";
          double grad = layers[l].error[n] * layers[l - 1].output[prevN];
          //cout << l << " " << n << " " << prevN << " " << d << " " << grad << "\n";
          double sum = d + grad, dif = abs(d - grad);

          if(sum && abs(dif / sum) >= 0.01) {
            cout << "Weight problem! " << dif / sum << "\n";
            assert(0);
          }
        }
      }
    }
  }

  void write(vector <double> &v, ofstream &out) {
    for(auto &i : v)
      out << i << " ";
    out << "\n";
  }

  void writeD(vector <double> &v, ofstream &out) {
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

  vector <double> readD(int lg, ifstream &in) {
    vector <double> v;
    double x;
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
      writeD(layers[i].error, out);

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
      layers[i].error = readD(sz, in);

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

void runTraining(vector <LayerInfo> &topology, vector <NetInput> &input, vector <double> &output,
                 int dataSize, int batchSize, int epochs, double LR, double split, string savePath, bool load) {

  int trainSize = dataSize * (1.0 - split);
  double minError = 1e10;

  Network NN(topology);

  NN.checkRemoval();

  if(load) {
    NN.load(savePath);

    double validationError = NN.calcError(input, output, trainSize, dataSize), trainError = NN.calcError(input, output, 0, trainSize);

    cout << "Validation error : " << validationError << " ; Training Error : " << trainError << "\n";
  }

  for(int epoch = 1; epoch <= epochs; epoch++) {
    cout << "----------------------------------------- Epoch " << epoch << "/" << epochs << " -----------------------------------------\n";

    double tStart = clock();

    for(int i = 0; i < trainSize; i++) {
      NN.train(input[i], output[i], LR);

      if(i % batchSize == batchSize - 1 || i == trainSize - 1) {
        //NN.updateWeights(input[i], LR);
        //NN.train(input[i], output[i], LR);
        //cout << "Training index: " << i << " / " << trainSize << "\n";
      }
    }

    double tEnd = clock();

    double validationError = NN.calcError(input, output, trainSize, dataSize), trainError = NN.calcError(input, output, 0, trainSize);

    cout << "Validation error    : " << validationError << " ; Training Error : " << trainError << "\n";
    cout << "Time taken for epoch: " << (tEnd - tStart) / CLOCKS_PER_SEC << "s\n";
    cout << "Learning rate       : " << LR << "\n";

    NN.save(savePath);

    if(validationError > minError) {
      LR *= 0.9;
    } else {
      minError = validationError;
    }
  }
}

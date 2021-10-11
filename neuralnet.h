#pragma once
#include <iomanip>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include <random>
#include <cstring>
#include <cassert>
#include <type_traits>
#include <thread>
#include <mutex>
#include <atomic>

using namespace std;

const double BETA1 = 0.9;
const double BETA2 = 0.999;
const double LR = 0.1;
const double SIGMOID_SCALE = 0.0030;

const int NO_ACTIV = 0;
const int SIGMOID  = 1;
const int RELU     = 2;

string testPos[5] = {
  "3k4/8/8/8/8/8/8/2QK4", ///KQvK
  "3k4/8/8/8/8/8/8/2RK4", ///KRvK
  "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR", /// startpos
  "2r3k1/5pp1/r7/Np5P/1P2pP2/3nP2q/3Q4/R2R2K1", /// king safety
  "8/8/4k3/2B2n2/6K1/5p2/8/5q2" /// something random
};

mutex M;

namespace tools {
  mt19937_64 gen(time(0));
  uniform_real_distribution <double> rng(0, 1);
  uniform_int_distribution <int> bin(0, 1);
  uniform_int_distribution <int> integer(0, (int)1e9);

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
    cout << "char " << c << "\n";
    exit(0);
  }

  return 6 * color + val;
}

NetInput fenToInput(string fen) {
  NetInput ans;
  int ind = 0;

  for(int i = 7; i >= 0; i--) {
    int j = 0;
    while(j < 8 && fen[ind] != '/') {
      if(fen[ind] < '0' || '9' < fen[ind]) {
        int piece = cod(fen[ind]);

        ans.ind.push_back(64 * piece +
                          8 * i + j); /// square
        j++;
      } else {
        j += fen[ind] - '0';
      }
      ind++;
    }
    ind++;
  }

  //sort(ans.ind.begin(), ans.ind.end());

  return ans;
}

class Gradient {
public:
  double m1, m2; /// momentums

  Gradient(double _m1, double _m2) {
    m1 = _m1;
    m2 = _m2;
  }

  Gradient() {
    m1 = m2 = 0;
  }

  double getValue(double grad) {
    if(grad == 0)
      return 0;

    m1 = m1 * BETA1 + grad * (1.0 - BETA1);
    m2 = m2 * BETA2 + (grad * grad) * (1.0 - BETA2);

    return LR * m1 / (sqrt(m2) + 1e-8);
  }
};

class Layer {
public:
  Layer(LayerInfo _info, int prevNumNeurons) {
    info = _info;

    int numNeurons = _info.size;
    vector <double> temp(numNeurons);

    bias = tools::createRandomArray(numNeurons);
    biasGrad.resize(numNeurons);
    biasGradients.resize(numNeurons);

    output.resize(numNeurons);


    if(prevNumNeurons) {
      weights.resize(prevNumNeurons);
      weightsGrad.resize(prevNumNeurons);
      weightsGradients.resize(prevNumNeurons);

      for(int i = 0; i < prevNumNeurons; i++) {
        weights[i] = tools::createRandomArray(numNeurons);
        weightsGrad[i].resize(numNeurons);
        weightsGradients[i].resize(numNeurons);
      }
    }
  }

  LayerInfo info;

  vector <double> bias, biasGradients;
  vector <Gradient> biasGrad;

  vector <double> output;

  vector <vector <double>> weights, weightsGradients;
  vector <vector <Gradient>> weightsGrad;
};

class Network {
public:

  Network(vector <LayerInfo> &topology) {
    for(int i = 0; i < (int)topology.size(); i++) {
      layers.push_back(Layer(topology[i], (i > 0 ? topology[i - 1].size : 0)));
    }
  }

  double activationFunction(double x, int type) {
    if(type == RELU)
      return max(x, 0.0);

    return 1.0 / (1.0 + exp(-SIGMOID_SCALE * x));
  }

  double activationFunctionDerivative(double x, int type) {
    if(type == RELU) {
      return (x > 0);
    }

    //double value = activationFunction(x, type);
    return x * (1 - x) * SIGMOID_SCALE;
  }

  double inverseSigmoid(double val) {
    return log(val / (1.0 - val)) / SIGMOID_SCALE;
  }

  double feedForward(NetInput &input) { /// feed forward
    double sum;

    for(int n = 0; n < layers[1].info.size; n++) {
      layers[1].output[n] = layers[1].bias[n];
    }

    for(auto &prevN : input.ind) {
      for(int n = 0; n < layers[1].info.size; n++)
        layers[1].output[n] += layers[1].weights[prevN][n];
    }

    for(int n = 0; n < layers[1].info.size; n++) {
      layers[1].output[n] = activationFunction(layers[1].output[n], layers[1].info.activationType);
    }

    for(int l = 2; l < (int)layers.size(); l++) {
      for(int n = 0; n < layers[l].info.size; n++) {
        sum = layers[l].bias[n];

        for(int prevN = 0; prevN < layers[l - 1].info.size; prevN++) {
          sum += layers[l - 1].output[prevN] * layers[l].weights[prevN][n];
        }

        layers[l].output[n] = activationFunction(sum, layers[l].info.activationType);
      }
    }

    return layers.back().output[0];
  }

  void backProp(NetInput &input, double &target) { /// back propagate and update gradients
    /// for output neuron
    double outputError = 2 * (layers.back().output[0] - target) *
                         activationFunctionDerivative(layers.back().output[0], layers.back().info.activationType);

    int l = layers.size() - 1;

    /// update gradients

    for(int prevN = 0; prevN < layers[l].info.size; prevN++) {
      layers[l].weightsGradients[prevN][0] += outputError * layers[l - 1].output[prevN];
    }

    layers[l].biasGradients[0] += outputError;

    l = layers.size() - 2;

    /// for hidden layers

    for(int n = 0; n < layers[l].info.size; n++) {
      double error = layers[l + 1].weights[n][0] * outputError *
                     activationFunctionDerivative(layers[l].output[n], layers[l].info.activationType);

      if(error == 0)
        continue;

      /// update gradients

      for(auto &prevN : input.ind) /// assumes l = 1 (only 3 layers)
        layers[l].weightsGradients[prevN][n] += error;

      layers[l].biasGradients[n] += error;
    }
  }

  void updateWeights() { /// update weights
    for(int l = 1; l < (int)layers.size(); l++) {
      for(int prevN = 0; prevN < layers[l - 1].info.size; prevN++) {
        for(int n = 0; n < layers[l].info.size; n++) {
          layers[l].weights[prevN][n] -= layers[l].weightsGrad[prevN][n].getValue(layers[l].weightsGradients[prevN][n]);
          layers[l].weightsGradients[prevN][n] = 0;
        }

      }

      for(int n = 0; n < layers[l].info.size; n++) {
        layers[l].bias[n] -= layers[l].biasGrad[n].getValue(layers[l].biasGradients[n]);
        layers[l].biasGradients[n] = 0;
      }
    }
  }

  void save(string path) {
    FILE *f = fopen(path.c_str(), "wb");
    int cnt = layers.size(), x;

    x = fwrite(&cnt, sizeof(int), 1, f);
    assert(x == 1);

    for(int i = 0; i < (int)layers.size(); i++) {
      int sz = layers[i].info.size;
      x = fwrite(&layers[i].bias[0], sizeof(double), sz, f);
      assert(x == sz);

      x = fwrite(&layers[i].biasGrad[0], sizeof(Gradient), sz, f);
      assert(x == sz);

      for(int j = 0; i && j < layers[i - 1].info.size; j++) {
        x = fwrite(&layers[i].weights[j][0], sizeof(double), sz, f);
        assert(x == sz);

        x = fwrite(&layers[i].weightsGrad[j][0], sizeof(Gradient), sz, f);
        assert(x == sz);
      }
    }

    fclose(f);
  }

  void load(string path) {
    FILE *f = fopen(path.c_str(), "rb");
    int cnt = layers.size(), x;

    x = fread(&cnt, sizeof(int), 1, f);
    assert(x == 1);

    for(int i = 0; i < (int)layers.size(); i++) {
      int sz = layers[i].info.size;
      x = fread(&layers[i].bias[0], sizeof(double), sz, f);
      assert(x == sz);

      x = fread(&layers[i].biasGrad[0], sizeof(Gradient), sz, f);
      assert(x == sz);

      for(int j = 0; i && j < layers[i - 1].info.size; j++) {
        x = fread(&layers[i].weights[j][0], sizeof(double), sz, f);
        assert(x == sz);

        x = fread(&layers[i].weightsGrad[j][0], sizeof(Gradient), sz, f);
        assert(x == sz);
      }
    }

    fclose(f);
  }

  int evaluate(string fen) {
    NetInput input = fenToInput(fen);
    double ans = feedForward(input);
    cout << "Fen: " << fen << " ; eval = " << inverseSigmoid(ans) << "\n";

    return int(inverseSigmoid(ans));
  }

  void evalTestPos() {
    for(int i = 0; i < 5; i++) {
      evaluate(testPos[i]);
    }
  }

  vector <Layer> layers;
};

void trainOnMiniBatch(Network &NN, vector <NetInput> &input, vector <double> &output, int l, int r) {
  for(int i = l; i < r; i++) {
    NN.feedForward(input[i]);
    NN.backProp(input[i], output[i]);
    //NN.updateGradients(input[i]);
  }
}

void trainOnBatch(Network &NN, vector <NetInput> &input, vector <double> &output, int l, int r, int nrThreads) {
  int batchSize = (r - l) / nrThreads + 1; /// split batch in batches for each thread

  vector <thread> threads(nrThreads);
  vector <Network> nets;

  for(int i = 0; i < nrThreads; i++) {
    nets.push_back(NN);
  }

  int ind = l, id = 0;
  for(auto &t : threads) {
    t = thread{ trainOnMiniBatch, ref(nets[id]), ref(input), ref(output), ind, min(r, ind + batchSize) };
    ind += batchSize;
    id++;
  }

  for(auto &t : threads)
    t.join();

  /// sum up gradients

  for(int j = 0; j < nrThreads; j++) {
    for(int l = 1; l < (int)NN.layers.size(); l++) {
      for(int prevN = 0; prevN < NN.layers[l - 1].info.size; prevN++) {
        for(int n = 0; n < NN.layers[l].info.size; n++) {
          NN.layers[l].weightsGradients[prevN][n] += nets[j].layers[l].weightsGradients[prevN][n];
        }
      }

      for(int n = 0; n < NN.layers[l].info.size; n++) {
        NN.layers[l].biasGradients[n] += nets[j].layers[l].biasGradients[n];
      }
    }
  }
}

void calcErrorBatch(Network &NN, atomic <double> &error, vector <NetInput> &input, vector <double> &output, int l, int r) {
  double errorBatch = 0;

  for(int i = l; i < r; i++) {
    double ans = NN.feedForward(input[i]);
    errorBatch += (ans - output[i]) * (ans - output[i]);
  }

  M.lock();

  error = error + errorBatch;

  M.unlock();
}

double calcError(Network &NN, vector <NetInput> &input, vector <double> &output, int l, int r) {
  atomic <double> error {0};
  const int nrThreads = 8;
  int batchSize = (r - l) / nrThreads + 1;
  vector <thread> threads(nrThreads);
  vector <Network> nets;

  for(int i = 0; i < nrThreads; i++)
    nets.push_back(NN);

  int id = 0, ind = l;
  for(auto &t : threads) {
    t = thread{ calcErrorBatch, ref(nets[id]), ref(error), ref(input), ref(output), ind, min(ind + batchSize, r) };
    ind += batchSize;
    id++;
  }

  for(auto &t : threads)
    t.join();

  return error / (r - l);
}

/// runs training on given input, output, dataSize, epochs, LR
/// splits data into training and testing according to split
/// saves NN to "savePath"
/// can load NN from "savePath"

void runTraining(vector <LayerInfo> &topology, vector <NetInput> &input, vector <double> &output,
                 int dataSize, int batchSize, int epochs, int nrThreads, double split, string loadPath, string savePath,
                 bool load, bool shuffle) {

  assert(input.size() == output.size());

  int trainSize = dataSize * (1.0 - split);
  double minError = 1e10;

  Network NN(topology);

  if(shuffle) {
    int nrInputs = input.size();
    cout << nrInputs << " positions\n";

    /// shuffle training data

    for(int i = nrInputs - 1; i >= 0; i--) {
      int nr = tools::integer(tools::gen) % (i + 1);
      swap(input[i], input[nr]);
      swap(output[i], output[nr]);
    }
  }

  if(load) {
    NN.load(loadPath);

    double validationError = calcError(NN, input, output, trainSize, dataSize), trainError = calcError(NN, input, output, 0, trainSize);

    cout << "Validation error : " << validationError << " ; Training Error : " << trainError << "\n";

    NN.evalTestPos();
  }

  for(int epoch = 1; epoch <= epochs; epoch++) {
    cout << "----------------------------------------- Epoch " << epoch << "/" << epochs << " -----------------------------------------\n";

    double tStart = clock();

    for(int i = 0; i < trainSize; i += batchSize) {
      cout << "Batch " << i / batchSize + 1 << "/" << trainSize / batchSize + 1 << "\r";
      trainOnBatch(NN, input, output, i, min(i + batchSize, trainSize), nrThreads);

      NN.updateWeights();
    }

    cout << "\n";

    double tEnd = clock();

    double testStart = clock();
    double validationError = calcError(NN, input, output, trainSize, dataSize), trainError = calcError(NN, input, output, 0, trainSize);
    double testEnd = clock();

    cout << "Validation error    : " << validationError << " ; Training Error : " << trainError << "\n";
    cout << "Time taken for epoch: " << (tEnd - tStart) / CLOCKS_PER_SEC << "s\n";
    cout << "Time taken for error: " << (testEnd - testStart) / CLOCKS_PER_SEC << "s\n";
    //cout << "Learning rate       : " << LR << "\n";

    NN.evalTestPos();

    NN.save(savePath);

    /*if(epoch % 10 == 0)
      LR *= 0.9; /// decay ?*/

    if(trainError > minError) {
      //LR *= 0.9;
    } else {
      minError = trainError;
    }
  }
}

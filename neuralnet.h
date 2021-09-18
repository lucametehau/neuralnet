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

using namespace std;

const double BETA1 = 0.9;
const double BETA2 = 0.999;
const double LR = 0.1;
const double SIGMOID_SCALE = 0.0075;

const int NO_ACTIV   = 0;
const int SIGMOID    = 1;
const int RELU       = 2;

string testPos[4] = {
  "3k4/8/8/8/8/8/8/2QK4 w - - 0 1", ///KQvK
  "3k4/8/8/8/8/8/8/2RK4 w - - 0 1", ///KRvK
  "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", /// startpos
  "2r3k1/5pp1/r7/Np5P/1P2pP2/3nP2q/3Q4/R2R2K1 w - - 0 1", /// king safety
};

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
    cout << c << "\n";
    assert(0);
  }

  return 6 * color + val;
}

NetInput fenToInput(char fen[]) {
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
  double grad;
  double m1, m2; /// momentums

  Gradient(double _m1, double _m2) {
    m1 = _m1;
    m2 = _m2;
    grad = 0;
  }

  Gradient() {
    grad = 0;
    m1 = m2 = 0;
  }

  void add(double _grad) {
    grad += _grad;
  }

  double getValue() {
    if(!grad)
      return 0;

    m1 = m1 * BETA1 + grad * (1.0 - BETA1);
    m2 = m2 * BETA2 + (grad * grad) * (1.0 - BETA2);

    grad = 0;
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

    output.resize(numNeurons);
    error.resize(numNeurons);


    if(prevNumNeurons) {
      weights.resize(prevNumNeurons);
      weightsGrad.resize(prevNumNeurons);

      for(int i = 0; i < prevNumNeurons; i++) {
        weights[i] = tools::createRandomArray(numNeurons);
        weightsGrad[i].resize(numNeurons);
      }
    }
  }

  LayerInfo info;

  vector <double> bias;
  vector <Gradient> biasGrad;

  vector <double> output, error;

  vector <vector <double>> weights;
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

  void backProp(double &target) {
    /// for output neurons
    layers.back().error[0] = (layers.back().output[0] - target) *
                             activationFunctionDerivative(layers.back().output[0], layers.back().info.activationType);

    /// for hidden layers

    int l = layers.size() - 2;

    for(int n = 0; n < layers[l].info.size; n++) {
      layers[l].error[n] = layers[l + 1].weights[n][0] * layers[l + 1].error[0] *
                           activationFunctionDerivative(layers[l].output[n], layers[l].info.activationType);
    }

    for(int l = (int)layers.size() - 3; l > 0; l--) {
      for(int n = 0; n < layers[l].info.size; n++) {
        double sum = 0;
        for(int nextN = 0; nextN < layers[l + 1].info.size; nextN++)
          sum += layers[l + 1].weights[n][nextN] * layers[l + 1].error[nextN];

        layers[l].error[n] = sum * activationFunctionDerivative(layers[l].output[n], layers[l].info.activationType);
      }
    }
  }

  void updateGradients(NetInput &input) {
    for(auto &prevN : input.ind) {
      for(int n = 0; n < layers[1].info.size; n++) {
        layers[1].weightsGrad[prevN][n].add(layers[1].error[n]);
      }
    }

    for(int n = 0; n < layers[1].info.size; n++) {
      layers[1].biasGrad[n].add(layers[1].error[n]);
    }

    for(int l = 2; l < (int)layers.size(); l++) {
      for(int n = 0; n < layers[l].info.size; n++) {
        for(int prevN = 0; prevN < layers[l - 1].info.size; prevN++) {
          layers[l].weightsGrad[prevN][n].add(layers[l].error[n] * layers[l - 1].output[prevN]);
        }

        layers[l].biasGrad[n].add(layers[l].error[n]);
      }
    }
  }

  void updateWeights(NetInput &input) {
    for(auto &prevN : input.ind) {
      for(int n = 0; n < layers[1].info.size; n++) {
        layers[1].weights[prevN][n] -= layers[1].weightsGrad[prevN][n].getValue();
      }
    }

    for(int n = 0; n < layers[1].info.size; n++) {
      layers[1].bias[n] -= layers[1].biasGrad[n].getValue();
    }

    for(int l = 2; l < (int)layers.size(); l++) {
      for(int n = 0; n < layers[l].info.size; n++) {
        for(int prevN = 0; prevN < layers[l - 1].info.size; prevN++) {
          layers[l].weights[prevN][n] -= layers[l].weightsGrad[prevN][n].getValue();
        }

        layers[l].bias[n] -= layers[l].biasGrad[n].getValue();
      }
    }
  }

  double calcError(vector <NetInput> &input, vector <double> &output, int l, int r) {
    double error = 0;

    for(int i = l; i < r; i++) {
      double ans = feedForward(input[i]);

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
          double ans = feedForward(input);

          double E1 = (ans - target) * (ans - target);

          layers[l].weights[prevN][n] += delta;

          //cout << fixed << setprecision(10) << " ans " << ans << "\n";

          ans = feedForward(input);

          //cout << fixed << setprecision(10) << " ans " << ans << "\n";

          layers[l].weights[prevN][n] -= delta;

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

  int evaluate(char s[]) {
    string fen = "";
    int ind = 0;
    char realFen[105];

    while(s[ind] != ' ')
      fen += s[ind++];

    strcpy(realFen, fen.c_str());

    NetInput input = fenToInput(realFen);
    double ans = feedForward(input);
    cout << "Fen: " << s << " ; eval = " << inverseSigmoid(ans) << "\n";

    return int(inverseSigmoid(ans));
  }

  void evalTestPos() {
    for(int i = 0; i < 4; i++) {
      char fen[105];
      strcpy(fen, testPos[i].c_str());

      evaluate(fen);
    }
  }

  vector <Layer> layers;
};

/// runs training on given input, output, dataSize, epochs, LR
/// splits data into training and testing according to split
/// saves NN to "savePath"
/// can load NN from "savePath"

void runTraining(vector <LayerInfo> &topology, vector <NetInput> &input, vector <double> &output,
                 int dataSize, int batchSize, int epochs, double split, string loadPath, string savePath,
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

    /*for(int i = 0; i < 100; i++) {
      cout << "Position #" << i << "/100" << "\n";
      for(auto &j : input[i].ind)
        cout << j << " ";
      cout << "\n" << output[i] << "\n";
    }*/
  }

  if(load) {
    NN.load(loadPath);

    double validationError = NN.calcError(input, output, trainSize, dataSize), trainError = NN.calcError(input, output, 0, trainSize);

    cout << "Validation error : " << validationError << " ; Training Error : " << trainError << "\n";

    NN.evalTestPos();
  }

  for(int epoch = 1; epoch <= epochs; epoch++) {
    cout << "----------------------------------------- Epoch " << epoch << "/" << epochs << " -----------------------------------------\n";

    double tStart = clock();

    for(int i = 0; i < trainSize; i++) {
      NN.feedForward(input[i]);
      NN.backProp(output[i]);
      NN.updateGradients(input[i]);

      if(i % batchSize == batchSize - 1 || i == trainSize - 1) {
        NN.updateWeights(input[i]);
      }
    }

    double tEnd = clock();

    double testStart = clock();
    double validationError = NN.calcError(input, output, trainSize, dataSize), trainError = NN.calcError(input, output, 0, trainSize);
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

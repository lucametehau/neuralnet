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
#include <immintrin.h>
#include <inttypes.h>
#include <omp.h>

using namespace std;

const int INPUT_NEURONS = 768;
const int HIDDEN_NEURONS = 512;

const float BETA1 = 0.9;
const float BETA2 = 0.999;
const float SIGMOID_SCALE = 0.00447111749925;
const float LAMBDA = 0.0001;
float LR = 0.1;

const int NO_ACTIV = 0;
const int SIGMOID = 1;
const int RELU = 2;

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
    uniform_int_distribution <int> bin(0, 1);
    uniform_int_distribution <int> integer(0, (int)1e9);
}

struct NetInput {
    uint64_t pieces[2];
    uint64_t occ;

    NetInput() {
        pieces[0] = pieces[1] = occ = 0;
    }

    void setPiece(int ind, int sq, int p) {
        pieces[ind] = (pieces[ind] << 4) | p;
        occ |= (1ULL << sq);
    }
};

struct GoodNetInput {
    uint8_t nr;
    uint16_t v[32];
};

struct OutputValues {
    float output[HIDDEN_NEURONS] __attribute__((aligned(64)));
} __attribute__((aligned(64)));

int cod(char c) {
    bool color = 0;
    if ('A' <= c && c <= 'Z') {
        color = 1;
        c += 32;
    }

    int val = 0;

    switch (c) {
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

    val++;

    return 6 * color + val;
}

NetInput fenToInput(string &fen) {
    NetInput ans;
    int ind = 0, nr = 1;
    uint16_t v[10];
    int m = 0;

    for (int i = 7; i >= 0; i--) {
        int j = 0;

        while (j < 8 && fen[ind] != '/') {
            if (fen[ind] < '0' || '9' < fen[ind]) {
                int piece = cod(fen[ind]);

                v[++m] = ((piece << 6) | (8 * i + j));
                j++;
            }
            else {
                j += fen[ind] - '0';
            }
            ind++;
        }

        for (int j = m; j >= 1; j--) {
            ans.setPiece((nr <= 16), v[j] & 63, v[j] >> 6);
            nr++;
        }
        m = 0;

        ind++;
    }

    return ans;
}

class Gradient {
public:
    float m1, m2; /// momentums

    Gradient(float _m1, float _m2) {
        m1 = _m1;
        m2 = _m2;
    }

    Gradient() {
        m1 = m2 = 0;
    }

    float getValue(float grad) {
        if (grad == 0)
            return 0;

        m1 = m1 * BETA1 + grad * (1.0 - BETA1);
        m2 = m2 * BETA2 + (grad * grad) * (1.0 - BETA2);

        return LR * m1 / (sqrt(m2) + 1e-8);
    }
};



int pieceCode(int piece, int sq) {
    //cout << piece << " " << sq << " " << kingCol << " " << int(kingCol) << "\n";
    return 64 * piece + sq;
}

void setInput(GoodNetInput &input_v, NetInput& input) {
    uint64_t m = input.occ;
    int nr = 1, val = max(0, __builtin_popcountll(input.occ) - 16);
    uint64_t temp[2] = { input.pieces[0], input.pieces[1] };
    input_v.nr = 0;

    while (m) {
        uint64_t lsb = m & -m;
        int sq = __builtin_ctzll(lsb);
        int bucket = (nr > val);
        int p = (temp[bucket] & 15) - 1;


        input_v.v[input_v.nr++] = pieceCode(p, sq);
        temp[bucket] >>= 4;

        nr++;
        m ^= lsb;
    }
}

class Network {
public:

    Network() {
        for (int i = 0; i < INPUT_NEURONS; i++) {
            float k = sqrtf(2.0 / INPUT_NEURONS);
            normal_distribution <float> rng(0, k);

            for (int j = 0; j < HIDDEN_NEURONS; j++) {
                inputWeights[i][j] = rng(tools::gen);
            }
        }

        float k = sqrtf(2.0 / HIDDEN_NEURONS);
        normal_distribution <float> rng(0, k);

        for (int i = 0; i < HIDDEN_NEURONS; i++)
            outputWeights[i] = rng(tools::gen);

        memset(inputWeightsGradients, 0, sizeof(inputWeightsGradients));
        memset(outputWeightsGradients, 0, sizeof(outputWeightsGradients));
        memset(inputBiases, 0, sizeof(inputBiases));
        outputBias = 0;
    }

    float activationFunction(float x, int type) {
        if (type == RELU)
            return max(x, 0.0f);

        return 1.0f / (1.0f + exp(-SIGMOID_SCALE * x));
    }

    float activationFunctionDerivative(float x, int type) {
        if (type == RELU) {
            return (x > 0);
        }

        //float value = activationFunction(x, type);
        return x * (1 - x) * SIGMOID_SCALE;
    }

    float inverseSigmoid(float val) {
        return log(val / (1 - val)) / SIGMOID_SCALE;
    }

    float feedForward(GoodNetInput &input_v, OutputValues &o) { /// feed forward

        memcpy(o.output, inputBiases, sizeof(o.output));
        __m256* v = (__m256*)o.output;

        for (int i = 0; i < input_v.nr; i++) {
            __m256* z = (__m256*)inputWeights[input_v.v[i]];

            for (int j = 0; j < batches; j++)
                v[j] = _mm256_add_ps(v[j], z[j]);
        }

        float sum = outputBias;

        __m256* w = (__m256*)outputWeights;

        for (int i = 0; i < batches; i++) {
            v[i] = _mm256_max_ps(zero, v[i]);

            __m256 tmp = _mm256_mul_ps(v[i], w[i]);
            float tempRes[8] __attribute__((aligned(16)));
            _mm256_store_ps(tempRes, tmp);

            sum += tempRes[0] + tempRes[1] + tempRes[2] + tempRes[3] + tempRes[4] + tempRes[5] + tempRes[6] + tempRes[7];
        }

        return activationFunction(sum, SIGMOID);
    }

    float feedForward(NetInput &input) {
        setInput(input_v, input);

        memcpy(output, inputBiases, sizeof(output));
        __m256* v = (__m256*)output;

        for (int i = 0; i < input_v.nr; i++) {
            __m256* z = (__m256*)inputWeights[input_v.v[i]];

            for (int j = 0; j < batches; j++)
                v[j] = _mm256_add_ps(v[j], z[j]);
        }

        float sum = outputBias;

        __m256* w = (__m256*)outputWeights;

        for (int i = 0; i < batches; i++) {
            v[i] = _mm256_max_ps(zero, v[i]);

            __m256 tmp = _mm256_mul_ps(v[i], w[i]);
            float tempRes[8] __attribute__((aligned(16)));
            _mm256_store_ps(tempRes, tmp);

            sum += tempRes[0] + tempRes[1] + tempRes[2] + tempRes[3] + tempRes[4] + tempRes[5] + tempRes[6] + tempRes[7];
        }

        return activationFunction(sum, SIGMOID);
    }

    void updateWeights() { /// update weights
        const int nrThreads = 8;
#pragma omp parallel for schedule(auto) num_threads(nrThreads)
        for (int prevN = 0; prevN < INPUT_NEURONS; prevN++) {
            for (int n = 0; n < HIDDEN_NEURONS; n++) {
                inputWeights[prevN][n] -= inputWeightsGrad[prevN][n].getValue(inputWeightsGradients[prevN][n]);
                inputWeightsGradients[prevN][n] = 0;
            }
        }

#pragma omp parallel for schedule(auto) num_threads(nrThreads)
        for (int n = 0; n < HIDDEN_NEURONS; n++) {
            inputBiases[n] -= inputBiasesGrad[n].getValue(inputBiasesGradients[n]);
            inputBiasesGradients[n] = 0;
        }

#pragma omp parallel for schedule(auto) num_threads(nrThreads)
        for (int n = 0; n < HIDDEN_NEURONS; n++) {
            outputWeights[n] -= outputWeightsGrad[n].getValue(outputWeightsGradients[n]);
            outputWeightsGradients[n] = 0;
        }
        
        outputBias -= outputBiasGrad.getValue(outputBiasGradient);
        outputBiasGradient = 0;
    }

    void save(string path) {
        FILE* f = fopen(path.c_str(), "wb");
        int cnt = 3, x;

        x = fwrite(&cnt, sizeof(int), 1, f);
        assert(x == 1);

        int sz = HIDDEN_NEURONS;

        x = fwrite(inputBiases, sizeof(float), sz, f);
        assert(x == sz);

        x = fwrite(inputBiasesGrad, sizeof(Gradient), sz, f);
        assert(x == sz);

        for (int j = 0; j < INPUT_NEURONS; j++) {
            x = fwrite(inputWeights[j], sizeof(float), sz, f);
            assert(x == sz);

            x = fwrite(inputWeightsGrad[j], sizeof(Gradient), sz, f);
            assert(x == sz);
        }

        sz = 1;

        x = fwrite(&outputBias, sizeof(float), sz, f);
        assert(x == sz);

        x = fwrite(&outputBiasGrad, sizeof(Gradient), sz, f);
        assert(x == sz);

        x = fwrite(outputWeights, sizeof(float), HIDDEN_NEURONS, f);
        assert(x == HIDDEN_NEURONS);

        x = fwrite(outputWeightsGrad, sizeof(Gradient), HIDDEN_NEURONS, f);
        assert(x == HIDDEN_NEURONS);

        fclose(f);
    }

    void load(string path) {
        FILE* f = fopen(path.c_str(), "rb");
        int cnt = 3, x;

        x = fwrite(&cnt, sizeof(int), 1, f);
        assert(x == 1);

        int sz = HIDDEN_NEURONS;

        x = fread(inputBiases, sizeof(float), sz, f);
        assert(x == sz);

        x = fread(inputBiasesGrad, sizeof(Gradient), sz, f);
        assert(x == sz);

        for (int j = 0; j < INPUT_NEURONS; j++) {
            x = fread(inputWeights[j], sizeof(float), sz, f);
            assert(x == sz);

            x = fread(inputWeightsGrad[j], sizeof(Gradient), sz, f);
            assert(x == sz);
        }

        sz = 1;

        x = fread(&outputBias, sizeof(float), sz, f);
        assert(x == sz);

        x = fread(&outputBiasGrad, sizeof(Gradient), sz, f);
        assert(x == sz);

        x = fread(outputWeights, sizeof(float), HIDDEN_NEURONS, f);
        assert(x == HIDDEN_NEURONS);

        x = fread(outputWeightsGrad, sizeof(Gradient), HIDDEN_NEURONS, f);
        assert(x == HIDDEN_NEURONS);

        fclose(f);
    }

    int evaluate(string fen) {
        NetInput input = fenToInput(fen);

        float ans = feedForward(input);
        cout << "Fen: " << fen << " ; eval = " << inverseSigmoid(ans) << "\n";

        return int(inverseSigmoid(ans));
    }

    void evalTestPos() {
        for (int i = 0; i < 5; i++) {
            evaluate(testPos[i]);
        }
    }

    float output[HIDDEN_NEURONS] __attribute__((aligned(64)));
    float inputBiases[HIDDEN_NEURONS], outputBias;
    float inputBiasesGradients[HIDDEN_NEURONS], outputBiasGradient;
    float inputWeights[INPUT_NEURONS][HIDDEN_NEURONS], outputWeights[HIDDEN_NEURONS] __attribute__((aligned(64)));
    float inputWeightsGradients[INPUT_NEURONS][HIDDEN_NEURONS], outputWeightsGradients[HIDDEN_NEURONS] __attribute__((aligned(64)));

    Gradient inputWeightsGrad[INPUT_NEURONS][HIDDEN_NEURONS], outputWeightsGrad[HIDDEN_NEURONS];
    Gradient inputBiasesGrad[HIDDEN_NEURONS], outputBiasGrad;

    GoodNetInput input_v;

    int lg = sizeof(__m256) / sizeof(float);
    int batches = HIDDEN_NEURONS / lg;

    __m256 zero = _mm256_setzero_ps();
};

struct ThreadGradients {
    ThreadGradients() {
        memset(inputWeightsGradients, 0, sizeof(inputWeightsGradients));
        memset(outputWeightsGradients, 0, sizeof(outputWeightsGradients));
        memset(inputBiasesGradients, 0, sizeof(inputBiasesGradients));
        outputBiasGradient = 0;
    }

    float activationFunctionDerivative(float x, int type) {
        if (type == RELU) {
            return (x > 0);
        }

        //float value = activationFunction(x, type);
        return x * (1 - x) * SIGMOID_SCALE;
    }

    void backProp(GoodNetInput &input_v, OutputValues &o, float outputWeights[], float& pred, float& target) { /// back propagate and update gradients
        /// for output neuron
        float outputError = 2 * (pred - target) * activationFunctionDerivative(pred, SIGMOID);

        /// update gradients

        __m256* v = (__m256*)outputWeightsGradients;
        __m256* w = (__m256*)o.output;
        __m256 error = _mm256_set1_ps(outputError);

        for (int i = 0; i < batches; i++)
            v[i] = _mm256_add_ps(v[i], _mm256_mul_ps(error, w[i]));

        outputBiasGradient += outputError;

        /// for hidden layers

        for (int n = 0; n < HIDDEN_NEURONS; n++) {
            float error = outputWeights[n] * outputError * activationFunctionDerivative(o.output[n], RELU);

            if (error == 0)
                continue;

            /// update gradients

            for (int i = 0; i < input_v.nr; i++)
                inputWeightsGradients[input_v.v[i]][n] += error;

            inputBiasesGradients[n] += error;
        }
    }

    int lg = sizeof(__m256) / sizeof(float);
    int batches = HIDDEN_NEURONS / lg;

    float inputBiasesGradients[HIDDEN_NEURONS], outputBiasGradient;
    float inputWeightsGradients[INPUT_NEURONS][HIDDEN_NEURONS], outputWeightsGradients[HIDDEN_NEURONS] __attribute__((aligned(64)));
};

void sumGradients(Network& NN, Network& nn) {
    for (int i = 0; i < INPUT_NEURONS; i++) {
        for (int j = 0; j < HIDDEN_NEURONS; j++)
            NN.inputWeightsGradients[i][j] += nn.inputWeightsGradients[i][j];
    }

    for (int i = 0; i < HIDDEN_NEURONS; i++)
        NN.outputWeightsGradients[i] += nn.outputWeightsGradients[i];

    for (int i = 0; i < HIDDEN_NEURONS; i++)
        NN.inputBiasesGradients[i] += nn.inputBiasesGradients[i];

    NN.outputBiasGradient += nn.outputBiasGradient;
}

void trainOnBatch(Network& NN, vector <NetInput>& input, vector <float>& output, int l, int r, int nrThreads) {
    vector <ThreadGradients> grads(nrThreads);

    vector <OutputValues> o(nrThreads);
    vector <GoodNetInput> input_v(nrThreads);

#pragma omp parallel for schedule(auto) num_threads(nrThreads)
    for (int i = l; i < r; i++) {
        int th = omp_get_thread_num();

        setInput(input_v[th], input[i]);

        float pred = NN.feedForward(input_v[th], o[th]);

        grads[th].backProp(input_v[th], o[th], NN.outputWeights, pred, output[i]);
    }

#pragma omp parallel for schedule(auto) num_threads(nrThreads)
    for (int i = 0; i < INPUT_NEURONS; i++) {
        for (int j = 0; j < HIDDEN_NEURONS; j++) {
            for (int t = 0; t < nrThreads; t++)
                NN.inputWeightsGradients[i][j] += grads[t].inputWeightsGradients[i][j];
        }
    }

#pragma omp parallel for schedule(auto) num_threads(nrThreads)
    for (int i = 0; i < HIDDEN_NEURONS; i++) {
        for (int t = 0; t < nrThreads; t++)
            NN.outputWeightsGradients[i] += grads[t].outputWeightsGradients[i];
    }

#pragma omp parallel for schedule(auto) num_threads(nrThreads)
    for (int i = 0; i < HIDDEN_NEURONS; i++) {
        for (int t = 0; t < nrThreads; t++)
            NN.inputBiasesGradients[i] += grads[t].inputBiasesGradients[i];
    }

    for(int t = 0; t < nrThreads; t++)
        NN.outputBiasGradient += grads[t].outputBiasGradient;
}

float calcError(Network& NN, vector <NetInput>& input, vector <float>& output, int l, int r) {
    float error = 0;
    const int nrThreads = 15;

    vector <Network> nets;

    for (int i = 0; i < nrThreads; i++)
        nets.push_back(NN);

#pragma omp parallel for schedule(auto) num_threads(nrThreads) reduction(+ : error)
    for (int i = l; i < r; i++) {
        int th = omp_get_thread_num();

        float ans = nets[th].feedForward(input[i]);
        error += (ans - output[i]) * (ans - output[i]);
    }

    return error / (r - l);
}

/// runs training on given input, output, dataSize, epochs, LR
/// splits data into training and testing according to split
/// saves NN to "savePath"
/// can load NN from "savePath"

Network NN;

void runTraining(vector <NetInput>& input, vector <float>& output,
    int dataSize, int batchSize, int epochs, int nrThreads, float split, string loadPath, string savePath,
    bool load, bool shuffle) {

    assert(input.size() == output.size());

    int trainSize = dataSize * (1.0 - split);

    /// shuffle training data (to avoid over fitting)

    if (shuffle) {
        int nrInputs = input.size();
        cout << nrInputs << " positions\n";
        for (int i = nrInputs - 1; i >= 0; i--) {
            int nr = tools::integer(tools::gen) % (i + 1);
            swap(input[i], input[nr]);
            swap(output[i], output[nr]);

            if (i % 10000000 == 0)
                cout << i << "\n";
        }
    }

    /// load network

    if (load) {
        NN.load(loadPath);

        float validationError = calcError(NN, input, output, trainSize, dataSize), trainError = calcError(NN, input, output, 0, trainSize);

        cout << "Validation error : " << validationError << " ; Training Error : " << trainError << "\n";

        NN.evalTestPos();
    }

    /// train

    for (int epoch = 1; epoch <= epochs; epoch++) {
        cout << "----------------------------------------- Epoch " << epoch << "/" << epochs << " -----------------------------------------\n";

        float tStart = clock();

        for (int i = 0; i < trainSize; i += batchSize) {
            cout << "Batch " << i / batchSize + 1 << "/" << trainSize / batchSize + 1 << " ; " << 1.0 * i / (clock() - tStart) << "positions/s\r";
            trainOnBatch(NN, input, output, i, min(i + batchSize, trainSize), nrThreads);

            NN.updateWeights();
        }

        cout << "\n";

        float tEnd = clock();

        float testStart = clock();
        float validationError = calcError(NN, input, output, trainSize, dataSize), trainError = calcError(NN, input, output, 0, trainSize);
        float testEnd = clock();

        cout << "Validation error    : " << validationError << " ; Training Error : " << trainError << "\n";
        cout << "Time taken for epoch: " << (tEnd - tStart) / CLOCKS_PER_SEC << "s\n";
        cout << "Time taken for error: " << (testEnd - testStart) / CLOCKS_PER_SEC << "s\n";
        //cout << "Learning rate       : " << LR << "\n";

        NN.evalTestPos();

        NN.save(savePath);

        if (epoch % 20 == 0)
            LR /= 5;
    }
}

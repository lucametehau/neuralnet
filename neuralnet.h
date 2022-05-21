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
#include "data.h"

using namespace std;

const int INPUT_NEURONS = 1536;
const int HIDDEN_NEURONS = 512;

const float BETA1 = 0.9;
const float BETA2 = 0.999;
const float SIGMOID_SCALE = 0.00447111749925f;
const float LR = 0.01f;

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
    uniform_int_distribution <uint64_t> integer;
}

struct OutputValues {
    float output[HIDDEN_NEURONS] __attribute__((aligned(32)));
} __attribute__((aligned(32)));

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

    float getValue(float &grad) {
        if (grad == 0)
            return 0;

        m1 = m1 * BETA1 + grad * (1.0 - BETA1);
        m2 = m2 * BETA2 + grad * grad * (1.0 - BETA2);
        grad = 0;

        return LR * m1 / (sqrt(m2) + 1e-8f);
    }
};

class Network {
public:

    Network() {
        float k = sqrt(2.0 / INPUT_NEURONS);
        normal_distribution <float> rng(0, k);
        for (int i = 0; i < INPUT_NEURONS; i++) {
            for (int j = 0; j < HIDDEN_NEURONS; j++) {
                inputWeights[i * HIDDEN_NEURONS + j] = rng(tools::gen);
            }
        }

        k = sqrt(2.0 / HIDDEN_NEURONS);
        normal_distribution <float> rng2(0, k);

        for (int i = 0; i < HIDDEN_NEURONS; i++)
            outputWeights[i] = rng2(tools::gen);

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

    // the below functions are taken from here and are used to compute the dot product
    // https://stackoverflow.com/questions/6996764/fastest-way-to-do-horizontal-sse-vector-sum-or-other-reduction

    float hsum_ps_sse3(__m128 v) {
        __m128 shuf = _mm_movehdup_ps(v);
        __m128 sums = _mm_add_ps(v, shuf);
        shuf = _mm_movehl_ps(shuf, sums);
        sums = _mm_add_ss(sums, shuf);
        return _mm_cvtss_f32(sums);
    }

    float hsum256_ps_avx(__m256 v) {
        __m128 vlow = _mm256_castps256_ps128(v);
        __m128 vhigh = _mm256_extractf128_ps(v, 1);
        vlow = _mm_add_ps(vlow, vhigh);
        return hsum_ps_sse3(vlow);
    }

    float feedForward(GoodNetInput &input_v, OutputValues &o) { /// feed forward

        memcpy(o.output, inputBiases, sizeof(o.output));
        __m256* v = (__m256*)o.output;

        for (int i = 0; i < input_v.nr; i++) {
            __m256* z = (__m256*)(inputWeights + input_v.v[i] * HIDDEN_NEURONS);

            for (int j = 0; j < batches; j++)
                v[j] = _mm256_add_ps(v[j], z[j]);
        }

        float sum = outputBias;

        __m256* w = (__m256*)outputWeights;
        __m256 acc = zero;

        for (int i = 0; i < batches; i++) {
            v[i] = _mm256_max_ps(zero, v[i]);
            acc = _mm256_add_ps(acc, _mm256_mul_ps(v[i], w[i]));
        }

        sum += hsum256_ps_avx(acc);

        return activationFunction(sum, SIGMOID);
    }

    float feedForward(NetInput &input) {
        setInput(input_v, input);

        memcpy(output, inputBiases, sizeof(output));
        __m256* v = (__m256*)output;

        for (int i = 0; i < input_v.nr; i++) {
            __m256* z = (__m256*)(inputWeights + input_v.v[i] * HIDDEN_NEURONS);

            for (int j = 0; j < batches; j++)
                v[j] = _mm256_add_ps(v[j], z[j]);
        }

        float sum = outputBias;

        __m256* w = (__m256*)outputWeights;
        __m256 acc = _mm256_setzero_ps();

        for (int i = 0; i < batches; i++) {
            v[i] = _mm256_max_ps(zero, v[i]);
            acc = _mm256_add_ps(acc, _mm256_mul_ps(v[i], w[i]));
        }

        sum += hsum256_ps_avx(acc);

        return activationFunction(sum, SIGMOID);
    }

    void updateWeights() { /// update weights
        const int nrThreads = 8;

#pragma omp parallel for schedule(auto) num_threads(8)
        for (int n = 0; n < INPUT_NEURONS * HIDDEN_NEURONS; n++) {
            inputWeights[n] -= inputWeightsGrad[n].getValue(inputWeightsGradients[n]);
        }

#pragma omp parallel for schedule(auto) num_threads(8)
        for (int n = 0; n < HIDDEN_NEURONS; n++) {
            inputBiases[n] -= inputBiasesGrad[n].getValue(inputBiasesGradients[n]);
        }

#pragma omp parallel for schedule(auto) num_threads(8)
        for (int n = 0; n < HIDDEN_NEURONS; n++) {
            outputWeights[n] -= outputWeightsGrad[n].getValue(outputWeightsGradients[n]);
        }
        
        outputBias -= outputBiasGrad.getValue(outputBiasGradient);
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
        
        sz = INPUT_NEURONS * HIDDEN_NEURONS;
        x = fwrite(inputWeights, sizeof(float), sz, f);
        assert(x == sz);

        x = fwrite(inputWeightsGrad, sizeof(Gradient), sz, f);
        assert(x == sz);

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

        sz = INPUT_NEURONS * HIDDEN_NEURONS;
        x = fread(inputWeights, sizeof(float), sz, f);
        assert(x == sz);

        x = fread(inputWeightsGrad, sizeof(Gradient), sz, f);
        assert(x == sz);

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

    float output[HIDDEN_NEURONS] __attribute__((aligned(32)));
    float inputBiases[HIDDEN_NEURONS], outputBias;
    float inputBiasesGradients[HIDDEN_NEURONS], outputBiasGradient;
    float inputWeights[INPUT_NEURONS * HIDDEN_NEURONS] __attribute__((aligned(32)));
    float outputWeights[HIDDEN_NEURONS] __attribute__((aligned(32)));
    float inputWeightsGradients[INPUT_NEURONS * HIDDEN_NEURONS] __attribute__((aligned(32)));
    float outputWeightsGradients[HIDDEN_NEURONS] __attribute__((aligned(32)));

    Gradient inputWeightsGrad[INPUT_NEURONS * HIDDEN_NEURONS], outputWeightsGrad[HIDDEN_NEURONS];
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

    void backProp(GoodNetInput& input_v, OutputValues& o, float outputWeights[], float& pred, float& target) { /// back propagate and update gradients
        /// for output neuron
        float outputError = 2 * (pred - target) * activationFunctionDerivative(pred, SIGMOID);

        /// update gradients

        __m256* v = (__m256*)outputWeightsGradients;
        __m256* w = (__m256*)o.output;
        __m256 errorval = _mm256_set1_ps(outputError);
        __m256* errorv = (__m256*)error;
        __m256* inputBiasesv = (__m256*)inputBiasesGradients;
        __m256* outputWeightsv = (__m256*)outputWeights;

        for (int i = 0; i < batches; i++) {
            v[i] = _mm256_add_ps(v[i], _mm256_mul_ps(w[i], errorval));

            // for hidden layers

            errorv[i] = _mm256_mul_ps(errorval, _mm256_mul_ps(outputWeightsv[i], _mm256_max_ps(w[i], zero)));
            inputBiasesv[i] = _mm256_add_ps(inputBiasesv[i], errorv[i]);
        }

        outputBiasGradient += outputError;


        for (int i = 0; i < input_v.nr; i++) {
            __m256* z = (__m256*)(inputWeightsGradients + input_v.v[i] * HIDDEN_NEURONS);

            for (int j = 0; j < batches; j++)
                z[j] = _mm256_add_ps(z[j], errorv[j]);
        }
    }


    int lg = sizeof(__m256) / sizeof(float);
    int batches = HIDDEN_NEURONS / lg;

    float error[HIDDEN_NEURONS] __attribute__((aligned(32)));
    float inputBiasesGradients[HIDDEN_NEURONS] __attribute__((aligned(32)));
    float outputBiasGradient;
    float inputWeightsGradients[INPUT_NEURONS * HIDDEN_NEURONS] __attribute__((aligned(32)));
    float outputWeightsGradients[HIDDEN_NEURONS] __attribute__((aligned(32)));

    __m256 zero = _mm256_setzero_ps();
};

void trainOnBatch(Network& NN, Dataset &dataset, int l, int r, int nrThreads) {
    vector <ThreadGradients> grads(nrThreads);

    vector <OutputValues> o(nrThreads);
    vector <GoodNetInput> input_v(nrThreads);

#pragma omp parallel for schedule(auto) num_threads(nrThreads)
    for (int i = l; i < r; i++) {
        int th = omp_get_thread_num();

        setInput(input_v[th], dataset.input[i]);

        float pred = NN.feedForward(input_v[th], o[th]);

        grads[th].backProp(input_v[th], o[th], NN.outputWeights, pred, dataset.output[i]);
    }

#pragma omp parallel for schedule(auto) num_threads(nrThreads)
    for (int i = 0; i < INPUT_NEURONS * HIDDEN_NEURONS; i++) {
        for (int t = 0; t < nrThreads; t++) {
            NN.inputWeightsGradients[i] += grads[t].inputWeightsGradients[i];
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

float calcError(Network& NN, Dataset &dataset, int l, int r) {
    float error = 0;
    const int nrThreads = 15;

    vector <Network> nets;

    for (int i = 0; i < nrThreads; i++)
        nets.push_back(NN);

#pragma omp parallel for schedule(auto) num_threads(nrThreads) reduction(+ : error)
    for (int i = l; i < r; i++) {
        int th = omp_get_thread_num();

        float ans = nets[th].feedForward(dataset.input[i]);
        error += (ans - dataset.output[i]) * (ans - dataset.output[i]);
    }

    return error / (r - l);
}

/// runs training on given input, output, dataSize, epochs, LR
/// splits data into training and testing according to split
/// saves NN to "savePath"
/// can load NN from "savePath"

Network NN;

void runTraining(Dataset &dataset, int dataSize, int batchSize, int epochs, int nrThreads, float split, string loadPath, string savePath, bool load, bool shuffle) {
    int trainSize = dataSize * (1.0 - split);

    /// shuffle training data (to avoid over fitting)

    if (shuffle) {
        cout << dataSize << " positions\n";
        static const int K = (int)1e7;
        for (int i = 0; i < dataSize; i++) {
            uniform_int_distribution <int> poz(max(0, i - K), min(dataSize - 1, i + K));
            int nr = poz(tools::gen);
            swap(dataset.input[i], dataset.input[nr]);
            swap(dataset.output[i], dataset.output[nr]);

            if (i % 10000000 == 0)
                cout << i << "\n";
        }
    }

    /// load network

    if (load) {
        NN.load(loadPath);

        float validationError = calcError(NN, dataset, trainSize, dataSize), trainError = calcError(NN, dataset, 0, trainSize);

        cout << "Validation error : " << validationError << " ; Training Error : " << trainError << "\n";

        NN.evalTestPos();
    }

    /// train

    for (int epoch = 1; epoch <= epochs; epoch++) {
        cout << "----------------------------------------- Epoch " << epoch << "/" << epochs << " -----------------------------------------\n";

        float tStart = clock();

        for (int i = 0; i < trainSize; i += batchSize) {
            //float t1 = clock();
            cout << "Batch " << i / batchSize + 1 << "/" << trainSize / batchSize + 1 << " ; " << 1.0 * i / (clock() - tStart) << "positions/s\r";
            trainOnBatch(NN, dataset, i, min(i + batchSize, trainSize), nrThreads);

            //float t2 = clock();

            NN.updateWeights();

            //float t3 = clock();

            //cout << (t2 - t1) / CLOCKS_PER_SEC << " for training only " << (t3 - t1) / CLOCKS_PER_SEC << " for whole batch\n";
        }

        cout << "\n";

        float tEnd = clock();

        float testStart = clock();
        float validationError = calcError(NN, dataset, trainSize, dataSize), trainError = calcError(NN, dataset, 0, trainSize);
        float testEnd = clock();

        cout << "Validation error    : " << validationError << " ; Training Error : " << trainError << "\n";
        cout << "Time taken for epoch: " << (tEnd - tStart) / CLOCKS_PER_SEC << "s\n";
        cout << "Time taken for error: " << (testEnd - testStart) / CLOCKS_PER_SEC << "s\n";
        //cout << "Learning rate       : " << LR << "\n";

        NN.evalTestPos();

        NN.save(savePath);

        /*if (epoch % 1000 == 0)
            LR /= 5;*/
    }
}

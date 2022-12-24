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
const int SIDE_NEURONS = 512;
const int HIDDEN_NEURONS = 2 * SIDE_NEURONS;

const float BETA1 = 0.9;
const float BETA2 = 0.999;
const float SIGMOID_SCALE = 0.00517421370f;
float LR = 0.01f;

const int NO_ACTIV = 0;
const int SIGMOID = 1;
const int RELU = 2;

const float GAME_RES = 0.5;
const float EVAL = 1.0 - GAME_RES;

string testPos[7] = {
    "3k4/8/8/8/8/8/8/2QK4 w", ///KQvK
    "3k4/8/8/8/8/8/8/2RK4 w", ///KRvK
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w", /// startpos
    "2r3k1/5pp1/r7/Np5P/1P2pP2/3nP2q/3Q4/R2R2K1 w", /// king safety
    "8/8/4k3/2B2n2/6K1/5p2/8/5q2 w", /// something random
    "r1bqkbnr/pppppppp/2n5/8/3P4/8/PPPKPPPP/RNBQ1BNR b kq - 2 2", /// weird position
    "2k5/8/1P6/2KN3P/8/8/8/8 w - - 1 71" /// again
};

mutex M;

namespace tools {
    mt19937_64 gen(time(0));
    uniform_int_distribution <int> bin(0, 1);
    uniform_int_distribution <uint64_t> integer;
};

struct OutputValues {
    float outputstm[SIDE_NEURONS] __attribute__((aligned(32)));
    float outputopstm[SIDE_NEURONS] __attribute__((aligned(32)));
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
            for (int j = 0; j < SIDE_NEURONS; j++) {
                inputWeights[i * SIDE_NEURONS + j] = rng(tools::gen);
            }
        }

        k = sqrt(2.0 / HIDDEN_NEURONS);
        normal_distribution <float> rng2(0, k);

        for (int i = 0; i < HIDDEN_NEURONS; i++)
            outputWeights[i] = rng2(tools::gen);

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

    void add_vectors(__m256*& a, __m256*& b) {
        for (int j = 0; j < batches; j += 4) {
            a[j] = _mm256_add_ps(a[j], b[j]);
            a[j + 1] = _mm256_add_ps(a[j + 1], b[j + 1]);
            a[j + 2] = _mm256_add_ps(a[j + 2], b[j + 2]);
            a[j + 3] = _mm256_add_ps(a[j + 3], b[j + 3]);
        }
    }

    void relu_vector(__m256*& a) {
        for (int j = 0; j < batches; j += 4) {
            a[j] = _mm256_max_ps(zero, a[j]);
            a[j + 1] = _mm256_max_ps(zero, a[j + 1]);
            a[j + 2] = _mm256_max_ps(zero, a[j + 2]);
            a[j + 3] = _mm256_max_ps(zero, a[j + 3]);
        }
    }

    float feedForward(GoodNetInput &input_v, OutputValues &o) { /// feed forward
        memcpy(o.outputstm, inputBiases, sizeof(inputBiases));
        memcpy(o.outputopstm, inputBiases, sizeof(inputBiases));
        __m256* vstm = (__m256*)o.outputstm;
        __m256* vopstm = (__m256*)o.outputopstm;

        bool stm = input_v.stm;

        for (int i = 0; i < input_v.nr; i++) {
            int n0 = input_v.v[stm][i] * SIDE_NEURONS;
            __m256* w2 = (__m256*) & inputWeights[n0];
            add_vectors(vstm, w2);

            n0 = input_v.v[stm ^ 1][i] * SIDE_NEURONS;
            __m256* w1 = (__m256*) & inputWeights[n0];
            add_vectors(vopstm, w1);
        }

        float sum = outputBias;
        __m256* w = (__m256*)outputWeights;
        __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps(), acc3 = _mm256_setzero_ps();

        relu_vector(vstm);
        relu_vector(vopstm);

        for (int i = 0; i < batches; i += 4) {
            acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(vstm[i], w[i]));
            acc1 = _mm256_add_ps(acc1, _mm256_mul_ps(vstm[i + 1], w[i + 1]));
            acc2 = _mm256_add_ps(acc2, _mm256_mul_ps(vstm[i + 2], w[i + 2]));
            acc3 = _mm256_add_ps(acc3, _mm256_mul_ps(vstm[i + 3], w[i + 3]));
        }

        for (int i = 0; i < batches; i += 4) {
            acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(vopstm[i], w[i + batches]));
            acc1 = _mm256_add_ps(acc1, _mm256_mul_ps(vopstm[i + 1], w[i + 1 + batches]));
            acc2 = _mm256_add_ps(acc2, _mm256_mul_ps(vopstm[i + 2], w[i + 2 + batches]));
            acc3 = _mm256_add_ps(acc3, _mm256_mul_ps(vopstm[i + 3], w[i + 3 + batches]));
        }

        acc0 = _mm256_add_ps(acc0, acc1);
        acc2 = _mm256_add_ps(acc2, acc3);
        acc0 = _mm256_add_ps(acc0, acc2);

        sum += hsum256_ps_avx(acc0);

        return activationFunction(sum, SIGMOID);
    }

    float feedForward(NetInput &input) {
        setInput(input_v, input);

        memcpy(outputstm, inputBiases, sizeof(outputstm));
        memcpy(outputopstm, inputBiases, sizeof(outputopstm));

        __m256* vstm = (__m256*)outputstm;
        __m256* vopstm = (__m256*)outputopstm;
        bool stm = input_v.stm;

        for (int i = 0; i < input_v.nr; i++) {
            int n0 = input_v.v[stm][i] * SIDE_NEURONS;
            __m256* w2 = (__m256*) & inputWeights[n0];
            add_vectors(vstm, w2);

            n0 = input_v.v[stm ^ 1][i] * SIDE_NEURONS;
            __m256* w1 = (__m256*) & inputWeights[n0];
            add_vectors(vopstm, w1);
        }

        float sum = outputBias;

        __m256* w = (__m256*)outputWeights;
        __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps(), acc3 = _mm256_setzero_ps();

        relu_vector(vstm);
        relu_vector(vopstm);

        for (int i = 0; i < batches; i += 4) {
            acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(vstm[i], w[i]));
            acc1 = _mm256_add_ps(acc1, _mm256_mul_ps(vstm[i + 1], w[i + 1]));
            acc2 = _mm256_add_ps(acc2, _mm256_mul_ps(vstm[i + 2], w[i + 2]));
            acc3 = _mm256_add_ps(acc3, _mm256_mul_ps(vstm[i + 3], w[i + 3]));
        }

        for (int i = 0; i < batches; i += 4) {
            acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(vopstm[i], w[i + batches]));
            acc1 = _mm256_add_ps(acc1, _mm256_mul_ps(vopstm[i + 1], w[i + 1 + batches]));
            acc2 = _mm256_add_ps(acc2, _mm256_mul_ps(vopstm[i + 2], w[i + 2 + batches]));
            acc3 = _mm256_add_ps(acc3, _mm256_mul_ps(vopstm[i + 3], w[i + 3 + batches]));
        }

        acc0 = _mm256_add_ps(acc0, acc1);
        acc2 = _mm256_add_ps(acc2, acc3);
        acc0 = _mm256_add_ps(acc0, acc2);

        sum += hsum256_ps_avx(acc0);

        return activationFunction(sum, SIGMOID);
    }

    void save(string path) {
        FILE* f = fopen(path.c_str(), "wb");
        int cnt = 3, x;

        x = fwrite(&cnt, sizeof(int), 1, f);
        assert(x == 1);

        int sz = SIDE_NEURONS;

        x = fwrite(inputBiases, sizeof(float), sz, f);
        assert(x == sz);

        x = fwrite(inputBiasesGrad, sizeof(Gradient), sz, f);
        assert(x == sz);
        
        sz = INPUT_NEURONS * SIDE_NEURONS;
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

        x = fread(&cnt, sizeof(int), 1, f);
        assert(x == 1);

        int sz = SIDE_NEURONS;

        x = fread(inputBiases, sizeof(float), sz, f);
        assert(x == sz);

        x = fread(inputBiasesGrad, sizeof(Gradient), sz, f);
        assert(x == sz);

        sz = INPUT_NEURONS * SIDE_NEURONS;
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
        cout << "Fen: " << fen << " ; stm = " << input.stm << " ; eval = " << inverseSigmoid(ans) << "\n";

        return int(inverseSigmoid(ans));
    }

    void evalTestPos() {
        for (int i = 0; i < 7; i++) {
            evaluate(testPos[i]);
        }
    }

    float outputstm[SIDE_NEURONS] __attribute__((aligned(32)));
    float outputopstm[SIDE_NEURONS] __attribute__((aligned(32)));
    float inputBiases[SIDE_NEURONS], outputBias; __attribute__((alligned(32)));
    float inputWeights[INPUT_NEURONS * SIDE_NEURONS] __attribute__((aligned(32)));
    float outputWeights[HIDDEN_NEURONS] __attribute__((aligned(32)));

    Gradient inputWeightsGrad[INPUT_NEURONS * SIDE_NEURONS], outputWeightsGrad[HIDDEN_NEURONS];
    Gradient inputBiasesGrad[SIDE_NEURONS], outputBiasGrad;

    GoodNetInput input_v;

    int lg = sizeof(__m256) / sizeof(float);
    int batches = SIDE_NEURONS / lg;

    __m256 zero = _mm256_setzero_ps();
};

struct ThreadGradients {
    ThreadGradients() {}

    float activationFunctionDerivative(float x, int type) {
        if (type == RELU) {
            return (x > 0);
        }

        //float value = activationFunction(x, type);
        return x * (1 - x) * SIGMOID_SCALE;
    }

    void add_vectors(__m256*& a, __m256*& b) {
        for (int j = 0; j < batches; j += 4) {
            a[j] = _mm256_add_ps(a[j], b[j]);
            a[j + 1] = _mm256_add_ps(a[j + 1], b[j + 1]);
            a[j + 2] = _mm256_add_ps(a[j + 2], b[j + 2]);
            a[j + 3] = _mm256_add_ps(a[j + 3], b[j + 3]);
        }
    }

    void backProp(GoodNetInput& input_v, OutputValues& o, float outputWeights[], float& pred, float& target) { /// back propagate and update gradients
        /// for output neuron
        float outputError = 2 * (pred - target) * activationFunctionDerivative(pred, SIGMOID);
        bool stm = input_v.stm;

        __m256* outputvstm = (__m256*)o.outputstm;
        __m256* outputvopstm = (__m256*)o.outputopstm;
        __m256* inputv = (__m256*)inputBiasesGradients;
        __m256* outputwgv = (__m256*)outputWeightsGradients;
        __m256* outputwv = (__m256*)outputWeights;
        __m256* errorstm = (__m256*)errora;
        __m256* erroropstm = (__m256*)errorb;
        __m256 ct = _mm256_set1_ps(outputError);

        for (int i = 0; i < batches; i++) {
            __m256 val = _mm256_mul_ps(ct, outputvstm[i]);
            outputwgv[i] = _mm256_add_ps(outputwgv[i], val);
            errorstm[i] = _mm256_mul_ps(outputwv[i], val); // max(outputvstm[i], zero) not needed as they are already relu'd
        }

        for (int i = 0; i < batches; i++) {
            __m256 val = _mm256_mul_ps(ct, outputvopstm[i]);
            outputwgv[i + batches] = _mm256_add_ps(outputwgv[i + batches], val);
            erroropstm[i] = _mm256_mul_ps(outputwv[i + batches], val); // same as above
        }

        /*for (int i = 0; i < SIDE_NEURONS; i++)
            inputBiasesGradients[i] += errora[i] + errorb[i];*/

        for (int i = 0; i < batches; i++) {
            inputv[i] = _mm256_add_ps(inputv[i], _mm256_add_ps(errorstm[i], erroropstm[i]));
        }

        outputBiasGradient += outputError;

        for (int i = 0; i < input_v.nr; i++) {
            int n0 = input_v.v[stm][i] * SIDE_NEURONS;
            __m256* v1 = (__m256*) & inputWeightsGradients[n0];
            add_vectors(v1, errorstm);

            n0 = input_v.v[stm ^ 1][i] * SIDE_NEURONS;
            __m256* v2 = (__m256*) & inputWeightsGradients[n0];
            add_vectors(v2, erroropstm);
        }
    }


    int lg = sizeof(__m256) / sizeof(float);
    const int batches = SIDE_NEURONS / lg;

    float errora[SIDE_NEURONS] __attribute__((aligned(32)));
    float errorb[SIDE_NEURONS] __attribute__((aligned(32)));
    //__m256* errorstm = (__m256*)errora;
    //__m256* erroropstm = (__m256*)errorb;
    float inputBiasesGradients[SIDE_NEURONS] __attribute__((aligned(32)));
    float outputBiasGradient;
    float inputWeightsGradients[INPUT_NEURONS * SIDE_NEURONS] __attribute__((aligned(32)));
    float outputWeightsGradients[HIDDEN_NEURONS] __attribute__((aligned(32)));
};

void trainOnBatch(Network& NN, Dataset &dataset, int l, int r, int nrThreads) {
    //nrThreads = omp_get_max_threads();
    vector <ThreadGradients> grads(nrThreads);
         
#pragma omp parallel for schedule(auto) num_threads(nrThreads)
    for (int i = l; i < r; i++) {
        int th = omp_get_thread_num();
        OutputValues output;
        GoodNetInput input_v;

        setInput(input_v, dataset.input[i]);

        float pred = NN.feedForward(input_v, output);

        grads[th].backProp(input_v, output, NN.outputWeights, pred, dataset.output[i]);
    }

#pragma omp parallel for schedule(auto) num_threads(8)
    for (int n = 0; n < INPUT_NEURONS * SIDE_NEURONS; n++) {
        float gradient = 0;
        for (int t = 0; t < nrThreads; t++) {
            gradient += grads[t].inputWeightsGradients[n];
            grads[t].inputWeightsGradients[n] = 0;
        }

        NN.inputWeights[n] -= NN.inputWeightsGrad[n].getValue(gradient);
    }

#pragma omp parallel for schedule(auto) num_threads(8)
    for (int n = 0; n < HIDDEN_NEURONS; n++) {
        float gradient = 0;
        for (int t = 0; t < nrThreads; t++) {
            gradient += grads[t].outputWeightsGradients[n];
            grads[t].outputWeightsGradients[n] = 0;
        }
        NN.outputWeights[n] -= NN.outputWeightsGrad[n].getValue(gradient);
    }

#pragma omp parallel for schedule(auto) num_threads(8)
    for (int n = 0; n < SIDE_NEURONS; n++) {
        float gradient = 0;
        for (int t = 0; t < nrThreads; t++) {
            gradient += grads[t].inputBiasesGradients[n];
            grads[t].inputBiasesGradients[n] = 0;
        }
        NN.inputBiases[n] -= NN.inputBiasesGrad[n].getValue(gradient);
    }

    float gradient = 0;
    for (int t = 0; t < nrThreads; t++) {
        gradient += grads[t].outputBiasGradient;
        grads[t].outputBiasGradient = 0;
    }

    NN.outputBias -= NN.outputBiasGrad.getValue(gradient);
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
        static const int K = (int)1e7; // to avoid many cache misses
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
            cout << "Batch " << i / batchSize + 1 << "/" << trainSize / batchSize + 1 << " ; " 
                << 1.0 * i / (clock() - tStart) * CLOCKS_PER_SEC << "  positions/s ; " 
                << round((i / batchSize) / (clock() - tStart) * CLOCKS_PER_SEC) << "batches/s\r";
            trainOnBatch(NN, dataset, i, min(i + batchSize, trainSize), nrThreads);

            //float t2 = clock();

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

        if (epoch % 10 == 0)
            LR /= 2;
    }
}

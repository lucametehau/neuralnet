#pragma once
#include <string>

using namespace std;

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

struct Dataset {
    int nr;
    float* output;
    NetInput* input;
};

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

int pieceCode(int piece, int sq) {
    //cout << piece << " " << sq << " " << kingCol << " " << int(kingCol) << "\n";
    return 64 * piece + sq;
}


NetInput fenToInput(string& fen) {
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

void setInput(GoodNetInput& input_v, NetInput& input) {
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
#pragma once
#include <string>

using namespace std;

enum {
    BLACK, WHITE
};

struct NetInput {
    bool stm;
    uint8_t kingSq[2];
    uint64_t pieces[2];
    uint64_t occ;

    NetInput() {
        pieces[0] = pieces[1] = occ = 0;
        kingSq[0] = kingSq[1] = 0;
        stm = 0;
    }

    void setPiece(int ind, int sq, int p) {
        pieces[ind] = (pieces[ind] << 4) | p;
        occ |= (1ULL << sq);

        if (p == 6)
            kingSq[BLACK] = sq;
        else if (p == 12)
            kingSq[WHITE] = sq;
    }
};

struct GoodNetInput {
    bool stm;
    uint8_t nr;
    uint16_t v[2][32];
};

struct Dataset {
    int nr;
    float* output;
    NetInput* input;

    Dataset() {
        nr = 0;
    }

    Dataset(int nrPos) {
        nr = 0;
        input = (NetInput*)malloc(sizeof(NetInput) * nrPos);
        output = (float*)malloc(sizeof(float) * nrPos);
    }

    void init(int nrPos) {
        nr = 0;
        input = (NetInput*)malloc(sizeof(NetInput) * nrPos);
        output = (float*)malloc(sizeof(float) * nrPos);
    }
};

int cod(char c) {
    bool color = BLACK;
    if ('A' <= c && c <= 'Z') {
        color = WHITE;
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

int kingInd(int kingSq) {
    return (kingSq & 4) > 0;
}

int pieceCode(int piece, int sq, int kingSq, int side) {
    if (side == BLACK) {
        sq ^= 56;
        kingSq ^= 56;
        piece = (piece >= 6 ? piece - 6 : piece + 6);
    }
    //kingSq = 0;
    //cout << piece << " " << sq << " " << kingCol << " " << int(kingCol) << "\n";
    return 2 * 64 * piece + 64 * kingInd(kingSq) + sq;
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

    //cout << fen << " ";

    if (fen[ind] == 'w')
        ans.stm = WHITE;
    else
        ans.stm = BLACK;
    //cout << ans.stm << "\n";
    return ans;
}

void setInput(GoodNetInput& input_v, NetInput& input) {
    uint64_t m = input.occ;
    int nr = 1, val = max(0, __builtin_popcountll(input.occ) - 16);
    uint64_t temp[2] = { input.pieces[0], input.pieces[1] };
    input_v.nr = 0;

    input_v.stm = input.stm;

    while (m) {
        uint64_t lsb = m & -m;
        int sq = __builtin_ctzll(lsb);
        int bucket = (nr > val);
        int p = (temp[bucket] & 15) - 1;


        input_v.v[WHITE][input_v.nr] = pieceCode(p, sq, input.kingSq[WHITE], WHITE);
        input_v.v[BLACK][input_v.nr] = pieceCode(p, sq, input.kingSq[BLACK], BLACK);
        input_v.nr++;
        temp[bucket] >>= 4;

        nr++;
        m ^= lsb;
    }

    //std::sort(input_v.v[WHITE], input_v.v[WHITE] + input_v.nr);
    //std::sort(input_v.v[BLACK], input_v.v[BLACK] + input_v.nr);
}
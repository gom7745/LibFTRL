/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 */
#ifndef FTRL_H
#define FTRL_H
#include <vector>
#include <memory>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <cstdio>
#include <iomanip>
#include <cstring>
#include <algorithm>

#include "omp.h"

using namespace std;

using FtrlFloat = double;
using FtrlLong = long long;
using FtrlInt = long;

FtrlLong const CHUNK_SIZE = 3000000000;
extern FtrlInt const WIDTH3;
extern FtrlInt const WIDTH4;
extern FtrlInt const WIDTH5;
extern FtrlInt const WIDTH13;
extern FtrlInt const ONE;
extern FtrlInt const TWO;
extern FtrlInt const THREE;
extern FtrlInt const FOUR;
extern FtrlInt const FIVE;
extern FtrlFloat const PONE;
extern FtrlFloat const EPSILON;

class Node {
public:
    FtrlLong idx = 0;
    FtrlFloat  val = 0.0f;
    Node(){};
    ~Node(){};
    Node(FtrlLong idx, FtrlFloat val): idx(idx), val(val){};
};

class Parameter {
public:
    FtrlFloat l1, l2, alpha, beta;
    FtrlInt nrPass, nrThreads;
    bool normalized, verbose, freq, autoStop, noAuc, inMemory;
    Parameter():l1(PONE), l2(PONE), alpha(PONE), beta(ONE), nrPass(ONE), nrThreads(ONE), normalized(false),\
    verbose(true), freq(true), autoStop(false), noAuc(false), inMemory(false){};
    ~Parameter(){};
};

class FtrlChunk {
public:
    FtrlLong l, nnz;
    FtrlInt chunkId;
    string fileName;

    vector<Node> nodes;
    vector<FtrlInt> nnzs;
    vector<FtrlFloat> labels;
    vector<FtrlFloat> R;

    void Read();
    void Write();
    void Clear();

    FtrlChunk(string dataName, FtrlInt chunkId);
    ~FtrlChunk(){};
};

class FtrlData {
public:
    string fileName;
    FtrlLong l, n;
    FtrlInt nrChunk;

    vector<FtrlChunk> chunks;

    explicit FtrlData(string fileName): fileName(fileName), l(0), n(0), nrChunk(0) {};
    ~FtrlData(){};
    void PrFtrlIntDataInfo();
    void SplitChunks();
    void WriteMeta();
private:
    FtrlInt genChunk(ifstream& fs, FtrlChunk& chunk);
};

class FtrlProblem {
public:
    shared_ptr<FtrlData> data;
    shared_ptr<FtrlData> testData;
    shared_ptr<Parameter> param;
    FtrlProblem() {};
    FtrlProblem(shared_ptr<FtrlData> &data, shared_ptr<FtrlData> &testData, shared_ptr<Parameter> &param)
        :data(data), testData(testData), param(param) {};
    ~FtrlProblem(){};


    vector<FtrlFloat> w, z, n, f;
    bool normlization = false;
    FtrlInt t = 0;
    FtrlLong feats = 0;
    FtrlFloat trLoss = 0.0f, vaLoss = 0.0f, vaAuc = 0.0f, funVal = 0.0f, gnorm = 0.0f, reg = 0.0f;
    FtrlFloat startTime = 0.0f;

    void Initialize(bool norm, string warmModelPath);
    void Solve();
    void PrFtrlIntEpochInfo();
    void PrFtrlIntHeaderInfo();
    void SaveModel(string modelPath);
    FtrlLong LoadModel(string modelPath);
    void Fun();
    void Validate();
private:
    FtrlFloat WTx(FtrlChunk& chunk, FtrlInt begin, FtrlInt end, FtrlFloat r, bool doUpdate);
    FtrlFloat g_calAuc(shared_ptr<FtrlData> currentData, vector<FtrlFloat>& vaLabels, vector<FtrlFloat> vaScores,\
            vector<FtrlFloat>& vaOrders);
    void Update(FtrlChunk& chunk, FtrlInt i, vector<FtrlFloat>& grad, FtrlFloat kappa, FtrlFloat r);
    FtrlFloat OneEpoch(shared_ptr<FtrlData> currentData, bool doUpdate, bool doAuc, FtrlFloat& auc,\
            vector<FtrlFloat>& grad);
};
#endif

/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 */
#include "ftrl.h"

FtrlInt const ONE = 1;
FtrlInt const TWO = 2;
FtrlInt const THREE = 3;
FtrlInt const FOUR = 4;
FtrlInt const FIVE = 5;
FtrlInt const WIDTH3 = 3;
FtrlInt const WIDTH4 = 4;
FtrlInt const WIDTH5 = 5;
FtrlInt const WIDTH13 = 13;
FtrlFloat const PONE = 0.1;
FtrlFloat const EPSILON = 1e-8;
FtrlFloat l1 = 0;
FtrlFloat l2 = 0;
FtrlFloat a = 0;
FtrlFloat b = 0;

FtrlChunk::FtrlChunk(string dataName, FtrlInt id): l(0), nnz(0), chunkId(id)
{
    fileName = dataName + ".bin." + to_string(id);
}

struct ChunkMeta {
    FtrlLong l = 0, nnz = 0;
    FtrlInt chunkId = 0;
};

void FtrlChunk::Write()
{
    FILE* fBin = fopen(fileName.c_str(), "wb");
    if (fBin == nullptr) {
        cout << "Error" << endl;
        exit(1);
    }

    ChunkMeta meta;
    meta.l = l;
    meta.nnz = nnz;
    meta.chunkId = chunkId;

    fwrite(reinterpret_cast<char*>(&meta), sizeof(ChunkMeta), 1, fBin);
    fwrite(labels.data(), sizeof(FtrlFloat), l, fBin);
    fwrite(nnzs.data(), sizeof(FtrlInt), l + 1, fBin);
    fwrite(R.data(), sizeof(FtrlFloat), l, fBin);
    fwrite(nodes.data(), sizeof(Node), nnz, fBin);
    fclose(fBin);
}

void FtrlChunk::Read()
{
    FILE* fTr = fopen(fileName.c_str(), "rb");
    if (fTr == nullptr) {
        cout << "Error" << endl;
        exit(1);
    }

    ChunkMeta meta;

    size_t bytes;
    bytes = fread(reinterpret_cast<char*>(&meta), sizeof(ChunkMeta), 1, fTr);
    l = meta.l;
    nnz = meta.nnz;
    chunkId = meta.chunkId;

    labels.resize(l);
    R.resize(l);
    nodes.resize(nnz);
    nnzs.resize(l + 1);

    bytes = fread(labels.data(), sizeof(FtrlFloat), l, fTr);
    bytes = fread(nnzs.data(), sizeof(FtrlInt), l + 1, fTr);
    bytes = fread(R.data(), sizeof(FtrlFloat), l, fTr);
    bytes = fread(nodes.data(), sizeof(Node), nnz, fTr);
    bytes++;

    fclose(fTr);
}

void FtrlChunk::Clear()
{
    labels.clear();
    nodes.clear();
    R.clear();
    nnzs.clear();
}

inline bool Exists(const string& name)
{
    ifstream f(name.c_str());
    return f.good();
}

struct DiskProblemMeta {
    FtrlLong l = 0, n = 0;
    FtrlInt nrChunk = 0;
};

void FtrlData::WriteMeta()
{
    string metaName = fileName + ".meta";
    FILE* fMeta = fopen(metaName.c_str(), "wb");
    if (fMeta == nullptr) {
        cout << "Error" << endl;
        exit(1);
    }

    DiskProblemMeta meta;
    meta.l = l;
    meta.n = n;
    meta.nrChunk = nrChunk;

    fwrite(reinterpret_cast<char*>(&meta), sizeof(DiskProblemMeta), 1, fMeta);
    fclose(fMeta);
}

void FtrlData::SplitChunks()
{
    string metaName = fileName  +  ".meta";
    if (Exists(metaName)) {
        FILE* fMeta = fopen(metaName.c_str(), "rb");
        DiskProblemMeta meta;
        if (fMeta == nullptr) {
            cout << "Error" << endl;
            exit(1);
        }
        size_t bytes;
        bytes = fread(reinterpret_cast<char*>(&meta), sizeof(DiskProblemMeta), 1, fMeta);
        bytes++;
        l = meta.l;
        n = meta.n;
        nrChunk = meta.nrChunk;
        for (FtrlInt chunkId = 0; chunkId < nrChunk; chunkId++) {
            FtrlChunk chunk(fileName, chunkId);
            chunks.push_back(chunk);
        }
        fclose(fMeta);
    }
    else {
        string line;
        ifstream fs(fileName);

        FtrlLong i = 0;
        FtrlInt chunkId = 0;
        FtrlChunk chunk(fileName, chunkId);
        nrChunk++;

        chunk.nnzs.push_back(i);

        while (getline(fs, line)) {
            FtrlFloat label = 0;
            istringstream iss(line);

            l++;
            chunk.l++;

            iss >> label;
            label = (label > 0) ? 1 : -1;
            chunk.labels.push_back(label);

            FtrlInt idx = 0;
            FtrlFloat val = 0;

            char dummy;
            FtrlFloat r = 0;
            FtrlInt max_nnz = 0;
            while (iss >> idx >> dummy >> val) {
                i++;
                max_nnz++;
                if (n < idx + 1) {
                    n = idx + 1;
                }
                chunk.nodes.push_back(Node(idx, val));
                r += val * val;
            }
            chunk.nnzs.push_back(i);
            chunk.R.push_back(1 / sqrt(r));
            if (i > CHUNK_SIZE) {

                chunk.nnz = i;
                chunk.Write();
                chunk.Clear();

                chunks.push_back(chunk);

                i = 0;
                chunkId++;
                chunk = FtrlChunk(fileName, chunkId);
                chunk.nnzs.push_back(i);
                nrChunk++;
            }
        }

        chunk.nnz = i;
        chunk.Write();
        chunk.Clear();

        chunks.push_back(chunk);
        FILE* fMeta = fopen(metaName.c_str(), "wb");
        if (fMeta == nullptr) {
            cout << "Error" << endl;
            exit(1);
        }
        DiskProblemMeta meta;
        meta.l = l;
        meta.n = n;
        meta.nrChunk = nrChunk;
        fwrite(reinterpret_cast<char*>(&meta), sizeof(DiskProblemMeta), 1, fMeta);
        fflush(fMeta);
        fclose(fMeta);
    }
}

void FtrlData::PrFtrlIntDataInfo()
{
    cout << "Data: " << fileName << "\t";
    cout << "#features: " << n << "\t";
    cout << "#instances: " << l << "\t";
    cout << "#chunks " << nrChunk << "\t";
    cout << endl;
}

void FtrlProblem::SaveModel(string modelPath)
{
    ofstream fOut(modelPath);
    fOut << "norm " << param->normalized << endl;
    fOut << "n " << data->n << endl;

    FtrlFloat* wa = w.data();
    FtrlFloat* na = n.data();
    FtrlFloat* za = z.data();
    for (FtrlLong j = 0; j < data->n; j++, wa++, na++, za++) {
        fOut << "w" << j << " " << *wa << " " <<  *na <<" " << *za << endl;
    }
    fOut.close();
}

FtrlLong FtrlProblem::LoadModel(string modelPath)
{
    ifstream fIn(modelPath);

    string dummy;
    FtrlLong nrFeature;

    fIn >> dummy >> normlization >> dummy >> nrFeature;
    w.resize(nrFeature);
    z.resize(nrFeature);
    n.resize(nrFeature);

    FtrlFloat* wptr = w.data();
    FtrlFloat* nptr = n.data();
    FtrlFloat* zptr = z.data();
    for (FtrlLong j = 0; j < nrFeature; j++, wptr++, nptr++, zptr++) {
        fIn >> dummy;
        fIn >> *wptr >> *nptr >> *zptr;
    }
    return nrFeature;
}

void FtrlProblem::Initialize(bool norm, string warmModelPath)
{
    f.resize(data->n, 0);
    if (warmModelPath.empty()) {
        feats = data->n;
        w.resize(data->n, 0);
        z.resize(data->n, 0);
        n.resize(data->n, 0);
    }
    else {
        FtrlLong nrFeature = LoadModel(warmModelPath);
        if (nrFeature < data->n) {
            w.resize(data->n, 0);
            z.resize(data->n, 0);
            n.resize(data->n, 0);
        }
        nrFeature = data->n;
    }
    t = 0;
    trLoss = 0.0, vaLoss = 0.0, funVal = 0.0, gnorm = 0.0;
    FtrlInt nrChunk = data->nrChunk;
    for (FtrlInt chunkId = 0; chunkId < nrChunk; chunkId++) {

        FtrlChunk& chunk = data->chunks[chunkId];

        chunk.Read();

        for (FtrlLong i = 0; i < chunk.l; i++) {

            for (FtrlInt s = chunk.nnzs[i]; s < chunk.nnzs[i + 1]; s++) {
                Node x = chunk.nodes[s];
                FtrlInt idx = x.idx;
                f[idx]++;
            }
        }
        if (!param->inMemory)
            chunk.Clear();
    }
    for (FtrlInt j = 0; j < data->n; j++) {
        if (param->freq)
            f[j] = 1;
        else
            f[j] = 1 / f[j];
    }
    startTime = omp_get_wtime();
}

void FtrlProblem::PrFtrlIntHeaderInfo()
{
    cout.width(WIDTH4);
    cout << "iter";
    if (param->verbose) {
    cout.width(WIDTH13);
    cout << "funVal";
    cout.width(WIDTH13);
    cout << "reg";
    cout.width(WIDTH13);
    cout << "|grad|";
    cout.width(WIDTH13);
    cout << "tr_logloss";
    }
    if (!testData->fileName.empty()) {
        cout.width(WIDTH13);
        cout << "va_logloss";
        cout.width(WIDTH13);
        cout << "vaAuc";
    }
    cout.width(WIDTH13);
    cout << "time";
    cout << endl;
}

void FtrlProblem::PrFtrlIntEpochInfo()
{
    cout.width(WIDTH4);
    cout << t + 1;
    if (param->verbose) {
        cout.width(WIDTH13);
        cout << scientific << setprecision(WIDTH3) << funVal;
        cout.width(WIDTH13);
        cout << scientific << setprecision(WIDTH3) << reg;
        cout.width(WIDTH13);
        cout << scientific << setprecision(WIDTH3) << gnorm;
        cout.width(WIDTH13);
        cout << fixed << setprecision(WIDTH5) << trLoss;
    }
    if (!testData->fileName.empty()) {
        cout.width(WIDTH13);
        cout << fixed << setprecision(WIDTH5) << vaLoss;
        cout.width(WIDTH13);
        cout << fixed << setprecision(WIDTH5) << vaAuc;
    }
    cout.width(WIDTH13);
    cout << fixed << setprecision(WIDTH5) << omp_get_wtime() - startTime;
    cout << endl;
}

FtrlFloat FtrlProblem::WTx(FtrlChunk& chunk, FtrlInt begin, FtrlInt end, FtrlFloat r, bool doUpdate = false)
{
    FtrlFloat p = 0;
    for (FtrlInt s = begin; s < end; s++) {
        Node& x = chunk.nodes[s];
        FtrlInt idx = x.idx;
        if (idx > data->n) {
            continue;
        }
        FtrlFloat val = x.val * r;
        if (doUpdate) {
            FtrlFloat zi, ni;
            zi = z[idx], ni = n[idx];
            if (abs(zi) > l1 * f[idx]) {
                w[idx] = -(zi - (TWO * (zi > 0) - ONE) * l1 * f[idx]) / ((b + sqrt(ni)) / a + l2 * f[idx]);
            }
            else {
                w[idx] = 0;
            }
        }
        p += w[idx] * val;
    }
    return p;
}

FtrlFloat FtrlProblem::g_calAuc(shared_ptr<FtrlData> currentData, vector<FtrlFloat>& vaLabels,\
        vector<FtrlFloat> vaScores, vector<FtrlFloat>& vaOrders)
{
    sort(vaOrders.begin(), vaOrders.end(), [&vaScores] (FtrlInt i, FtrlInt j) {return vaScores[i] < vaScores[j];});

    FtrlFloat prev_score = vaScores[0];
    FtrlLong aucM, aucN, begin, stuckPos, stuckNeg;
    aucM = 0, aucN = 0, begin = 0, stuckPos = 0, stuckNeg = 0;
    FtrlFloat sumPosRank = 0;

    for (FtrlInt i = 0; i < currentData->l; i++) {
        FtrlInt sortedI = vaOrders[i];

        FtrlFloat score = vaScores[sortedI];

        if (score !=  prev_score) {
            sumPosRank += stuckPos * (begin + begin - 1 + stuckPos + stuckNeg) * 0.5;
            prev_score = score;
            begin = i;
            stuckNeg = 0;
            stuckPos = 0;
        }

        FtrlFloat label = vaLabels[sortedI];

        if (label > 0) {
            aucM++;
            stuckPos++;
        }
        else {
            aucN++;
            stuckNeg++;
        }
    }
    sumPosRank += stuckPos * (begin + begin - ONE + stuckPos + stuckNeg) * PONE * FIVE;
    FtrlFloat auc = (sumPosRank - PONE * FIVE * aucM * (aucM + ONE)) / (aucM * aucN + EPSILON);
    return auc;
}

FtrlFloat FtrlProblem::OneEpoch(shared_ptr<FtrlData> currentData, bool doUpdate, bool doAuc,\
        FtrlFloat& auc, vector<FtrlFloat>& grad)
{
    FtrlFloat loss = 0;
    FtrlInt nrChunk, globalI;
    nrChunk = currentData->nrChunk, globalI = 0;
    vector<FtrlFloat> vaLabels(currentData->l, 0), vaScores(currentData->l, 0), vaOrders(currentData->l, 0);
    vector<FtrlInt> outerOrder(nrChunk);
    iota(outerOrder.begin(), outerOrder.end(), 0);
    random_shuffle(outerOrder.begin(), outerOrder.end());
    for (FtrlInt chunkId:outerOrder) {
        FtrlChunk& chunk = currentData->chunks[chunkId];
        chunk.Read();
        vector<FtrlInt> innerOrder(chunk.l);
        iota(innerOrder.begin(), innerOrder.end(), 0);
        random_shuffle(innerOrder.begin(), innerOrder.end());
        FtrlFloat localLoss = 0.0;
#pragma omp parallel for schedule(static)reduction( + : localLoss)
        for (FtrlLong ii = 0; ii < chunk.l; ii++) {
            FtrlInt i = innerOrder[ii];
            FtrlFloat y, p;
            FtrlFloat r = param->normalized ? chunk.R[i] : 1;
            y = chunk.labels[i], p = WTx(chunk, chunk.nnzs[i], chunk.nnzs[i + 1], r, doUpdate);

            if (doAuc) {
                vaScores[globalI + i] = p;
                vaOrders[globalI + i] = globalI + i;
                vaLabels[globalI + i] = y;
            }

            FtrlFloat expM;
            if (p * y > 0) {
                expM = exp(-y * p);
                localLoss += log(1 + expM);
            }
            else {
                expM = exp(y * p);
                localLoss += -y * p + log(1 + expM);
            }

            bool doGrad = grad.size() > 0;
            if (!doGrad && !doUpdate)
                continue;
            FtrlFloat tmp = (p * y > 0) ? expM / (1 + expM) :  1 / (1 + expM);
            FtrlFloat kappa = -y * tmp;

            for (FtrlInt s = chunk.nnzs[i]; s < chunk.nnzs[i + 1]; s++) {
                Node& x = chunk.nodes[s];
                FtrlInt idx = x.idx;
                FtrlFloat val, g, theta;
                val = x.val * r, g = kappa * val, theta = 0;
                theta = 1 / a * (sqrt(n[idx] + g * g) - sqrt(n[idx]));
                z[idx] += g - theta * w[idx];
                n[idx] += g * g;
                if (doGrad) {
                    g += l2 * f[idx] * w[idx];
                    grad[idx] += g;
                }
            }
        }
        globalI += chunk.l;
        chunk.Clear();
        loss += localLoss;
    }

    if (doAuc) {
        auc = g_calAuc(currentData, vaLabels, vaScores, vaOrders);
    }

    return loss / currentData->l;
}

void FtrlProblem::Validate()
{
    vector<FtrlFloat> grad;
    vaLoss = OneEpoch(testData, false, true, vaAuc, grad);
}

void FtrlProblem::Fun()
{
    vector<FtrlFloat> grad(data->n, 0);
    funVal = 0.0,  gnorm = 0.0, reg = 0.0;
    trLoss = OneEpoch(data, false, false, vaAuc, grad);
    for (FtrlInt j = 0; j < data->n; j++) {
        gnorm += grad[j] * grad[j];
        reg += (l1 * abs(w[j])  +  PONE * FIVE * l2 * w[j] * w[j]);
    }
    funVal = trLoss * data->l + reg;
    gnorm = sqrt(gnorm);
}

void FtrlProblem::Solve()
{
    l1 = param->l1, l2 = param->l2, a = param->alpha, b = param->beta;
    PrFtrlIntHeaderInfo();
    FtrlFloat bestVaLoss = numeric_limits<FtrlFloat>::max();
    vector<FtrlFloat> prev_w(data->n, 0);
    vector<FtrlFloat> prev_n(data->n, 0);
    vector<FtrlFloat> prev_z(data->n, 0);

    vector<FtrlFloat> grad;
    for (t = 0; t < param->nrPass; t++) {
        trLoss = OneEpoch(data, true, false, vaAuc, grad);
        if (param->verbose)
            Fun();
        if (!testData->fileName.empty())
            Validate();

        PrFtrlIntEpochInfo();
        if (param->autoStop) {
            if (vaLoss > bestVaLoss) {
#ifndef debug
                memcpy_s(w.data(), prev_w.data(), data->n * sizeof(FtrlFloat));
                memcpy_s(n.data(), prev_n.data(), data->n * sizeof(FtrlFloat));
                memcpy_s(z.data(), prev_z.data(), data->n * sizeof(FtrlFloat));
#else
                memcpy(w.data(), prev_w.data(), data->n * sizeof(FtrlFloat));
                memcpy(n.data(), prev_n.data(), data->n * sizeof(FtrlFloat));
                memcpy(z.data(), prev_z.data(), data->n * sizeof(FtrlFloat));
#endif
                cout << "Auto-stop. Use model at" << t <<"th iteration."<<endl;
                break;
            } else {
#ifndef debug
                memcpy_s(prev_w.data(), w.data(), data->n * sizeof(FtrlFloat));
                memcpy_s(prev_n.data(), n.data(), data->n * sizeof(FtrlFloat));
                memcpy_s(prev_z.data(), z.data(), data->n * sizeof(FtrlFloat));
#else
                memcpy(prev_w.data(), w.data(), data->n * sizeof(FtrlFloat));
                memcpy(prev_n.data(), n.data(), data->n * sizeof(FtrlFloat));
                memcpy(prev_z.data(), z.data(), data->n * sizeof(FtrlFloat));
#endif
                bestVaLoss = vaLoss;
            }
        }
    }
}

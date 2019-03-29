/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 */
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <iomanip>
#include <memory>
#include <cmath>
#include <stdexcept>
#include <vector>
#include <cstdlib>

#include "ftrl.h"

using namespace std;

struct Option {
    string testPath, modelPath, outputPath;
    bool error;

    Option():error(false) {};
};

string predictHelp() {
    return string(
"usage: predict test_file model_file output_file\n");
}

Option ParseOption(FtrlInt argc, vector<string>& args) {
    Option option;
    option.error = false;
    if(argc == 1) {
        cout << predictHelp() << endl;
        option.error = true;
        return option;
    }

    if(argc != 4) {
        cout << "cannot parse argument" << endl;
        option.error = true;
        return option;
    }

    option.testPath = string(args[1]);
    option.modelPath = string(args[2]);
    option.outputPath = string(args[3]);

    return option;
}

void predict(string testPath, string modelPath, string outputPath) {
    FtrlProblem prob;
    FtrlLong n = prob.LoadModel(modelPath);
    ofstream fOut(outputPath);

    shared_ptr<FtrlData> testData = make_shared<FtrlData>(testPath);
    testData->SplitChunks();
    cout << "Te_data: ";
    testData->PrFtrlIntDataInfo();

    FtrlInt nrChunk = testData->nrChunk;
    FtrlFloat localVaLoss = 0.0;

    for (FtrlInt chunkId = 0; chunkId < nrChunk; chunkId++) {

        FtrlChunk chunk = testData->chunks[chunkId];
        chunk.Read();

        for (FtrlInt i = 0; i < chunk.l; i++) {

            FtrlFloat y, wTx;
            y = chunk.labels[i], wTx = 0;

            for (FtrlInt s = chunk.nnzs[i]; s < chunk.nnzs[i + 1]; s++) {
                Node x = chunk.nodes[s];
                FtrlInt idx = x.idx;
                if (idx > n) {
                    continue;
                }
                FtrlFloat val = x.val;
                wTx += prob.w[idx] * val;
            }

            FtrlFloat expM;

            if (wTx*y > 0) {
                expM = exp(-y * wTx);
                localVaLoss += log(1 + expM);
            }
            else {
                expM = exp(y * wTx);
                localVaLoss += -y * wTx + log(1 + expM);
            }
            fOut << 1 / (1 + exp(-wTx)) << "\n";
        }
        chunk.Clear();
    }
    localVaLoss = localVaLoss / testData->l;
    cout << "logloss = " << fixed << setprecision(5) << localVaLoss << endl;
}

int main(int argc, char **argv) {
    vector<string> args;
    for(FtrlInt i = 0; i < argc; i++)
        args.push_back(string(argv[i]));
    Option option = ParseOption(argc, args);
    if(option.error == false) {
        predict(option.testPath, option.modelPath, option.outputPath);
    }
    return 0;
}

/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 */
#include <iostream>
#include <cstring>
#include <stdexcept>

#include "ftrl.h"

#include <cfenv>

struct Option {
    shared_ptr<Parameter> param;
    FtrlInt verbose, solver;
    string dataPath, testPath, modelPath, warmModelPath;
    bool error;

    Option():verbose(0), solver(1), error(false) {};
};

string BaseName(string path)
{
    size_t pos = path.rfind('/');
    if (pos == string::npos)
        return path;
    else
        return path.substr(pos+1);
}

bool IsNumerical(string str)
{
    FtrlInt c = 0;
    for (auto ch:str) {
        if (isdigit(ch))
            c++;
    }
    return c > 0;
}

string TrainHelp()
{
    return string(
    "usage: train [options] training_set_file test_set_file\n"
    "\n"
    "options:\n"
    "-s <solver>: set solver type (default 1)\n"
    "     0 -- AdaGrad framework\n"
    "     1 -- FTRL framework\n"
    "     2 -- RDA framework\n"
    "-a <alpha>: set Initial learning rate\n"
    "-b <beta>: set shrinking base for learning rate schedule\n"
    "-l1 <lambda_1>: set regularization coefficient on l1 regularizer (default 0.1)\n"
    "-l2 <lambda_2>: set regularization coefficient on l2 regularizer (default 0.1)\n"
    "-t <iter>: set number of iterations (default 20)\n"
    "-p <path>: set path to test set\n"
    "-m <path>: set path to warm model\n"
    "-c <threads>: set number of cores\n"
    "--norm: Apply instance-wise normlization.\n"
    "--no-auc: disable auc\n"
    "--in-memory: keep data in memroy\n"
    "--auto-stop: stop at the iteration that achieves the best validation loss (must be used with -p)\n"
    );
}

Option ParseOption(FtrlInt argc, vector<string>& args)
{
    Option option;
    option.error = false;
    option.verbose = 1;
    option.param = make_shared<Parameter>();

    if (argc == 1) {
        cout << TrainHelp() << endl;
        option.error = true;
        return option;
    }

    FtrlInt i = 0;
    for (i = 1; i < argc; i++) {
        if (args[i].compare("-s") == 0) {
            if ((i + 1) >= argc) {
                cout << "need to specify solver type\
                                        after -s" << endl;
                option.error = true;
            }
            i++;

            if (!IsNumerical(args[i])) {
                cout << "-s should be followed by a number" << endl;
                option.error = true;
            }
            option.solver = atoi(args[i].c_str());
        }
        else if (args[i].compare("-l1") == 0) {
            if ((i + 1) >= argc) {
                cout << "need to specify l1 regularization\
                                        coefficient after -l1" << endl;
                option.error = true;
            }
            i++;

            if (!IsNumerical(args[i])) {
                cout << "-l1 should be followed by a number" << endl;
                option.error = true;
            }
            option.param->l1 = atof(args[i].c_str());
        }
        else if (args[i].compare("-l2") == 0) {
            if ((i + 1) >= argc) {
                cout << "need to specify l2\
                                        regularization coefficient\
                                        after -l2" << endl;
                option.error = true;
            }
            i++;

            if (!IsNumerical(args[i])) {
                cout << "-l2 should be followed by a number" << endl;
                option.error = true;
            }
            option.param->l2 = atof(args[i].c_str());
        }
        else if (args[i].compare("-t") == 0) {
            if ((i + 1) >= argc) {
                cout << "need to specify max number of\
                                        iterations after -t" << endl;
                option.error = true;
            }
            i++;

            if (!IsNumerical(args[i])) {
                cout << "-t should be followed by a number" << endl;
                option.error = true;
            }
            option.param->nrPass = atoi(args[i].c_str());
        }
        else if (args[i].compare("-a") == 0) {
            if ((i + 1) >= argc) {
                cout << "missing core numbers after -c" << endl;
                option.error = true;
            }
            i++;
            if (!IsNumerical(args[i])) {
                cout << "-c should be followed by a number" << endl;
                option.error = true;
            }
            option.param->alpha = atof(args[i].c_str());
        }
        else if (args[i].compare("-b") == 0) {
            if ((i + 1) >= argc) {
                cout << "missing core numbers after -c" << endl;
                option.error = true;
            }
            i++;
            if (!IsNumerical(args[i])) {
                cout << "-c should be followed by a number" << endl;
                option.error = true;
            }
            option.param->beta = atof(args[i].c_str());
        }
        else if (args[i].compare("-c") == 0) {
            if ((i + 1) >= argc) {
                cout << "missing core numbers after -c" << endl;
                option.error = true;
            }
            i++;
            if (!IsNumerical(args[i])) {
                cout << "-c should be followed by a number" << endl;
                option.error = true;
            }
            option.param->nrThreads = atof(args[i].c_str());
        }
        else if (args[i].compare("-p") == 0) {
            if (i == argc-1) {
                cout << "need to specify path after -p" << endl;
                option.error = true;
            }
            i++;

            option.testPath = string(args[i]);
        }
        else if (args[i].compare("-m") == 0) {
            if (i == argc-1) {
                cout << "need to specify warmstart model path after -m" << endl;
                option.error = true;
            }
            i++;

            option.warmModelPath = string(args[i]);
        }
        else if (args[i].compare("--norm") == 0) {
            option.param->normalized = true;
        }
        else if (args[i].compare("--verbose") == 0) {
            option.param->verbose = true;
        }
        else if (args[i].compare("--freq") == 0) {
            option.param->freq = true;
        }
        else if (args[i].compare("--auto-stop") == 0) {
            option.param->autoStop = true;
        }
        else if (args[i].compare("--no-auc") == 0) {
            option.param->noAuc = true;
        }
        else if (args[i].compare("--in-memory") == 0) {
            option.param->inMemory = true;
        }
        else {
            break;
        }
    }

    if (i != argc-2 && i != argc-1) {
        cout << "cannot parse commmand" << endl;
        option.error = true;
        return option;
    }
    option.dataPath = string(args[i++]);

    if (i < argc) {
        option.modelPath = string(args[i]);
    } else if (i == argc) {
        option.modelPath = BaseName(option.dataPath)  +  ".model";
    } else {
        cout << "cannot parse commmand" << endl;
        option.error = true;
    }

    return option;
}

int main(int argc, char *argv[])
{
    vector<string> args;
    for (FtrlInt i = 0; i < argc; i++)
        args.push_back(string(argv[i]));
    Option option = ParseOption(argc, args);
    if (option.error == false) {
        omp_set_num_threads(option.param->nrThreads);

        shared_ptr<FtrlData> data = make_shared<FtrlData>(option.dataPath);
        shared_ptr<FtrlData> testData = make_shared<FtrlData>(option.testPath);
        data->SplitChunks();
        cout << "Tr_data: ";
        data->PrFtrlIntDataInfo();

        if (!testData->fileName.empty()) {
            testData->SplitChunks();
            cout << "Va_data: ";
            testData->PrFtrlIntDataInfo();
        }

        FtrlProblem prob(data, testData, option.param);
        prob.Initialize(option.param->normalized, option.warmModelPath);
        if (option.solver == 1) {
            cout << "Solver Type: FTRL" << endl;
            prob.Solve();
        }
        prob.SaveModel(option.modelPath);
    }
    return 0;
}

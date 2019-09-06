#ifndef FTRL_H
#define FTRL_H

#include <vector>
#include <algorithm>
#include <string>
#include <memory>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <cstdio>
#include <iomanip>
#include <cstring>
#include <map>
#include <set>

#include "omp.h"

using namespace std;


typedef double FtrlFloat;
typedef long FtrlInt;
typedef long long FtrlLong;

FtrlLong const chunk_size = 3000000000;

class Node {
public:
    FtrlLong idx;
    FtrlFloat  val;
    Node(){};
    Node(FtrlLong idx, FtrlFloat val): idx(idx), val(val){};
};

class Parameter {

public:
    FtrlFloat l1, l2, l3, alpha, beta;
    FtrlInt nr_pass = 20, nr_threads = 1;
    bool normalized, verbose, freq, auto_stop, no_auc, in_memory, one_pass, weight, save_update, causE, updated;
    Parameter():l1(0.1), l2(0.1), l3(0), alpha(0.1), beta(1), normalized(false), verbose(true), freq(false), auto_stop(false), no_auc(false), in_memory(false), one_pass(false), weight(false), save_update(false), causE(false), updated(false){};
};

class FtrlChunk {
public:
    FtrlLong l, nnz;
    FtrlInt chunk_id;
    string file_name;

    vector<Node> nodes;
    vector<FtrlInt> nnzs;
    vector<FtrlFloat> labels;
    vector<FtrlFloat> R;


    void read();
    void write();
    void clear();

    FtrlChunk(string data_name, FtrlInt chunk_id);
};

class FtrlData {
public:
    string file_name;
    FtrlLong l, n;
    FtrlInt nr_chunk;
    string meta_name;
    string profile_name;
    bool weighted;

    vector<FtrlChunk> chunks;
    shared_ptr<map<FtrlLong, string>> pos_featmap;
    vector<FtrlInt> length;
    vector<FtrlFloat> weight;
    FtrlFloat sum_weight;
    set<FtrlLong> nnz_idx;

    FtrlData(string file_name): file_name(file_name), l(0), n(0), nr_chunk(0), weighted(false), sum_weight(0) {
        meta_name = file_name + ".meta";
    };
    void print_data_info();
    void split_chunks();
    void write_meta();
    void read_meta();
    void parse_profile(string profile_name);
};

class FtrlProblem {
public:
    shared_ptr<FtrlData> data;
    shared_ptr<FtrlData> test_data;
    shared_ptr<Parameter> param;
    FtrlProblem() {};
    FtrlProblem(shared_ptr<FtrlData> &data, shared_ptr<FtrlData> &test_data, shared_ptr<Parameter> &param)
        :data(data), test_data(test_data), param(param) {};

    vector<FtrlFloat> w, z, n, f;
    map<FtrlLong, FtrlFloat*> wznf_map;
	bool normalization = false, use_map = false;
    FtrlInt t = 0;
	FtrlLong feats = 0;
    FtrlFloat tr_loss = 0.0f, va_loss = 0.0f, va_auc = 0.0f, fun_val = 0.0f, gnorm = 0.0f, reg = 0.0f;
    FtrlFloat start_time = 0.0f;

    void initialize(bool norm, string warm_model_path);
    void solve();
    void solve_adagrad();
    void solve_rda();
    void split_train();
    void print_epoch_info();
    void print_header_info();
    void save_model(string model_path);
    FtrlLong load_model(string model_path);
    void save_model_txt(string model_path);
    void save_model_updated_txt(string model_path);
    FtrlLong load_model_txt(string model_path);
    void fun();
    void validate();
};

FtrlFloat cal_auc(vector<FtrlFloat> &va_labels, vector<FtrlFloat> &va_scores, vector<FtrlFloat> &va_orders);
void causE(FtrlProblem &prob_sc, FtrlProblem &prob_st);
#endif

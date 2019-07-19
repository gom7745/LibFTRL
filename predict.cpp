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

struct Option
{
    string test_path, model_path, output_path;
};

string predict_help()
{
    return string(
"usage: predict test_file model_file output_file\n");
}

Option parse_option(int argc, char **argv)
{
    vector<string> args;
    for(int i = 0; i < argc; i++)
        args.push_back(string(argv[i]));

    if(argc == 1)
        throw invalid_argument(predict_help());

    Option option;

    if(argc != 4)
        throw invalid_argument("cannot parse argument");

    option.test_path = string(args[1]);
    option.model_path = string(args[2]);
    option.output_path = string(args[3]);

    return option;
}

void predict(string test_path, string model_path, string output_path)
{
    FtrlProblem prob;
    FtrlLong n = prob.load_model_txt(model_path);
    ofstream f_out(output_path);

    shared_ptr<FtrlData> test_data = make_shared<FtrlData>(test_path);
    test_data->split_chunks();
    cout << "Te_data: ";
    test_data->print_data_info();

    FtrlInt nr_chunk = test_data->nr_chunk, global_i = 0;
    FtrlFloat local_va_loss = 0.0, va_auc = 0.0;
    vector<FtrlFloat> va_labels(test_data->l, 0), va_scores(test_data->l, 0), va_orders(test_data->l, 0);

    for (FtrlInt chunk_id = 0; chunk_id < nr_chunk; chunk_id++) {

        FtrlChunk chunk = test_data->chunks[chunk_id];
        chunk.read();

#pragma omp parallel for schedule(static) reduction(+: local_va_loss)
        for (FtrlInt i = 0; i < chunk.l; i++) {

            FtrlFloat y = chunk.labels[i], wTx = 0;
            FtrlFloat r=prob.normalization ? chunk.R[i]:1;

            for (FtrlInt s = chunk.nnzs[i]; s < chunk.nnzs[i+1]; s++) {
                Node x = chunk.nodes[s];
                FtrlInt idx = x.idx;
                if (idx > n) {
                    continue;
                }
                FtrlFloat val = x.val*r;
                wTx += prob.w[idx]*val;
            }
            va_scores[global_i+i] = wTx;
            va_orders[global_i+i] = global_i+i;
            va_labels[global_i+i] = y;

            FtrlFloat exp_m;

            if (wTx*y > 0) {
                exp_m = exp(-y*wTx);
                local_va_loss += log(1+exp_m);
            }
            else {
                exp_m = exp(y*wTx);
                local_va_loss += -y*wTx+log(1+exp_m);
            }
            //f_out << 1/(1+exp(-wTx)) << "\n";
        }
        global_i += chunk.l;
        chunk.clear();
    }
    local_va_loss = local_va_loss / test_data->l;
    va_auc = cal_auc(va_labels, va_scores, va_orders);
    cout << "logloss = " << fixed << setprecision(5) << local_va_loss << endl;
    cout << "auc = " << fixed << setprecision(5) << va_auc << endl;
}

int main(int argc, char **argv)
{
    Option option;
    try
    {
        option = parse_option(argc, argv);
    }
    catch(invalid_argument const &e)
    {
        cout << e.what() << endl;
        return 1;
    }

    predict(option.test_path, option.model_path, option.output_path);
}

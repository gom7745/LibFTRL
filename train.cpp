#include <iostream>
#include <cstring>
#include <stdexcept>

#include "ftrl.h"

#include <fenv.h>


struct Option
{
    shared_ptr<Parameter> param;
    FtrlInt verbose, solver;
    string data_path, test_path, model_path, warm_model_path;
    bool error;
};

string basename(string path)
{
    const char *ptr = strrchr(&*path.begin(), '/');
    if(!ptr)
        ptr = path.c_str();
    else
        ptr++;
    return string(ptr);
}

bool is_numerical(char *str)
{
    int c = 0;
    while(*str != '\0')
    {
        if(isdigit(*str))
            c++;
        str++;
    }
    return c > 0;
}

string train_help()
{
    return string(
    "usage: train [options] training_set_file test_set_file\n"
    "\n"
    "options:\n"
    "-s <solver>: set solver type (default 1)\n"
    "     0 -- AdaGrad framework\n"
    "     1 -- FTRL framework\n"
    "     2 -- RDA framework\n"
    "-a <alpha>: set initial learning rate\n"
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

Option parse_option(int argc, char **argv)
{
    vector<string> args;
    for(int i = 0; i < argc; i++)
        args.push_back(string(argv[i]));

    Option option;
    option.error = false;
    option.verbose = 1;
    option.param = make_shared<Parameter>();

    if(argc == 1) {
        cout << train_help() << endl;
        option.error = true;
        return option;
    }

    int i = 0;
    for(i = 1; i < argc; i++)
    {
        if(args[i].compare("-s") == 0)
        {
            if((i+1) >= argc) {
                cout << "need to specify solver type\
                                        after -s" << endl;
                option.error = true;
            }
            i++;

            if(!is_numerical(argv[i])) {
                cout << "-s should be followed by a number" << endl;
                option.error = true;
            }
            option.solver = atoi(argv[i]);
        }
        else if(args[i].compare("-l1") == 0)
        {
            if((i+1) >= argc) {
                cout << "need to specify l1 regularization\
                                        coefficient after -l1" << endl;
                option.error = true;
            }
            i++;

            if(!is_numerical(argv[i])) {
                cout << "-l1 should be followed by a number" << endl;
                option.error = true;
            }
            option.param->l1 = atof(argv[i]);
        }
        else if(args[i].compare("-l2") == 0)
        {
            if((i+1) >= argc) {
                cout << "need to specify l2\
                                        regularization coefficient\
                                        after -l2" << endl;
                option.error = true;
            }
            i++;

            if(!is_numerical(argv[i])) {
                cout << "-l2 should be followed by a number" << endl;
                option.error = true;
            }
            option.param->l2 = atof(argv[i]);
        }
        else if(args[i].compare("-t") == 0)
        {
            if((i+1) >= argc) {
                cout << "need to specify max number of\
                                        iterations after -t" << endl;
                option.error = true;
            }
            i++;

            if(!is_numerical(argv[i])) {
                cout << "-t should be followed by a number" << endl;
                option.error = true;
            }
            option.param->nr_pass = atoi(argv[i]);
        }
        else if(args[i].compare("-a") == 0)
        {
            if((i+1) >= argc) {
                cout << "missing core numbers after -c" << endl;
                option.error = true;
            }
            i++;
            if(!is_numerical(argv[i])) {
                cout << "-c should be followed by a number" << endl;
                option.error = true;
            }
            option.param->alpha = atof(argv[i]);
        }
        else if(args[i].compare("-b") == 0)
        {
            if((i+1) >= argc) {
                cout << "missing core numbers after -c" << endl;
                option.error = true;
            }
            i++;
            if(!is_numerical(argv[i])) {
                cout << "-c should be followed by a number" << endl;
                option.error = true;
            }
            option.param->beta = atof(argv[i]);
        }
        else if(args[i].compare("-c") == 0)
        {
            if((i+1) >= argc) {
                cout << "missing core numbers after -c" << endl;
                option.error = true;
            }
            i++;
            if(!is_numerical(argv[i])) {
                cout << "-c should be followed by a number" << endl;
                option.error = true;
            }
            option.param->nr_threads = atof(argv[i]);
        }
        else if(args[i].compare("-p") == 0)
        {
            if(i == argc-1) {
                cout << "need to specify path after -p" << endl;
                option.error = true;
            }
            i++;

            option.test_path = string(args[i]);
        }
        else if(args[i].compare("-m") == 0)
        {
            if(i == argc-1) {
                cout << "need to specify warmstart model path after -m" << endl;
                option.error = true;
            }
            i++;

            option.warm_model_path = string(args[i]);
        }
        else if(args[i].compare("--norm") == 0)
        {
            option.param->normalized = true;
        }
        else if(args[i].compare("--verbose") == 0)
        {
            option.param->verbose = true;
        }
        else if(args[i].compare("--freq") == 0)
        {
            option.param->freq = true;
        }
        else if(args[i].compare("--auto-stop") == 0)
        {
            option.param->auto_stop = true;
        }
        else if(args[i].compare("--no-auc") == 0)
        {
            option.param->no_auc = true;
        }
        else if(args[i].compare("--in-memory") == 0)
        {
            option.param->in_memory = true;
        }
        else
        {
            break;
        }
    }

    if(i != argc-2 && i != argc-1) {
        cout << "cannot parse commmand" << endl;
        option.error = true;
        return option;
    }
    option.data_path = string(args[i++]);

    if(i < argc) {
        option.model_path = string(args[i]);
    } else if(i == argc) {
        option.model_path = basename(option.data_path) + ".model";
    } else {
        cout << "cannot parse commmand" << endl;
        option.error = true;
    }

    return option;
}

int main(int argc, char *argv[])
{
    Option option = parse_option(argc, argv);
    if(option.error == false)
    {
        omp_set_num_threads(option.param->nr_threads);

        shared_ptr<FtrlData> data = make_shared<FtrlData>(option.data_path);
        shared_ptr<FtrlData> test_data = make_shared<FtrlData>(option.test_path);
        data->split_chunks();
        cout << "Tr_data: ";
        data->print_data_info();

        if (!test_data->file_name.empty()) {
            test_data->split_chunks();
            cout << "Va_data: ";
            test_data->print_data_info();
        }

        FtrlProblem prob(data, test_data, option.param);
        prob.initialize(option.param->normalized, option.warm_model_path);
        if (option.solver == 1) {
            cout << "Solver Type: FTRL" << endl;
            prob.solve();
        }
        prob.save_model(option.model_path);
    }
    return 0;
}

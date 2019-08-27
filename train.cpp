#include <iostream>
#include <cstring>
#include <stdexcept>

#include "ftrl.h"
#include "tools.h"

#include <fenv.h>
#include <cassert>


struct Option
{
    shared_ptr<Parameter> param;
    FtrlInt verbose, solver;
    string data_path, test_path, model_path, warm_model_path, data_profile, test_profile, featmap_path, pos_bias_map_path, warm_update_model_path;
    string data_path_st, test_path_st, model_path_st, warm_model_path_st;
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
    "usage: train [options] training_set_file training_set_profile [model_file]\n"
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
    "-l3 <lambda_3>: set regularization coefficient on l2 regularizer for CausE (default 0)\n"
    "-t <iter>: set number of iterations (default 20)\n"
    "-tr-st <path>: set path to st train set\n"
    "-p-st <path>: set path to st test set\n"
    "-m-st <path>: set path to st warm model\n"
    "-save-m-st <path>: set path to st model\n"
    "-p <path>: set path to test set\n"
    "-m <path>: set path to warm model\n"
    "-c <threads>: set number of cores\n"
    "-fm <path>: set path to featuremap\n"
    "-dp <path>: set path to data profile\n"
    "-tp <path>: set path to test profile\n"
    "--norm: Apply instance-wise normalization.\n"
    "--freq: Apply frequency calibrated regularization.\n"
    "--no-auc: disable auc\n"
    "--in-memory: keep data in memroy\n"
    "--one-pass: train wihtout generate binary files\n"
    "--weight: train with weighted instance\n"
    "--save-update: save updated weight only\n"
    "--causE: causE FTRL\n"
    "--auto-stop: stop at the iteration that achieves the best validation loss (must be used with -p)\n"
    );
}

Option parse_option(int argc, char **argv)
{
    vector<string> args;
    for(int i = 0; i < argc; i++)
        args.push_back(string(argv[i]));

    if(argc == 1)
        throw invalid_argument(train_help());

    Option option;
    option.verbose = 1;
    option.param = make_shared<Parameter>();
    int i = 0;
    for(i = 1; i < argc; i++)
    {
        if(args[i].compare("-s") == 0)
        {
            if((i+1) >= argc)
                throw invalid_argument("need to specify solver type\
                                        after -s");
            i++;

            if(!is_numerical(argv[i]))
                throw invalid_argument("-s should be followed by a number");
            option.solver = atoi(argv[i]);
        }
        else if(args[i].compare("-l1") == 0)
        {
            if((i+1) >= argc)
                throw invalid_argument("need to specify l1 regularization\
                                        coefficient after -l1");
            i++;

            if(!is_numerical(argv[i]))
                throw invalid_argument("-l1 should be followed by a number");
            option.param->l1 = atof(argv[i]);
        }
        else if(args[i].compare("-l2") == 0)
        {
            if((i+1) >= argc)
                throw invalid_argument("need to specify l2\
                                        regularization coefficient\
                                        after -l2");
            i++;

            if(!is_numerical(argv[i]))
                throw invalid_argument("-l2 should be followed by a number");
            option.param->l2 = atof(argv[i]);
        }
        else if(args[i].compare("-l3") == 0)
        {
            if((i+1) >= argc)
                throw invalid_argument("need to specify l3\
                                        regularization coefficient\
                                        after -l3");
            i++;

            if(!is_numerical(argv[i]))
                throw invalid_argument("-l3 should be followed by a number");
            option.param->l3 = atof(argv[i]);
        }
        else if(args[i].compare("-t") == 0)
        {
            if((i+1) >= argc)
                throw invalid_argument("need to specify max number of\
                                        iterations after -t");
            i++;

            if(!is_numerical(argv[i]))
                throw invalid_argument("-t should be followed by a number");
            option.param->nr_pass = atoi(argv[i]);
        }
        else if(args[i].compare("-a") == 0)
        {
            if((i+1) >= argc)
                throw invalid_argument("missing core numbers after -c");
            i++;
            if(!is_numerical(argv[i]))
                throw invalid_argument("-c should be followed by a number");
            option.param->alpha = atof(argv[i]);
        }
        else if(args[i].compare("-b") == 0)
        {
            if((i+1) >= argc)
                throw invalid_argument("missing core numbers after -c");
            i++;
            if(!is_numerical(argv[i]))
                throw invalid_argument("-c should be followed by a number");
            option.param->beta = atof(argv[i]);
        }
        else if(args[i].compare("-c") == 0)
        {
            if((i+1) >= argc)
                throw invalid_argument("missing core numbers after -c");
            i++;
            if(!is_numerical(argv[i]))
                throw invalid_argument("-c should be followed by a number");
            option.param->nr_threads = atof(argv[i]);
        }
        else if(args[i].compare("-p-st") == 0)
        {
            if(i == argc-1)
                throw invalid_argument("need to specify path after -p-st");
            i++;

            option.test_path_st = string(args[i]);
        }
        else if(args[i].compare("-tr-st") == 0)
        {
            if(i == argc-1)
                throw invalid_argument("need to specify path after -tr-st");
            i++;

            option.data_path_st = string(args[i]);
        }
        else if(args[i].compare("-m-st") == 0)
        {
            if(i == argc-1)
                throw invalid_argument("need to specify path after -m-st");
            i++;

            option.warm_model_path_st = string(args[i]);
        }
        else if(args[i].compare("-save-m-st") == 0)
        {
            if(i == argc-1)
                throw invalid_argument("need to specify path after -save-m-st");
            i++;

            option.model_path_st = string(args[i]);
        }
        else if(args[i].compare("-p") == 0)
        {
            if(i == argc-1)
                throw invalid_argument("need to specify path after -p");
            i++;

            option.test_path = string(args[i]);
        }
        else if(args[i].compare("-fm") == 0)
        {
            if(i == argc-1)
                throw invalid_argument("need to specify path after -fm");
            i++;

            option.featmap_path = string(args[i]);
        }
        else if(args[i].compare("-dp") == 0)
        {
            if(i == argc-1)
                throw invalid_argument("need to specify path after -dp");
            i++;

            option.data_profile = string(args[i]);
        }
        else if(args[i].compare("-tp") == 0)
        {
            if(i == argc-1)
                throw invalid_argument("need to specify path after -tp");
            i++;

            option.test_profile = string(args[i]);
        }
        else if(args[i].compare("-m") == 0)
        {
            if(i == argc-1)
                throw invalid_argument("need to specify warmstart model path after -m");
            i++;

            option.warm_model_path = string(args[i]);
        }
        else if(args[i].compare("--norm") == 0)
        {
            option.param->normalized = true;
        }
        else if(args[i].compare("--freq") == 0)
        {
            option.param->freq = true;
        }
        else if(args[i].compare("--verbose") == 0)
        {
            option.param->verbose = true;
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
        else if(args[i].compare("--one-pass") == 0)
        {
            option.param->one_pass = true;
        }
        else if(args[i].compare("--weight") == 0)
        {
            option.param->weight = true;
        }
        else if(args[i].compare("--save-update") == 0)
        {
            option.param->save_update = true;
        }
        else if(args[i].compare("--causE") == 0)
        {
            option.param->causE = true;
        }
        else
        {
            break;
        }
    }

    if(i != argc-2 && i != argc-1)
        throw invalid_argument("cannot parse commmand\n");
    option.data_path = string(args[i++]);

    if(i < argc) {
        option.model_path = string(args[i]);
    } else if(i == argc) {
        option.model_path = basename(option.data_path) + ".model";
    } else {
        throw invalid_argument("cannot parse commmand\n");
    }

    return option;
}

int main(int argc, char *argv[])
{
    try
    {
        Option option = parse_option(argc, argv);
        omp_set_num_threads(option.param->nr_threads);

        shared_ptr<FtrlData> data = make_shared<FtrlData>(option.data_path);
        shared_ptr<FtrlData> test_data = make_shared<FtrlData>(option.test_path);
        if(!option.data_profile.empty())
            data->parse_profile(option.data_profile);
        if (!option.param->one_pass)
            data->split_chunks();
        cout << "Tr_data: ";
        data->print_data_info();

        if (!test_data->file_name.empty()) {
            if(!option.test_profile.empty())
                test_data->parse_profile(option.test_profile);
            if(!option.param->one_pass)
                test_data->split_chunks();
            cout << "Va_data: ";
            test_data->print_data_info();
        }
        if(!option.featmap_path.empty()) {
            map<FtrlLong, string> pos_featmap = get_pos_featmap(option.featmap_path);
            data->pos_featmap = make_shared<map<FtrlLong, string>> (pos_featmap);
            test_data->pos_featmap = make_shared<map<FtrlLong, string>> (pos_featmap);
        }

        FtrlProblem prob(data, test_data, option.param);
        prob.initialize(option.param->normalized, option.warm_model_path);
        if (option.param->causE) {
            cout << "Solver Type: CausE FTRL" << endl;
            shared_ptr<FtrlData> data_st = make_shared<FtrlData>(option.data_path_st);
            shared_ptr<FtrlData> test_data_st = make_shared<FtrlData>(option.test_path_st);
            data_st->split_chunks();
            cout << "Tr_data St: ";
            data_st->print_data_info();
            if (!test_data_st->file_name.empty()) {
                test_data_st->split_chunks();
                cout << "Va_data St: ";
                test_data->print_data_info();
            }
            assert(!test_data_st->file_name.empty() && !data_st->file_name.empty());
            FtrlProblem prob_st(data_st, test_data_st, option.param);
            prob_st.initialize(option.param->normalized, option.warm_model_path_st);
            causE(prob, prob_st);
            prob.save_model_txt(option.model_path);
            prob_st.save_model_txt(option.model_path_st);
        }
        else {
            if (option.solver == 1) {
                cout << "Solver Type: FTRL" << endl;
                if(option.param->one_pass)
                    prob.split_train();
                else
                    prob.solve();
            }
            else if (option.solver == 2) {
                cout << "Solver Type: RDA" << endl;
                prob.solve_rda();
            }
            else {
                cout << "Solver Type: AdaGrad" << endl;
                prob.solve_adagrad();
            }
            prob.save_model_txt(option.model_path);
        }
    }
    catch (invalid_argument &e)
    {
        cerr << e.what() << endl;
        return 1;
    }
    return 0;
}

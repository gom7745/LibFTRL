#include "ftrl.h"

FtrlFloat CausE_one_epoch(FtrlProblem &prob, FtrlProblem &prob_r, shared_ptr<FtrlData> &data, bool update) {
    FtrlFloat l1 = prob.param->l1, l2 = prob.param->l2, l3 = prob.param->l3, a = prob.param->alpha, b = prob.param->beta;
    FtrlInt global_i = 0;
    vector<FtrlFloat> &w = prob.w, &z = prob.z, &n = prob.n, &f = prob.f;
    vector<FtrlFloat> &w_r = prob_r.w; //&z_r = prob_r.z, &n_r = prob_r.n, &f_r = prob_r.f;
    vector<FtrlInt> outer_order(data->nr_chunk);
    iota(outer_order.begin(), outer_order.end(), 0);
    random_shuffle(outer_order.begin(),outer_order.end());
    FtrlFloat loss = 0.0;
    for (auto chunk_id:outer_order) {
        FtrlChunk chunk = data->chunks[chunk_id];
        if(!prob.param->in_memory)
            chunk.read();
        vector<FtrlInt> inner_oder(chunk.l);
        iota(inner_oder.begin(), inner_oder.end(),0);
        random_shuffle(inner_oder.begin(), inner_oder.end());
        FtrlFloat local_loss = 0.0;

#pragma omp parallel for schedule(guided) reduction(+: local_loss)
        for (FtrlInt ii = 0; ii < chunk.l; ii++) {
            FtrlInt i = inner_oder[ii];
            FtrlFloat y=chunk.labels[i], wTx=0;
            FtrlFloat r=prob.param->normalized ? chunk.R[i]:1;

            for (FtrlInt s = chunk.nnzs[i]; s < chunk.nnzs[i+1]; s++) {
                Node x = chunk.nodes[s];
                FtrlInt idx = x.idx;
                FtrlFloat val = x.val*r, zi = z[idx], ni = n[idx];
                FtrlFloat cond = zi - l3*w_r[idx];
                if(update) {
                    if (abs(cond) > l1*f[idx]) {
                        w[idx] = -(cond-(2*(cond>0)-1)*l1*f[idx]) / ((b+sqrt(ni))/a+l2*f[idx]);
                    }
                    else {
                        w[idx] = 0;
                    }
                }
                wTx += w[idx]*val;
            }

            FtrlFloat exp_m, tmp, weight;
            if(data->weighted)
                weight = data->weight[global_i + i] / data->sum_weight;
            else
                weight = 1;

            if (wTx*y > 0) {
                exp_m = exp(-y*wTx);
                tmp = exp_m/(1+exp_m) * weight;
                local_loss += log(1+exp_m) * weight;
            }
            else {
                exp_m = exp(y*wTx);
                tmp = 1/(1+exp_m) * weight;
                local_loss += -y*wTx+log(1+exp_m) * weight;
            }

            FtrlFloat kappa = -y*tmp;

            if(update) {
                FtrlFloat g_norm = 0;
                for (FtrlInt s = chunk.nnzs[i]; s < chunk.nnzs[i+1]; s++) {
                    Node x = chunk.nodes[s];
                    FtrlInt idx = x.idx;
                    FtrlFloat val = x.val*r, g = kappa*val, theta=0;
                    g_norm += g*g;
                    theta = 1/a*(sqrt(n[idx]+g*g)-sqrt(n[idx]));
                    z[idx] += g-theta*w[idx];
                    n[idx] += g*g;
                }
            }
        }
        global_i += chunk.l;
        if(!prob.param->in_memory)
            chunk.clear();
        loss += local_loss;
    }
    return loss / data->l ;
}

FtrlChunk::FtrlChunk(string data_name, FtrlInt id) {
    l = 0, nnz = 0;
    chunk_id = id;
    file_name = data_name+".bin."+to_string(id);
}

struct chunk_meta {
    FtrlLong l, nnz;
    FtrlInt chunk_id;
};

void FtrlChunk::write() {
    ofstream f_bin(file_name, ios::out | ios::binary);

    chunk_meta meta;
    meta.l = l;
    meta.nnz = nnz;
    meta.chunk_id = chunk_id;

    f_bin.write(reinterpret_cast<char*>(&meta), sizeof(chunk_meta));
    f_bin.write(reinterpret_cast<char*>(labels.data()), sizeof(FtrlFloat) * l);
    f_bin.write(reinterpret_cast<char*>(nnzs.data()), sizeof(FtrlInt) * (l+1));
    f_bin.write(reinterpret_cast<char*>(R.data()), sizeof(FtrlFloat) * l);
    f_bin.write(reinterpret_cast<char*>(nodes.data()), sizeof(Node) * nnz);
}

void FtrlChunk::read() {
    ifstream f_bin(file_name, ios::in | ios::binary);

    chunk_meta meta;

    f_bin.read(reinterpret_cast<char *>(&meta), sizeof(chunk_meta));
    l = meta.l;
    nnz = meta.nnz;
    chunk_id = meta.chunk_id;

    labels.resize(l);
    R.resize(l);
    nodes.resize(nnz);
    nnzs.resize(l+1);

    f_bin.read(reinterpret_cast<char*>(labels.data()), sizeof(FtrlFloat) * l);
    f_bin.read(reinterpret_cast<char*>(nnzs.data()), sizeof(FtrlInt) * (l+1));
    f_bin.read(reinterpret_cast<char*>(R.data()), sizeof(FtrlFloat) * l);
    f_bin.read(reinterpret_cast<char*>(nodes.data()), sizeof(Node) * nnz);
}

void FtrlChunk::clear() {
    labels.clear();
    nodes.clear();
    R.clear();
    nnzs.clear();
}

inline bool exists(const string& name) {
    ifstream f(name.c_str());
    return f.good();
}

struct disk_problem_meta {
    FtrlLong l, n;
    FtrlInt nr_chunk;
};

void FtrlData::write_meta() {
    string meta_name = file_name + ".meta";
    ofstream f_meta(meta_name, ios::out | ios::binary);

    disk_problem_meta meta;
    meta.l = l;
    meta.n = n;
    meta.nr_chunk = nr_chunk;

    f_meta.write(reinterpret_cast<char*>(&meta), sizeof(disk_problem_meta));
}

void FtrlData::read_meta() {
    ifstream f_meta(meta_name, ios::in | ios::binary);

    disk_problem_meta meta;

    f_meta.read(reinterpret_cast<char*>(&meta), sizeof(disk_problem_meta));
    l = meta.l;
    n = meta.n;
    nr_chunk = meta.nr_chunk;
}

void FtrlData::parse_profile(string profile_name) {
    this->profile_name = profile_name;
    ifstream fs(profile_name);

    string line, dummy;
    fs >> dummy >> this->n;
    fs >> dummy >> this->weighted;
    FtrlInt length;
    FtrlFloat reweight, sum_reweight = 0;
    while(fs >> length >> reweight) {
        this->length.push_back(length);
        this->weight.push_back(reweight);
        sum_reweight += reweight;
    }

    for(unsigned int i=0;i<this->weight.size();i++) {
        this->weight[i] = sum_reweight / this->weight[i];
        this->sum_weight += this->weight[i];
    }
}

void FtrlData::split_chunks() {
    if(exists(meta_name)) {
        read_meta();
        for(FtrlInt chunk_id=0; chunk_id < nr_chunk; chunk_id++) {
            FtrlChunk chunk(file_name, chunk_id);
            chunks.push_back(chunk);
        }
    }
    else {
        string line;
        ifstream fs(file_name);

        FtrlInt i = 0, chunk_id = 0;
        FtrlChunk chunk(file_name, chunk_id);
        nr_chunk++;

        chunk.nnzs.push_back(i);

        while (getline(fs, line)) {
            FtrlFloat label = 0;
            istringstream iss(line);

            l++;
            chunk.l++;

            iss >> label;
            label = (label>0)? 1:-1;
            chunk.labels.push_back(label);

            FtrlInt idx = 0;
            FtrlFloat val = 0;

            char dummy;
            FtrlFloat r = 0;
            FtrlInt max_nnz = 0;
            while (iss >> idx >> dummy >> val) {
                i++;
                max_nnz++;
                if (n < idx+1) {
                    n = idx+1;
                }
                nnz_idx.insert(idx);
                chunk.nodes.push_back(Node(idx, val));
                r += val*val;
            }
            chunk.nnzs.push_back(i);
            chunk.R.push_back(1/sqrt(r));
            if (i > chunk_size) {

                chunk.nnz = i;
                chunk.write();
                chunk.clear();

                chunks.push_back(chunk);

                i = 0;
                chunk_id++;
                chunk = FtrlChunk(file_name, chunk_id);
                chunk.nnzs.push_back(i);
                nr_chunk++;
            }
        }

        chunk.nnz = i;
        chunk.write();
        chunk.clear();

        chunks.push_back(chunk);
        write_meta();
    }
}

void FtrlProblem::split_train() {
    print_header_info();
    FtrlFloat l1 = param->l1, l2 = param->l2, a = param->alpha, b = param->beta;
    int num_of_threads = param->nr_threads*1;
    vector<string> lines(num_of_threads);

    auto one_epoch = [&] (shared_ptr<FtrlData> &data, bool update) {
        ifstream fs(data->file_name);
        FtrlFloat loss = 0.0;
        vector<FtrlFloat> ls(num_of_threads);
        fill(ls.begin(), ls.end(), 0);
        string tmp;
        FtrlLong global_i = 0;
        while (getline(fs, tmp)) {
            lines[0] = tmp;
            for(int j=1;j < num_of_threads;j++) {
                getline(fs, lines[j]);
            }
            vector<vector<Node>> nodes(num_of_threads);
            vector<FtrlFloat> rs(num_of_threads);
            vector<FtrlFloat> labels(num_of_threads);
            vector<FtrlInt> ns(num_of_threads);
            fill(rs.begin(), rs.end(), 0);
            fill(ns.begin(), ns.end(), 0);
            fill(labels.begin(), labels.end(), 0);
            FtrlFloat local_loss = 0.0;
#pragma omp parallel for schedule(static) reduction(+: local_loss)
            for(int j=0;j < num_of_threads;j++) {
                string &line = lines[j];
                FtrlFloat &label = labels[j];
                istringstream iss(line);

                ls[j]++;

                iss >> label;
                label = (label>0)? 1:-1;

                FtrlInt idx = 0;
                FtrlFloat val = 0;

                char dummy;
                while (iss >> idx >> dummy >> val) {
                    if (ns[j] < idx+1) {
                        ns[j] = idx+1;
                    }
                    nodes[j].push_back(Node(idx, val));
                    rs[j] += val*val;
                }
                FtrlFloat y=label, wTx=0;
                FtrlFloat r = param->normalized ? rs[j]:1;
                for(Node &x:nodes[j]) {
                    FtrlInt idx = x.idx;
                    FtrlFloat val = x.val*r, zi = z[idx], ni = n[idx];
                    FtrlInt sign=0;
                    if(update) {
                        if (abs(zi) > l1) {
                            if(zi>0) {sign=1;}
                            else if (zi<0) {sign=-1;}
                            w[idx] = (sign*l1-zi) / ((b+sqrt(ni))/a+l2);
                            //w[idx] = -(zi-(2*(zi>0)-1)*l1*f[idx]) / ((b+sqrt(ni))/a+l2*f[idx]);
                        }
                        else {
                            w[idx] = 0;
                        }
                    }
                    wTx += w[idx]*val;
                }
                FtrlFloat exp_m, tmp, weight;
                weight = data->weighted ? data->weight[global_i + j]:1;

                if (wTx*y > 0) {
                    exp_m = exp(-y*wTx);
                    tmp = exp_m/(1+exp_m) * weight / data->sum_weight;
                    local_loss += log(1+exp_m) * weight / data->sum_weight;
                }
                else {
                    exp_m = exp(y*wTx);
                    tmp = 1/(1+exp_m) * weight / data->sum_weight;
                    local_loss += -y*wTx+log(1+exp_m) * weight / data->sum_weight;
                }

                if(update) {
                    FtrlFloat kappa = -y*tmp;
                    for(Node &x:nodes[j]) {
                        FtrlInt idx = x.idx;
                        FtrlFloat val = x.val*r;
                        FtrlFloat g = kappa*val;
                        //FtrlFloat theta = 1/a*(sqrt(n[idx]+g*g)-sqrt(n[idx]));
                        FtrlFloat theta = (sqrt(n[idx]+g*g)-sqrt(n[idx]))/a;
                        z[idx] += g-theta*w[idx];
                        n[idx] += g*g;
                    }
                }
            }
            loss += local_loss;
            global_i += num_of_threads;
        }
        FtrlLong l = 0;
        for(auto ll:ls) {
            l += ll;
        }
        data->l = l;
        return loss / l ;
    };
    tr_loss = one_epoch(data, true);
    if (!test_data->file_name.empty()) {
        va_loss = one_epoch(test_data, false);
    }
    va_auc = 0;
    print_epoch_info();
}

void FtrlData::print_data_info() {
    cout << "Data: " << file_name << "\t";
    cout << "#features: " << n << "\t";
    cout << "#instances: " << l << "\t";
    cout << "#chunks " << nr_chunk << "\t";
    cout << endl;
}

void FtrlProblem::save_model(string model_path) {
    ofstream f(model_path, ios::out | ios::binary);

    FtrlLong nr_feature = w.size();
    f.write(reinterpret_cast<char*>(&param->normalized), sizeof(bool));
    f.write(reinterpret_cast<char*>(&data->n), sizeof(FtrlLong));
    f.write(reinterpret_cast<char*>(w.data()), sizeof(FtrlFloat) * nr_feature);
    f.write(reinterpret_cast<char*>(n.data()), sizeof(FtrlFloat) * nr_feature);
    f.write(reinterpret_cast<char*>(z.data()), sizeof(FtrlFloat) * nr_feature);
}

FtrlLong FtrlProblem::load_model(string model_path) {
    ifstream f(model_path, ios::in | ios::binary);

    bool normalized;
    FtrlLong nr_feature;
    f.read(reinterpret_cast<char*>(&normalized), sizeof(bool));
    f.read(reinterpret_cast<char*>(&nr_feature), sizeof(FtrlLong));
    w.resize(nr_feature);
    n.resize(nr_feature);
    z.resize(nr_feature);
    f.read(reinterpret_cast<char*>(w.data()), sizeof(FtrlFloat) * nr_feature);
    f.read(reinterpret_cast<char*>(n.data()), sizeof(FtrlFloat) * nr_feature);
    f.read(reinterpret_cast<char*>(z.data()), sizeof(FtrlFloat) * nr_feature);

    return nr_feature;
}

void FtrlProblem::save_model_txt(string model_path) {
    ofstream f_out(model_path);
    FtrlLong nr_feature = w.size();
    f_out << "norm " << param->normalized << endl;
    f_out << "n " << nr_feature << endl;

    FtrlFloat *wa = w.data();
    FtrlFloat *na = n.data();
    FtrlFloat *za = z.data();
    char buffer[1024];
    for (FtrlLong j = 0; j < data->n; j++, wa++, na++, za++)
    {
        sprintf(buffer, "w%lld %lf %lf %lf", j, *wa, *na, *za);
        f_out << buffer << endl;
    }
    f_out.close();
}

void FtrlProblem::save_model_updated_txt(string model_path) {
    ofstream f_out(model_path);
    FtrlLong nr_feature = w.size();
    f_out << "norm " << param->normalized << endl;
    f_out << "n " << nr_feature << endl;

    vector<FtrlLong> updated_idx;
    for(auto idx:data->nnz_idx) {
        updated_idx.push_back(idx);
    }
    sort(updated_idx.begin(), updated_idx.end());

    FtrlFloat *wa = w.data();
    FtrlFloat *na = n.data();
    FtrlFloat *za = z.data();
    char buffer[1024];
    for (FtrlLong j:updated_idx)
    {
        sprintf(buffer, "w%lld %lf %lf %lf", j, *(wa+j), *(na+j), *(za+j));
        f_out << buffer << endl;
    }
    f_out.close();
}

FtrlLong FtrlProblem::load_model_txt(string model_path) {

    ifstream f_in(model_path);

    string dummy;
    FtrlLong nr_feature;

    f_in >> dummy >> normalization >> dummy >> nr_feature;
    w.resize(nr_feature);
    z.resize(nr_feature);
    n.resize(nr_feature);
    FtrlFloat *wptr = w.data();
    FtrlFloat *nptr = n.data();
    FtrlFloat *zptr = z.data();

    for(FtrlLong j = 0; j < nr_feature; j++, wptr++, nptr++, zptr++)
    {
        f_in >> dummy;
        f_in >> *wptr >> *nptr >> *zptr;
    }

    return nr_feature;
}

void FtrlProblem::initialize(bool norm, string warm_model_path) {
    f.resize(data->n, 0);
    if(warm_model_path.empty()) {
        feats = data->n;
        w.resize(data->n, 0);
        z.resize(data->n, 0);
        n.resize(data->n, 0);
    }
    else {
        ifstream f_in(warm_model_path);
        string dummy;
        FtrlLong nr_feature = load_model_txt(warm_model_path);
        if(nr_feature >= data->n) {
            feats = nr_feature;
        }
        else {
            feats = data->n;
            w.resize(data->n);
            z.resize(data->n);
            n.resize(data->n);
            FtrlFloat *wptr = &w[nr_feature];
            FtrlFloat *nptr = &n[nr_feature];
            FtrlFloat *zptr = &z[nr_feature];
            for(FtrlLong j = nr_feature; j < data->n; j++, wptr++, nptr++, zptr++)
            {
                *wptr = 0; *nptr = 0; *zptr = 0;
            }
        }
    }
    t = 0;
    tr_loss = 0.0, va_loss = 0.0, fun_val = 0.0, gnorm = 0.0;
    FtrlInt nr_chunk = data->nr_chunk;
    for (FtrlInt chunk_id = 0; chunk_id < nr_chunk; chunk_id++) {

        FtrlChunk chunk = data->chunks[chunk_id];

        chunk.read();

        for (FtrlInt i = 0; i < chunk.l; i++) {

            for (FtrlInt s = chunk.nnzs[i]; s < chunk.nnzs[i+1]; s++) {
                Node x = chunk.nodes[s];
                FtrlInt idx = x.idx;
                f[idx]++;
            }
        }
        if(!param->in_memory)
            chunk.clear();
    }
    for (FtrlInt j = 0; j < data->n; j++) {
        if (param->freq)
            f[j]  = 1/f[j];
        else
            f[j]  = 1;
    }
    start_time = omp_get_wtime();
}

void FtrlProblem::print_header_info() {
    cout.width(4);
    cout << "iter";
    if (param->verbose) {
    cout.width(13);
    cout << "fun_val";
    cout.width(13);
    cout << "reg";
    cout.width(13);
    cout << "|grad|";
    cout.width(13);
    cout << "tr_logloss";
    }
    if(!test_data->file_name.empty()) {
        cout.width(13);
        cout << "va_logloss";
        cout.width(13);
        cout << "va_auc";
    }
    cout.width(13);
    cout << "time";
    cout << endl;
}
void FtrlProblem::print_epoch_info() {
    cout.width(4);
    cout << t+1;
    if (param->verbose) {
        cout.width(13);
        cout << scientific << setprecision(3) << fun_val;
        cout.width(13);
        cout << scientific << setprecision(3) << reg;
        cout.width(13);
        cout << scientific << setprecision(3) << gnorm;
        cout.width(13);
        cout << fixed << setprecision(5) << tr_loss;
    }
    if (!test_data->file_name.empty()) {
        cout.width(13);
        cout << fixed << setprecision(5) << va_loss;
        cout.width(13);
        cout << fixed << setprecision(5) << va_auc;
    }
    cout.width(13);
    cout << fixed << setprecision(5) << omp_get_wtime() - start_time;
    cout << endl;
}

FtrlFloat cal_auc(vector<FtrlFloat> &va_labels, vector<FtrlFloat> &va_scores, vector<FtrlFloat> &va_orders) {
    FtrlFloat auc = 0.0;
    sort(va_orders.begin(), va_orders.end(), [&va_scores] (FtrlInt i, FtrlInt j) {return va_scores[i] < va_scores[j];});

    FtrlFloat prev_score = va_scores[0];
    FtrlLong M = 0, N = 0;
    FtrlLong begin = 0, stuck_pos = 0, stuck_neg = 0;
    FtrlFloat sum_pos_rank = 0;

    FtrlLong l = va_labels.size();
    for (FtrlInt i = 0; i < l; i++)
    {
        FtrlInt sorted_i = va_orders[i];

        FtrlFloat score = va_scores[sorted_i];

        if (score != prev_score)
        {
            sum_pos_rank += stuck_pos*(begin+begin-1+stuck_pos+stuck_neg)*0.5;
            prev_score = score;
            begin = i;
            stuck_neg = 0;
            stuck_pos = 0;
        }

        FtrlFloat label = va_labels[sorted_i];

        if (label > 0)
        {
            M++;
            stuck_pos ++;
        }
        else
        {
            N++;
            stuck_neg ++;
        }
    }
    sum_pos_rank += stuck_pos*(begin+begin-1+stuck_pos+stuck_neg)*0.5;
    auc = (sum_pos_rank - 0.5*M*(M+1)) / (M*N);

    return auc;
}

void FtrlProblem::validate() {
    FtrlInt nr_chunk = test_data->nr_chunk, global_i = 0;
    FtrlFloat local_va_loss = 0.0;
    vector<FtrlFloat> va_labels(test_data->l, 0), va_scores(test_data->l, 0), va_orders(test_data->l, 0);
    for (FtrlInt chunk_id = 0; chunk_id < nr_chunk; chunk_id++) {

        FtrlChunk chunk = test_data->chunks[chunk_id];
        chunk.read();

#pragma omp parallel for schedule(static) reduction(+: local_va_loss)
        for (FtrlInt i = 0; i < chunk.l; i++) {

            FtrlFloat y = chunk.labels[i], wTx = 0;
            FtrlFloat r=param->normalized ? chunk.R[i]:1;

            for (FtrlInt s = chunk.nnzs[i]; s < chunk.nnzs[i+1]; s++) {
                Node x = chunk.nodes[s];
                FtrlInt idx = x.idx;
                if (idx > data->n) {
                    continue;
                }
                FtrlFloat val = x.val*r;
                wTx += w[idx]*val;
            }
            va_scores[global_i+i] = wTx;
            va_orders[global_i+i] = global_i+i;
            va_labels[global_i+i] = y;

            FtrlFloat exp_m, weight;
            if(data->weighted)
                weight = data->weight[global_i + i] / data->sum_weight;
            else
                weight = 1;

            if (wTx*y > 0) {
                exp_m = exp(-y*wTx);
                local_va_loss += log(1+exp_m) * weight;
            }
            else {
                exp_m = exp(y*wTx);
                local_va_loss += -y*wTx+log(1+exp_m) * weight;
            }
        }
        global_i += chunk.l;
        chunk.clear();
    }
    va_loss = local_va_loss / test_data->l;

    if(param->no_auc)
        va_auc = 0;
    else
        va_auc = cal_auc(va_labels, va_scores, va_orders);
    return;
}

void FtrlProblem::fun() {
    FtrlFloat l1 = param->l1, l2 = param->l2;
    vector<FtrlFloat> grad(data->n, 0);
    FtrlInt nr_chunk = data->nr_chunk, global_i = 0;
    fun_val = 0.0, tr_loss = 0.0, gnorm = 0.0, reg = 0.0;
    for (FtrlInt chunk_id = 0; chunk_id < nr_chunk; chunk_id++) {

        FtrlChunk chunk = data->chunks[chunk_id];

        if(!param->in_memory)
            chunk.read();

        FtrlFloat local_tr_loss = 0.0;

#pragma omp parallel for schedule(guided) reduction(+: local_tr_loss)
        for (FtrlInt i = 0; i < chunk.l; i++) {

            FtrlFloat y=chunk.labels[i], wTx=0;
            FtrlFloat r=param->normalized ? chunk.R[i]:1;

            for (FtrlInt s = chunk.nnzs[i]; s < chunk.nnzs[i+1]; s++) {
                Node x = chunk.nodes[s];
                FtrlInt idx = x.idx;
                FtrlFloat val = x.val*r;
                wTx += w[idx]*val;
            }

            FtrlFloat exp_m, tmp, weight;
            if(data->weighted)
                weight = data->weight[global_i + i] / data->sum_weight;
            else
                weight = 1;

            if (wTx*y > 0) {
                exp_m = exp(-y*wTx);
                tmp = exp_m/(1+exp_m) * weight;
                local_tr_loss += log(1+exp_m) * weight;
            }
            else {
                exp_m = exp(y*wTx);
                tmp = 1/(1+exp_m) * weight;
                local_tr_loss += -y*wTx+log(1+exp_m) * weight;
            }

            FtrlFloat kappa = -y*tmp;

            for (FtrlInt s = chunk.nnzs[i]; s < chunk.nnzs[i+1]; s++) {
                Node x = chunk.nodes[s];
                FtrlInt idx = x.idx;
                FtrlFloat val = x.val*r, g = kappa*val+l2*f[idx]*w[idx];
                grad[idx] += g;
            }
        }
        tr_loss += local_tr_loss;
        global_i += chunk.l;
        if(!param->in_memory)
            chunk.clear();
    }
    for (FtrlInt j = 0; j < data->n; j++) {
        gnorm += grad[j]*grad[j];
        reg += (l1*abs(w[j]) + 0.5*l2*w[j]*w[j]);
    }
    fun_val = tr_loss + reg;
    tr_loss /= data->l;
    gnorm = sqrt(gnorm);
}

inline KL FtrlProblem::cal_KL(FtrlFloat y, FtrlFloat wTx, FtrlFloat weight) {
    KL kl;
    FtrlFloat exp_m, tmp;

    if (wTx*y > 0) {
        exp_m = exp(-y*wTx);
        tmp = exp_m/(1+exp_m) * weight;
        kl.local_loss += log(1+exp_m) * weight;
    }
    else {
        exp_m = exp(y*wTx);
        tmp = 1/(1+exp_m) * weight;
        kl.local_loss += -y*wTx+log(1+exp_m) * weight;
    }

    kl.kappa = -y*tmp;

    return kl;
}

inline void FtrlProblem::updateFTRL(FtrlInt idx) {
    FtrlFloat l1 = param->l1, l2 = param->l2, a = param->alpha, b = param->beta;
    FtrlFloat zi = z[idx], ni = n[idx];
    if (abs(zi) > l1*f[idx]) {
        w[idx] = -(zi-(2*(zi>0)-1)*l1*f[idx]) / ((b+sqrt(ni))/a+l2*f[idx]);
    }
    else {
        w[idx] = 0;
    }
}

inline FtrlFloat FtrlProblem::cal_wTx(Node *begin, Node *end, FtrlFloat r, FtrlFloat kappa=0, bool do_update=false, bool ftrl_update=false) {
    FtrlFloat wTx = 0, g_norm = 0;
    FtrlFloat l1 = param->l1, l2 = param->l2, a = param->alpha, b = param->beta;
    for (Node *N = begin; N != end; N++) {
        Node &x = *N;
        FtrlInt idx = x.idx;
        FtrlFloat val = x.val*r, g = 0, g_square = 0;
        if (idx > w.size())
            continue;
        if(do_update) {
            if (param->solver == 0) {
                g = kappa*val+l2*f[idx]*w[idx];
                g_square = g*g;
                n[idx] += g_square;
                w[idx] -= (a/(b+sqrt(n[idx])))*g;
            }
            else if (param->solver == 1) {
                g = kappa*val;
                g_square = g*g;
                FtrlFloat theta = 1/a*(sqrt(n[idx]+g_square)-sqrt(n[idx]));
                n[idx] += g_square;
                z[idx] += g-theta*w[idx];
            }
            else { // RDA
                g = kappa*val;
                z[idx] += g;
                w[idx] = -z[idx] / ((b+sqrt(n[idx]))/a+l2*f[idx]);
                g_square = g*g;
                n[idx] += g_square;
            }
            g_norm += g_square;
        }
        else {
            if(param->solver == 1 && ftrl_update) {
                updateFTRL(idx);
            }
            wTx += w[idx]*val;
        }
    }

    if (do_update)
        return g_norm;
    else
        return wTx;
}

FtrlFloat FtrlProblem::one_epoch(shared_ptr<FtrlData> &data, bool do_update, bool is_train, bool do_auc=false) {
    FtrlInt nr_chunk = data->nr_chunk, global_i = 0;
    vector<FtrlFloat> va_labels, va_scores, va_orders;
    vector<FtrlInt> outer_order(nr_chunk);
    iota(outer_order.begin(), outer_order.end(), 0);
    random_shuffle(outer_order.begin(),outer_order.end());
    FtrlFloat local_loss = 0.0;
    if (do_auc) {
        va_labels.resize(data->l, 0);
        va_scores.resize(data->l, 0);
        va_orders.resize(data->l, 0);
    }
    for (auto chunk_id:outer_order) {
        FtrlChunk chunk = data->chunks[chunk_id];
        if(!param->in_memory)
            chunk.read();
        vector<FtrlInt> inner_oder(chunk.l);
        iota(inner_oder.begin(), inner_oder.end(),0);
        random_shuffle(inner_oder.begin(), inner_oder.end());

#pragma omp parallel for schedule(guided) reduction(+: local_loss)
        for (FtrlInt ii = 0; ii < chunk.l; ii++) {
            FtrlInt i = inner_oder[ii];
            FtrlFloat y=chunk.labels[i];
            FtrlFloat r=param->normalized ? chunk.R[i]:1;

            Node *begin = &chunk.nodes[chunk.nnzs[i]], *end = &chunk.nodes[chunk.nnzs[i+1]];
            FtrlFloat wTx = cal_wTx(begin, end, r, 0, false, is_train);
            if (do_auc) {
                va_scores[global_i+i] = wTx;
                va_orders[global_i+i] = global_i+i;
                va_labels[global_i+i] = y;
            }

            FtrlFloat exp_m, tmp, weight;
            if(data->weighted)
                weight = data->weight[global_i + i] / data->sum_weight;
            else
                weight = 1;

            KL kl = cal_KL(y, wTx, weight);
            FtrlFloat kappa = kl.kappa;
            local_loss += kl.local_loss;

            FtrlFloat g_norm;
            if(do_update)
               g_norm = cal_wTx(begin, end, r, kappa, true, false);
        }
        global_i += chunk.l;
        if(!param->in_memory)
            chunk.clear();
    }

    FtrlFloat loss = local_loss / data->l;
    if(do_auc)
        va_auc = cal_auc(va_labels, va_scores, va_orders);
    else
        va_auc = 0;

    return loss;
}

void FtrlProblem::solve() {
    print_header_info();
    FtrlInt nr_chunk = data->nr_chunk, global_i = 0;
    FtrlFloat l1 = param->l1, l2 = param->l2, a = param->alpha, b = param->beta;
    FtrlFloat best_va_loss = numeric_limits<FtrlFloat>::max();
    vector<FtrlFloat> prev_w(data->n, 0);
    vector<FtrlFloat> prev_n(data->n, 0);
    vector<FtrlFloat> prev_z(data->n, 0);

    for (t = 0; t < param->nr_pass; t++) {
        cout << "train:" << endl;
        tr_loss = one_epoch(data, true, true);

        if (!test_data->file_name.empty()) {
            cout << "validate:" << endl;
            va_loss = one_epoch(test_data, false, false, !param->no_auc);
        }

        print_epoch_info();
        if(param->auto_stop) {
            if(va_loss > best_va_loss){
                memcpy(w.data(), prev_w.data(), data->n * sizeof(FtrlFloat));
                memcpy(n.data(), prev_n.data(), data->n * sizeof(FtrlFloat));
                memcpy(z.data(), prev_z.data(), data->n * sizeof(FtrlFloat));
                cout << "Auto-stop. Use model at" << t <<"th iteration."<<endl;
                break;
            }else{
                memcpy(prev_w.data(), w.data(), data->n * sizeof(FtrlFloat));
                memcpy(prev_n.data(), n.data(), data->n * sizeof(FtrlFloat));
                memcpy(prev_z.data(), z.data(), data->n * sizeof(FtrlFloat));
                best_va_loss = va_loss;
            }
        }
    }
}

void causE(FtrlProblem &prob_sc, FtrlProblem &prob_st) {
    FtrlFloat best_va_loss = numeric_limits<FtrlFloat>::max();
    vector<FtrlFloat> prev_w_st(prob_st.data->n, 0);
    vector<FtrlFloat> prev_n_st(prob_st.data->n, 0);
    vector<FtrlFloat> prev_z_st(prob_st.data->n, 0);
    vector<FtrlFloat> prev_w_sc(prob_sc.data->n, 0);
    vector<FtrlFloat> prev_n_sc(prob_sc.data->n, 0);
    vector<FtrlFloat> prev_z_sc(prob_sc.data->n, 0);

    prob_st.print_header_info();
    for (FtrlInt t = 0; t < prob_st.param->nr_pass; t++, prob_st.t++, prob_sc.t++) {
        prob_st.tr_loss = CausE_one_epoch(prob_st, prob_sc, prob_st.data, true);
        if (!prob_st.test_data->file_name.empty()) {
            prob_st.va_loss = CausE_one_epoch(prob_st, prob_sc, prob_st.test_data, false);
        }
        prob_sc.tr_loss = CausE_one_epoch(prob_sc, prob_st, prob_sc.data, true);
        if (!prob_sc.test_data->file_name.empty()) {
            prob_sc.va_loss = CausE_one_epoch(prob_sc, prob_st, prob_sc.test_data, false);
        }
        prob_st.print_epoch_info();
        prob_sc.print_epoch_info();

        if(prob_sc.param->auto_stop) {
            if(prob_sc.va_loss > best_va_loss){
                memcpy(prob_sc.w.data(), prev_w_sc.data(), prob_sc.data->n * sizeof(FtrlFloat));
                memcpy(prob_sc.n.data(), prev_n_sc.data(), prob_sc.data->n * sizeof(FtrlFloat));
                memcpy(prob_sc.z.data(), prev_z_sc.data(), prob_sc.data->n * sizeof(FtrlFloat));
                memcpy(prob_st.w.data(), prev_w_st.data(), prob_st.data->n * sizeof(FtrlFloat));
                memcpy(prob_st.n.data(), prev_n_st.data(), prob_st.data->n * sizeof(FtrlFloat));
                memcpy(prob_st.z.data(), prev_z_st.data(), prob_st.data->n * sizeof(FtrlFloat));
                cout << "Auto-stop. Use model at" << t <<"th iteration."<<endl;
                break;
            }else{
                memcpy(prev_w_sc.data(), prob_sc.w.data(), prob_sc.data->n * sizeof(FtrlFloat));
                memcpy(prev_n_sc.data(), prob_sc.n.data(), prob_sc.data->n * sizeof(FtrlFloat));
                memcpy(prev_z_sc.data(), prob_sc.z.data(), prob_sc.data->n * sizeof(FtrlFloat));
                memcpy(prev_w_st.data(), prob_st.w.data(), prob_st.data->n * sizeof(FtrlFloat));
                memcpy(prev_n_st.data(), prob_st.n.data(), prob_st.data->n * sizeof(FtrlFloat));
                memcpy(prev_z_st.data(), prob_st.z.data(), prob_st.data->n * sizeof(FtrlFloat));
                best_va_loss = prob_sc.va_loss;
            }
        }
    }
}

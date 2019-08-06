#include "tools.h"

map<FtrlLong, string> get_pos_featmap(string file_name) {
    string line;
    ifstream fs(file_name);
    map<FtrlLong, string> pos_featmap;

    while(getline(fs, line)) {
        istringstream iss(line);
        string key;
        FtrlLong value;
        iss >> key >> value;
        if(key.find("Position") != string::npos)
            pos_featmap[value] = key;
    }

    return pos_featmap;
}

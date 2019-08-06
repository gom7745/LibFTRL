#ifndef TOOLS_H
#define TOOLS_H

#include <map>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include "ftrl.h"

using namespace std;

map<FtrlLong, string> get_pos_featmap(string file_name);

#endif

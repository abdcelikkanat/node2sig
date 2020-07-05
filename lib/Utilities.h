#ifndef UTILITIES_H
#define UTILITIES_H
#include <string>
#include <sstream>
#include <vector>
#include <iostream>


namespace Constants
{
    const std::string ProgramName = "nodesig";
};

using namespace std;



int parse_arguments(int argc, char** argv, string &edgeFile, string &embFile, unsigned int &walkLen,
                    unsigned int &dimension, float &contProb, int &featureBlockSize, int &weightBlockSize, int &nodesBlockSize,
                    bool &verbose) {

    vector <string> parameter_names{"--help",
                                    "--edgefile", "--embfile", "--walklen", "--dim",
                                    "--prob", "--featureBlockSize", "--weightBlockSize", "--nodesBlockSize",
                                    "--verbose"
    };

    string arg_name;
    stringstream help_msg, help_msg_required, help_msg_opt;

    // Set the help message
    help_msg_required << "\nUsage: ./" << Constants::ProgramName;
    help_msg_required << " " << parameter_names[1] << " EDGE_FILE "
                      << parameter_names[2] << " EMB_FILE "
                      << parameter_names[3] << " WALK_LENGTH "<< "\n";

    help_msg_opt << "\nOptional parameters:\n";
    help_msg_opt << "\t[ " << parameter_names[4] << " (Default: " << dimension << ") ]\n";
    help_msg_opt << "\t[ " << parameter_names[5] << " (Default: " << contProb << ") ]\n";
    help_msg_opt << "\t[ " << parameter_names[6] << " (Default: " << featureBlockSize <<") ]\n";
    help_msg_opt << "\t[ " << parameter_names[7] << " (Default: " << weightBlockSize << ") ]\n";
    help_msg_opt << "\t[ " << parameter_names[8] << " (Default: " << nodesBlockSize << ") ]\n";
    help_msg_opt << "\t[ " << parameter_names[9] << " (Default: " << verbose << ") ]\n";
    help_msg_opt << "\t[ " << parameter_names[0] << ", -h ] Shows this message";

    help_msg << "" << help_msg_required.str() << help_msg_opt.str();

    // Read the argument values
    for(int i=1; i<argc; i=i+2) {

        arg_name.assign(argv[i]);

        if (arg_name.compare(parameter_names[1]) == 0) {
            edgeFile = argv[i + 1];
        } else if (arg_name.compare(parameter_names[2]) == 0) {
            embFile = argv[i + 1];
        } else if (arg_name.compare(parameter_names[3]) == 0) {
            walkLen = stoi(argv[i + 1]);
        } else if (arg_name.compare(parameter_names[4]) == 0) {
            dimension = stoi(argv[i + 1]);
        } else if (arg_name.compare(parameter_names[5]) == 0) {
            contProb = stod(argv[i + 1]);
        } else if (arg_name.compare(parameter_names[6]) == 0) {
            featureBlockSize = stoi(argv[i + 1]);
        } else if (arg_name.compare(parameter_names[7]) == 0) {
            weightBlockSize = stoi(argv[i + 1]);
        } else if (arg_name.compare(parameter_names[8]) == 0) {
            nodesBlockSize = stoi(argv[i + 1]);
        } else if (arg_name.compare(parameter_names[9]) == 0) {
            verbose = stoi(argv[i + 1]);
        } else if (arg_name.compare(parameter_names[0]) == 0 or arg_name.compare("-h") == 0) {
            cout << help_msg.str() << endl;
            return 1;
        } else {
            cout << "Invalid argument name: " << arg_name << endl;
            return -2;
        }
        arg_name.clear();

    }

    // Check if the required parameters were set or not
    if(edgeFile.empty() || embFile.empty() || walkLen == 0) {
        cout << "Please enter the required parameters: ";
        cout << help_msg_required.str() << endl;

        return -4;
    }

    // Check if the constraints are satisfied
    if( contProb < 0 ||  contProb > 1) {
        cout << "The probability should be between 0 and 1." << endl;
        return -5;
    }

    return 0;

}

#endif //UTILITIES_H

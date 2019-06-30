/*  This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
    */

/*
 * File:   data.cpp
 *
 * Reader for csv files for OKS / PTA
 * Author: Giovanni Mahlknecht Free University of Bozen/Bolzano
 *         giovanni.mahlknecht@inf.unibz.it
 *
 * Created on November 19, 2015, 3:09 PM
 */


#include <string>
#include <sstream>
#include <fstream>
#include "data.h"
#include <cstring>
using namespace std;

double* readCSV(std::string filename, long tuplesToRead, int column) {
	double *tuples = new double[tuplesToRead + 1];
	tuples[0] = 0.0;
	ifstream conffile(filename);
	if (conffile.is_open()) {
		string line;
		long num = 0;
		while (std::getline(conffile, line) && num < tuplesToRead) {
			std::istringstream is_line(line);
			std::string value;
			for (int i = 0; i <= column; i++) {
				//skip until correct column
				if (!std::getline(is_line, value, ',')) {
					fprintf(stderr, "Error, not enough columns\n");
					exit(-1);
				}
			}
			tuples[num + 1] = std::stod(value);
			num++;
		}
		if (num < tuplesToRead) {
			fprintf(stderr, "Error, not enough rows\n");
			exit(-1);
		}
	}

	return (tuples);
}

itatuple* readCSV(std::string filename, long tuplesToRead) {
	itatuple *tuples = new itatuple[tuplesToRead + 1];
	tuples[0].te = 0;
	tuples[0].ts = 0;
	tuples[0].value = 0;
	ifstream conffile(filename);
	if (conffile.is_open()) {
		string line;
		long num = 0;
		while (std::getline(conffile, line) && num < tuplesToRead) {
			std::istringstream is_line(line);
			std::string value;
			if (std::getline(is_line, value, ',')) {
				tuples[num + 1].group = value;
			} else {
				fprintf(stderr, "Error, not enough columns\n");
				exit(-1);
			}
			if (std::getline(is_line, value, ',')) {
				tuples[num + 1].value = std::stod(value);
			} else {
				fprintf(stderr, "Error, not enough columns\n");
				exit(-1);
			}
			if (std::getline(is_line, value, ',')) {
				tuples[num + 1].ts = std::stol(value);
			} else {
				fprintf(stderr, "Error, not enough columns\n");
				exit(-1);
			}
			if (std::getline(is_line, value, ',')) {
				tuples[num + 1].te = std::stol(value);
			} else {
				fprintf(stderr, "Error, not enough columns\n");
				exit(-1);
			}
			num++;
		}
		if (num < tuplesToRead) {
			fprintf(stderr, "Error, not enough rows\n");
			exit(-1);
		}
	}
	return (tuples);
}


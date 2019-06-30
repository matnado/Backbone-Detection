struct itatuple {
	std::string group;
	double value;
	long ts;
	long te;
};

double* readCSV(std::string, long, int);
itatuple* readCSV(std::string, long);

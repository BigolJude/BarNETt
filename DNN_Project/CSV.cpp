#include <fstream>
#include <iostream>
#include <sstream>
#include <list>
#include "CSV.h"
using namespace std;

list<list<float>> CSV::read(string file)
{
	list<list<float>> values;
	string line, column;
	ifstream fileStream(file);

	if (!fileStream.is_open())
	{
		cout << "something went wrong";
	}

	cout << "the column names" << endl;
	if (fileStream.good())
	{
		getline(fileStream, line);
		stringstream stream(line);
		string val;
		while (stream >> val)
		{
			cout << "column: " << val << ", ";
		}
		cout << endl;
	}

	cout << "the values name" << endl;
	while (getline(fileStream, line))
	{
		list<float> newLine;
		size_t index = 0;
		string subString;
		string comma = ",";
		while ((index = line.find(comma)) != string::npos)
		{
			subString = line.substr(0, index);
			newLine.push_back(stof(subString));
			line.erase(0, index + comma.length());
		}
		values.push_back(newLine);
	}
	return values;
}
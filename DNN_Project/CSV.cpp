#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>
#ifdef _DEBUG
#ifndef DBG_NEW
#define DBG_NEW new ( _NORMAL_BLOCK , __FILE__ , __LINE__ )
#define new DBG_NEW
#endif
#endif  // _DEBUG

#include <fstream>
#include <iostream>
#include <sstream>
#include <list>
#include "CSV.h"
using namespace std;

list<list<double>> CSV::read(string file)
{
	list<list<double>> values;
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
		list<double> newLine;
		size_t index = 0;
		string subString;
		string comma = ",";

		while ((index = line.find(comma)) != string::npos)
		{
			subString = line.substr(0, index);

			if (newLine.size() <= 3)
			{
				newLine.push_back(stod(subString));
			}

			line.erase(0, index + comma.length());
		}
		if (line == "Iris-setosa")
		{
			newLine.push_back(1);
		}
		else if (line == "Iris-versicolor")
		{
			newLine.push_back(2);
		}
		else if (line == "Iris-virginica")
		{
			newLine.push_back(3);
		}
		values.push_back(newLine);
	}
	return values;
}
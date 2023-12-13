#include <iostream>
#include <list>
#include <fstream>
#include <sstream>
#include <string>
#include <utility>
#include <random>
#include <vector>
#include "layer.h"
#include "loss.h"
#include "Initialisation.h"
#include "Activation.h"
#include "Network.h"
#include "CSV.h"
#include "ExampleNetworks.h"
using namespace std;

int main()
{
	list<list<double>> values = CSV::read("..\\DNN_Project\\Dataset\\iris.csv");
	
	vector<list<double>> valuesVector(values.begin(), values.end());
	random_device randomDevice;
	mt19937 generator(randomDevice());
	shuffle(valuesVector.begin(), valuesVector.end(), generator);
	
	Network* irisNetwork = ExampleNetworks::irisDatasetNetwork();
	
	for(int epochs = 0; epochs < 40; ++epochs)
	{
		double averageLoss = 0;		
		vector<list<double>>::iterator valuesIt = valuesVector.begin();
		for (int i = 0; i < values.size(); ++i)
		{
			list<double> inputs = *valuesIt;
			list<double>::iterator inputsIt = inputs.end();
			advance(inputsIt, -1);
	
			list<double> expected;
	
			if (*inputsIt == 1)
			{
				expected = { 1,0,0 };
			}
			else if (*inputsIt == 2)
			{
				expected = { 0, 1, 0 };
			}
			else if (*inputsIt == 3)
			{
				expected = { 0, 0, 1 };
			}
	
			inputs.pop_back();
	
			irisNetwork->train(inputs, 0.0001, expected);
	
			averageLoss = averageLoss + irisNetwork->getError();
	
			inputs.clear();
			expected.clear();
	
			advance(valuesIt, 1);
		}
		cout << averageLoss / values.size() << endl;
	}
}
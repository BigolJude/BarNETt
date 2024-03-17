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

// Memory leak debugging
#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>
#ifdef _DEBUG
#ifndef DBG_NEW
#define DBG_NEW new ( _NORMAL_BLOCK , __FILE__ , __LINE__ )
#define new DBG_NEW
#endif
#endif  // _DEBUG

int main()
{	
	Layer layer1 = Layer(2, 4, "relu");
	Layer layer2 = Layer(4, 10, "relu");
	Layer layer3 = Layer(10, 2, "relu");
	
	Network* network = new Network();
	
	network->addLayer(layer1);
	network->addLayer(layer2);
	network->addLayer(layer3);

	float error = 1;
	bool mode = true;
	float learningRate = 0.01;

	while (error > 0.2)
	{
		float oldError = error;
		float tempError = 0;

		network->train({ 0.0 ,0.5 }, learningRate, { 1, 0 });
		tempError = network->getError();
		network->train({ 0.5 ,0.0 }, learningRate, { 0, 1 });
		error = (tempError + network->getError()) / 2;

		float errorDifference = error - oldError;

		if (errorDifference > 0.01)
		{
			learningRate = learningRate - 0.001;
		}
		else if (errorDifference == 0)
		{
			learningRate = learningRate + 0.001;
		}

		mode = !mode;
		std::cout << error << endl;
	}

	
	network->predict({ 0.5, 0.0 });
	list<double> predictions = network->getPrediction();

	network->predict({ 0.0, 0.5 });
	predictions = network->getPrediction();

	delete(network);

	_CrtSetReportMode(_CRT_WARN, _CRTDBG_MODE_DEBUG);
	_CrtDumpMemoryLeaks();
}
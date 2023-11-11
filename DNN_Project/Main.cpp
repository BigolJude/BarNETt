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
using namespace std;

int main()
{
	//list<list<double>> values = CSV::read("C:\\iris.csv");
	//
	//vector<list<double>> valuesVector(values.begin(), values.end());
	//random_device randomDevice;
	//mt19937 generator(randomDevice());
	//shuffle(valuesVector.begin(), valuesVector.end(), generator);
	//
	//cout << "finished" << endl;
	//
	//Layer* layer1 = new Layer(1, 1, "relu");
	//Layer* layer2 = new Layer(1, 1, "relu");
	//
	//Network* network = new Network();
	//
	//network->addLayer(*layer1);
	//network->addLayer(*layer2);
	////network->addLayer(*layer3);
	////network->addLayer(*layer4);
	//
	//for(int epochs = 0; epochs < 10; ++epochs)
	//{
	//	vector<list<double>>::iterator valuesIt = valuesVector.begin();
	//	for (int i = 0; i < values.size(); ++i)
	//	{
	//		list<double> inputs = *valuesIt;
	//		list<double>::iterator inputsIt = inputs.end();
	//		advance(inputsIt, -1);
	//
	//		list<double> expected;
	//
	//		if (*inputsIt == 1)
	//		{
	//			expected = { 1,0,0 };
	//		}
	//		else if (*inputsIt == 2)
	//		{
	//			expected = { 0, 1, 0 };
	//		}
	//		else if (*inputsIt == 3)
	//		{
	//			expected = { 0, 0, 1 };
	//		}
	//
	//		inputs.pop_back();
	//
	//		network->train(inputs, 0.001, expected);
	//
	//		cout << i << "." << "Error: " << network->getError() << endl;
	//		cout << "-------" << endl;
	//
	//		inputs.clear();
	//		expected.clear();
	//
	//		advance(valuesIt, 1);
	//	}
	//}
	
	Neuron* neuron1 = new Neuron({ 0.3, 0.5 });
	Neuron* neuron2 = new Neuron({ 0.5, 0.5 });
	list<Neuron*> neurons1 = { neuron1}; 
	list<Neuron*> neurons2 = { neuron2, neuron2 };
	Layer* layer = new Layer(neurons1, "relu");
	Layer* layer2 = new Layer(neurons2, "relu");
	Network* network = new Network();
	network->addLayer(layer);
	network->addLayer(layer2);

	network->train({ 10 }, 0.5, { 1, 0 });
	network->train({ 1 }, 0.5, { 0, 1 });
	network->train({ 10 }, 0.5, { 1, 0 });

	float crossEntropyTests = Loss::crossEntropy({ 0.775,0.116,00.39,0.07 }, { 1,0,0,0 });
	cout << "loss: " << crossEntropyTests << endl;
	cout << "finished!" << endl;
	
	delete(neuron1);
	delete(neuron2);
	delete(layer);
	delete(layer2);
	//delete(layer3);
	//delete(layer4);
	delete(network);
}
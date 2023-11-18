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
	list<list<double>> values = CSV::read("C:\\iris.csv");
	
	vector<list<double>> valuesVector(values.begin(), values.end());
	random_device randomDevice;
	mt19937 generator(randomDevice());
	shuffle(valuesVector.begin(), valuesVector.end(), generator);
	
	cout << "finished" << endl;
	
	Layer* layer1 = new Layer(4, 4, "relu");
	Layer* layer2 = new Layer(4, 10, "relu");
	Layer* layer3 = new Layer(10, 20, "relu");
	Layer* layer4 = new Layer(20, 10, "relu");
	Layer* layer5 = new Layer(10, 3, "relu");
	
	Network* network = new Network();
	
	network->addLayer(*layer1);
	network->addLayer(*layer2);
	network->addLayer(*layer3);
	network->addLayer(*layer4);
	network->addLayer(*layer5);
	
	for(int epochs = 0; epochs < 10; ++epochs)
	{
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
	
			network->train(inputs, 1, expected);
	
			cout << i << "." << "Error: " << network->getError() << endl;
			cout << "-------" << endl;
	
			inputs.clear();
			expected.clear();
	
			advance(valuesIt, 1);
		}
	}
	//list<double> weights1 = { 0.15, 0.5 };
	//list<double> weights2 = { 0.25, 0.5 };
	//
	//Neuron* neuron1 = new Neuron(weights1);
	//Neuron* neuron2 = new Neuron(weights1);
	//Neuron* neuron3 = new Neuron(weights2);
	//Neuron* neuron4 = new Neuron(weights2);
	//list<Neuron*> neurons1 = { neuron1, neuron2 }; 
	//list<Neuron*> neurons2 = { neuron3, neuron4 };
	//
	//Layer* layer1 = new Layer(neurons1, 0.5, "relu");
	//Layer* layer2 = new Layer(neurons2, 0.5, "relu");
	//
	//Network* network = new Network();
	//network->addLayer(*layer1);
	//network->addLayer(*layer2);
	//
	//double averageError = 1;
	//while(averageError > 0.2)
	//{
	//	network->train({ 5, 5 }, 0.05, { 1, 0 });
	//	averageError = network->getError();
	//	network->train({ 10, 10 }, 0.05, { 0, 1 });
	//	averageError = (averageError + network->getError()) / 2;
	//	cout << "Loss: " << averageError << endl;
	//}
	//network->predict({ 10, 10 });
	//
	//list<double> predictions = Activation::SoftMax(network->getPrediction());
	//list<double>::iterator predictionsIt = predictions.begin();
	//
	//for (int i = 0; i < predictions.size(); ++i)
	//{
	//	cout << i << "." << *predictionsIt << endl;
	//	advance(predictionsIt, 1);
	//}
	//
	//network->predict({ 5, 5 });
	//
	//predictions = Activation::SoftMax(network->getPrediction());
	//predictionsIt = predictions.begin();
	//
	//for (int i = 0; i < predictions.size(); ++i)
	//{
	//	cout << i << "." << *predictionsIt << endl;
	//	advance(predictionsIt, 1);
	//}
	//
	//float crossEntropyTests = Loss::crossEntropy({ 0.775,0.116,00.39,0.07 }, { 1,0,0,0 });
	//cout << "loss: " << crossEntropyTests << endl;
	//cout << "finished!" << endl;
	
	delete(layer1);
	delete(layer2);
	//delete(layer3);
	//delete(layer4);
	delete(network);
}
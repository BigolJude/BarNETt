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
	//list<list<double>> values = CSV::read("D:\\iris.csv");
	//
	//vector<list<double>> valuesVector(values.begin(), values.end());
	//random_device randomDevice;
	//mt19937 generator(randomDevice());
	//shuffle(valuesVector.begin(), valuesVector.end(), generator);
	//
	//cout << "finished" << endl;
	//
	//Layer* layer1 = new Layer(4, 4, 0.5, "relu");
	//Layer* layer2 = new Layer(4, 3, 0.5, "relu");
	//
	//Network* network = new Network();
	//
	//network->addLayer(*layer1);
	//network->addLayer(*layer2);
	//
	//for(int epochs = 0; epochs < 1000; ++epochs)
	//{
	//	double averageLoss = 0;		
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
	//		network->train(inputs, 0.0001, expected);
	//
	//		averageLoss = averageLoss + network->getError();
	//
	//		inputs.clear();
	//		expected.clear();
	//
	//		advance(valuesIt, 1);
	//		//cout << "----" << endl;
	//	}
	//	cout << averageLoss / values.size() << endl;
	//}

	Neuron* neuron1 = new Neuron({ 0.1, 0.2 });
	Neuron* neuron2 = new Neuron({ 0.3, 0.4 });
	Neuron* neuron3 = new Neuron({ 0.5, 0.6 });
	Neuron* neuron4 = new Neuron({ 0.4, 0.4 });
	Neuron* neuron7 = new Neuron({ 0.4, 0.5 });
	Neuron* neuron5 = new Neuron({ 0.1, 0.2, 0.4 });
	Neuron* neuron6 = new Neuron({ 0.4, 0.5, 0.3 });

	//Neuron* neuron6 = new Neuron({ 0.25, 0.1 });
	
	
	list<Neuron*> neurons1 = { neuron1, neuron2 }; 
	list<Neuron*> neurons2 = { neuron3, neuron4, neuron7 };
	list<Neuron*> neurons3 = { neuron5, neuron6 };

	Layer* layer1 = new Layer(neurons1, 0.5, "relu");
	Layer* layer2 = new Layer(neurons2, 0.5, "relu");
	Layer* layer3 = new Layer(neurons3, 0.5, "relu");
	
	Network* network = new Network();
	network->addLayer(*layer1);
	network->addLayer(*layer2);
	network->addLayer(*layer3);
	
	double averageError = 1;
	double learningRate = 1;
	while(averageError > 0.30 || isnan(averageError))
	{
        if (isnan(averageError))
		{
			neuron1 = new Neuron({ 0.1, 0.2 });
			neuron2 = new Neuron({ 0.3, 0.4 });
			neuron3 = new Neuron({ 0.5, 0.6 });
			neuron4 = new Neuron({ 0.4, 0.4 });
			neuron7 = new Neuron({ 0.4, 0.5 });
			neuron5 = new Neuron({ 0.1, 0.2, 0.3 });
			neuron6 = new Neuron({ 0.4, 0.5, 0.4 });


			neurons1 = { neuron1, neuron2 };
			neurons2 = { neuron3, neuron4, neuron7 };
			neurons3 = { neuron5, neuron6 };

			layer1 = new Layer(neurons1, 0.5, "relu");
			layer2 = new Layer(neurons2, 0.5, "relu");
			layer3 = new Layer(neurons3, 0.5, "relu");

			network = new Network();
			network->addLayer(*layer1);
			network->addLayer(*layer2);
			network->addLayer(*layer3);
			learningRate -= 0.04;

			cout << "learningRate: " << learningRate << endl;
		}

		averageError = 0;
		network->train({ 5, 1 }, learningRate, { 0, 1 });
		averageError += network->getError();

		network->train({ 1, 5 }, learningRate, { 1, 0 });
		averageError += network->getError();

		network->train({ 5, 1 }, learningRate, { 0, 1 });		
		averageError += network->getError();

		network->train({ 1, 5 }, learningRate, { 1, 0 });
		averageError += network->getError();

		averageError = averageError / 4;
		cout << "Loss: " << averageError << endl;
	}
	network->predict({ 5, 1 });
	
	list<double> predictions = Activation::SoftMax(network->getPrediction());
	list<double>::iterator predictionsIt = predictions.begin();
	
	float crossEntropyTests = Loss::crossEntropy({ 0.775,0.116,00.39,0.07 }, { 1,0,0,0 });
	cout << "loss: " << crossEntropyTests << endl;
	cout << "finished!" << endl;
	
	delete(layer1);
	delete(layer2);

	delete(network);
}
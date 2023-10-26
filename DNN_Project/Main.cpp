#include <iostream>
#include <list>
#include "layer.h"
#include "Initialisation.h"
#include "Activation.h"
#include "Network.h"


int main()
{
	std::cout << "Hello World" << endl;
	list<float> inputs1{ 8.4, 7.3, 192, 0.55 };
	list<float> inputs2{ 8.4, 8.0, 190, 0.49 };
	list<float> inputs3{ 6.2, 5.1, 84,  0.60 };
	list<float> inputs4{ 6.1, 5.5, 85,  0.57 };
	list<float> inputs5{ 9.0, 9.5, 83,	0.59 };
	list<float> inputs6{ 8.5, 4.2, 188, 0.46 };
	list<float> inputs7{ 7.9, 7.1, 182, 0.56 };
	list<float> inputs8{ 7.4, 5.2, 79,  0.44 };

	Layer* layer1 = new Layer(4, 4, "relu");
	Layer* layer2 = new Layer(4, 10, "relu");
	Layer* layer3 = new Layer(10, 5, "relu");
	Layer* layer4 = new Layer(5, 2, "softmax");

	Network* network = new Network();

	network->addLayer(*layer1);
	network->addLayer(*layer2);
	network->addLayer(*layer3);
	network->addLayer(*layer4);

	float learningRate = 0.0001;
	float expected = 1;
	
	network->predict(inputs8);

	list<float> prediction = Activation::SoftMax(network->getPrediction());

	cout << "finished!" << endl;

	delete(layer1);
	delete(layer2);
	delete(layer3);
	delete(layer4);
	inputs1.clear();
	inputs2.clear();
}
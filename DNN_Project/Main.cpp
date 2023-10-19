#include <iostream>
#include <list>
#include "Neuron.h"
#include "Initialisation.h"
#include "Activation.h"


int main()
{
	std::cout << "Hello World" << endl;
	list<float> inputs1{ 0.1, 0.1, 1 };
	list<float> inputs2{ 0.5, 0.5, 1 };
	
	Neuron* neuron = new Neuron(inputs1);

	neuron->train(0.0001, 0);

	cout << "here" << endl;

	cout << "weight: " << neuron->getWeight() << endl;

	delete(neuron);
	inputs1.clear();
	inputs2.clear();
}

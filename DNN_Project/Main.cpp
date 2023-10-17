#include <iostream>
#include <list>
#include "Neuron.h"
#include "Initialisation.h"
#include "Activation.h"


int main()
{
	std::cout << "Hello World" << endl;
	list<float> inputs {1.0, 4.0};
	list<float> weights{ -0.5, 0.2 };
	
	Neuron* neuron = new Neuron(inputs, weights);

	neuron->weigh();
	cout << neuron->getWeight() << endl;

	float activation = Activation::ReLu(neuron->getWeight());

	cout << activation << endl;

	//list<float> reluValues = Activation::SoftMax(weights);

	delete(neuron);
	inputs.clear();
	weights.clear();

}
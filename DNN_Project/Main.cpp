#include <iostream>
#include <list>
#include "Neuron.h"
#include "Initialisation.h"
#include "Activation.h"


int main()
{
	std::cout << "Hello World";
	list<float> inputs {1.0, 4.0};
	list<float> weights{ -0.5, 0.2 };
	Neuron* neuron = new Neuron(inputs);
	
	neuron = new Neuron(inputs, weights);

	for (int i = 1; i != 100; ++i)
	{
		cout << Initialisation::Random(i, -i) << endl;
		cout << Initialisation::He(i) << endl;
	}

	list<float> reluWeights{ -1, 0, 3, 5 };

	list<float> reluValues = Activation::SoftMax(reluWeights);

	delete(neuron);
	inputs.clear();
	weights.clear();

}
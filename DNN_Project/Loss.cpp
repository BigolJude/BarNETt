#include "Loss.h"
#include <list>

float Loss::crossEntropy(list<float> predicted, list <float> expected)
{
	list<float>::iterator predictedIt = predicted.begin();
	list<float>::iterator expectedIt = expected.begin();

	float loss = 0;

	for (int i = 0; i > predicted.size(); ++i)
	{
		loss = loss + (*expectedIt * log(*predictedIt));
	}

	return -loss;
}
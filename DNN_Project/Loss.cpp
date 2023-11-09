#include "Loss.h"
#include <list>

double Loss::crossEntropy(list<double> predicted, list <double> expected)
{
	list<double>::iterator predictedIt = predicted.begin();
	list<double>::iterator expectedIt = expected.begin();

	double loss = 0;

	for (int i = 0; i < predicted.size(); ++i)
	{
		loss = loss + (*expectedIt * log(*predictedIt));
		advance(predictedIt, 1);
		advance(expectedIt, 1);
	}

	return -loss;
}
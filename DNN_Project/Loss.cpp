#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>
#ifdef _DEBUG
#ifndef DBG_NEW
#define DBG_NEW new ( _NORMAL_BLOCK , __FILE__ , __LINE__ )
#define new DBG_NEW
#endif
#endif  // _DEBUG

#include "Loss.h"
#include <list>

/// <summary>
/// Standard cross-entropy loss.
/// </summary>
/// <param name="predicted"></param>
/// <param name="expected"></param>
/// <returns>The loss of the predicted outputs in regards to the expected.</returns>
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
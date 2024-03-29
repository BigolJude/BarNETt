#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>
#ifdef _DEBUG
#ifndef DBG_NEW
#define DBG_NEW new ( _NORMAL_BLOCK , __FILE__ , __LINE__ )
#define new DBG_NEW
#endif
#endif  // _DEBUG

#include <math.h>
#include "pch.h"
#include "CppUnitTest.h"
#include "..\DNN_Project\Initialisation.h"
#include "..\DNN_Project\Activation.h"
#include "..\DNN_Project\Loss.h"
#include "..\DNN_Project\Neuron.h"
#include "..\DNN_Project\Layer.h"
#include "..\DNN_Project\Network.h"
using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace DNNProjectTests
{
	TEST_CLASS(DNNProjectTests)
	{
	public:
		#pragma region Initialisation
		// Values found in https://machinelearningmastery.com/weight-initialization-for-deep-learning-neural-networks/
		TEST_METHOD(Initialisation_He_Valid)
		{
			Assert::AreEqual(Initialisation::He(4.0), sqrt(2.0/4.0));
		}
		#pragma endregion

		#pragma region Activation
		TEST_METHOD(Activation_ReLu_AboveZero)
		{
			Assert::AreEqual(Activation::ReLu(1.0), 1.0);
		}

		TEST_METHOD(Activation_ReLu_BelowZero)
		{
			Assert::AreEqual(Activation::ReLu(-1.0), 0.0);
		}

		TEST_METHOD(Activation_LReLu_AboveZero)
		{
			Assert::AreEqual(Activation::LReLu(1.0), 1.0);
		}

		TEST_METHOD(Activation_LReLu_BelowZero)
		{
			Assert::AreEqual(Activation::LReLu(-1.0), -0.01);
		}

		TEST_METHOD(Activation_SoftMax_ResultsEqualOne)
		{
			list<double> values = Activation::SoftMax({ 0.33, 0.33, 0.33 });
			list<double>::iterator valuesIt = values.begin();
			double result = 0;
			double expected = 1;
			for(int i = 0; i < values.size(); ++i)
			{
				result += *valuesIt;
				advance(valuesIt, 1);
			}
			Assert::AreEqual(round(result), round(expected));
		}
		#pragma endregion

		#pragma region Loss
		TEST_METHOD(Loss_CrossEntropy)
		{
			double value = Loss::crossEntropy({ 0.5, 0.5 }, { 1, 0 });
			Assert::AreEqual(value, 0.69314718055994529);
		}
		#pragma endregion

		#pragma region Forward_Pass
		TEST_METHOD(Forward_Pass)
		{
			list<double> weights1{ 0.1 };
			list<double> weights2{ 0.2 };

			Neuron* neuron1 = new Neuron(weights1);
			Neuron* neuron2 = new Neuron(weights2);
			Neuron* neuron3 = new Neuron({ 0.3, 0.4 });
			Neuron* neuron4 = new Neuron({ 0.5, 0.6 });

			list<Neuron*> neurons1{ neuron1, neuron2 };
			list<Neuron*> neurons2{ neuron3, neuron4 };

			Layer* layer1 = new Layer(neurons1, "LReLu" );
			Layer* layer2 = new Layer(neurons2, "LReLu" );
			Network* network = new Network();

			network->addLayer(*layer1);
			network->addLayer(*layer2);

			network->predict({ 4 });

			list<double> outputs = layer2->getActivationOutputs();
			list<double>::iterator outputsIt = outputs.begin();

			Assert::AreEqual(ceil((*outputsIt*100))/100, ceil(0.44*100)/100);

			advance(outputsIt, 1);
			Assert::AreEqual(ceil(*outputsIt * 100) / 100, ceil(0.69 * 100) / 100);
		}

		#pragma endregion

		#pragma region Backpropogation
		TEST_METHOD(Backpropogation_2x2Network)
		{
			list<double> weights;
			list<double> weights2;
			Neuron* neuron = new Neuron({0.1, 0.5});
			Neuron* neuron2 = new Neuron({0.2, 0.5});
			Neuron* neuron3 = new Neuron({ 0.5, 0.6, 0.5});
			Neuron* neuron4 = new Neuron({ 0.7, 0.8, 0.5});
			list<Neuron*> neurons = { neuron, neuron2 };
			list<Neuron*> neurons2 = { neuron3, neuron4 };
			Layer* layer = new Layer(neurons, "relu");
			Layer* layer2 = new Layer(neurons2, "relu");
			Network* network = new Network();
			network->addLayer(*layer);
			network->addLayer(*layer2);

			network->train({ 0.4 }, 0.5, { 1, 0 });
			list<double> weightsAfter = neuron->getWeights();
			list<double> weightsAfter2 = neuron2->getWeights();
			list<double> weightsAfter3 = neuron3->getWeights();
			list<double> weightsAfter4 = neuron4->getWeights();
			
		}

		TEST_METHOD(NeuralNetwork_TwoLayers)
		{
			Neuron* neuron1 = new Neuron({ 0.1, 0.2, 0.5 });
			Neuron* neuron2 = new Neuron({ 0.3, 0.4, 0.5 });
			Neuron* neuron3 = new Neuron({ 0.5, 0.6, 0.5 });
			Neuron* neuron4 = new Neuron({ 0.7, 0.8, 0.5 });

			list<Neuron*> neurons1 = { neuron1, neuron2 };
			list<Neuron*> neurons2 = { neuron3, neuron4 };

			Layer* layer1 = new Layer(neurons1, "relu");
			Layer* layer2 = new Layer(neurons2, "relu");

			Network* network = new Network();
			network->addLayer(*layer1);
			network->addLayer(*layer2);

			double averageError = 1;
			while (averageError > 0.2333)
			{
				averageError = 0;
				network->train({ 5, 1 }, 0.001, { 0, 1 });
				averageError += network->getError();
				network->train({ 1, 5 }, 0.001, { 1, 0 });
				averageError += network->getError();
				network->train({ 5, 1 }, 0.001, { 0, 1 });
				averageError += network->getError();
				network->train({ 1, 5 }, 0.001, { 1, 0 });
				averageError = (averageError + network->getError()) / 4;
			}
			network->predict({ 1, 5 });

			list<double> predictions = Activation::SoftMax(network->getPrediction());
			list<double>::iterator predictionsIt = predictions.begin();

			double prediction1 = *predictionsIt;
			advance(predictionsIt, 1);
			double prediction2 = *predictionsIt;
			
			Assert::IsTrue(prediction1 > prediction2);
		}

		TEST_METHOD(NeuralNetwork_ThreeLayers)
		{
			Neuron* neuron1 = new Neuron({ 0.1, 0.2 });
			Neuron* neuron2 = new Neuron({ 0.3, 0.4 });
			Neuron* neuron3 = new Neuron({ 0.5, 0.6 });
			Neuron* neuron4 = new Neuron({ 0.7, 0.8 });
			Neuron* neuron5 = new Neuron({ 0.5, 0.4 });
			Neuron* neuron6 = new Neuron({ 0.3, 0.2 });

			list<Neuron*> neurons1 = { neuron1, neuron2 };
			list<Neuron*> neurons2 = { neuron3, neuron4 };
			list<Neuron*> neurons3 = { neuron5, neuron6 };

			Layer* layer1 = new Layer(neurons1, "relu");
			Layer* layer2 = new Layer(neurons2, "relu");
			Layer* layer3 = new Layer(neurons3, "relu");

			Network* network = new Network();
			network->addLayer(*layer1);
			network->addLayer(*layer2);
			network->addLayer(*layer3);

			double averageError = 1;
			while (averageError > 0.5)
			{
				network->train({ 5, 1 }, 0.25, { 0, 1 });
				averageError += network->getError();
				network->train({ 1, 5 }, 0.25, { 1, 0 });
				averageError += network->getError();
				network->train({ 5, 1 }, 0.25, { 0, 1 });
				averageError += network->getError();
				network->train({ 1, 5 }, 0.25, { 1, 0 });
				averageError = (averageError + network->getError()) / 4;
			}
			network->predict({ 1, 5 });

			list<double> predictions = Activation::SoftMax(network->getPrediction());
			list<double>::iterator predictionsIt = predictions.begin();

			double prediction1 = *predictionsIt;
			advance(predictionsIt, 1);
			double prediction2 = *predictionsIt;

			Assert::IsTrue(prediction1 > prediction2);
		}
		#pragma endregion

	private:

	};
}
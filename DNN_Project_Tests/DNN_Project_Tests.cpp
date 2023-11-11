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

		}
		#pragma endregion


		#pragma region Backpropogation
		TEST_METHOD(Backpropogation_SingularNueron)
		{
			Neuron* neuron = new Neuron({0.5, 1.0});
			list<Neuron*> neurons = { neuron };
			Layer* layer = new Layer( neurons, "relu");
			Network* network = new Network();
			network->addLayer(*layer);
			network->addLayer(*layer);
			network->traverseLayer(0, 0, 1);
		}
		#pragma endregion


	private:

	};
}

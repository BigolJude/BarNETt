by Jude Barnett
# How to use the network:

The entire model is contained within a `Network` class. This class contains the functions to add layers to the system, predict and train the model.

- Start by instantiating a new network class.
	- example `Network* network = new Network();`

- Create **separate** layer classes either by:
	- Creating the individual neurons with pre-set weights.
		- example: `Layer* layer = new Layer(neurons, "relu")`
	- Creating the layer with the previous layer count, the inputs if it's the first layer, and the count of neurons in the current layer.
		- example: `Layer* layer = new Layer(3, 4, "relu");`
		- At current, this has only HE initialisation.
	- Once instantiated add the layers to the network: 
		- example: `network->addLayer(layer);`
	- Layer and neurons classes are required to be individual pointers because use of the same layer or neuron will not create two instances of the class.

- To train the model we call the `Train` function of class `Network` and pass in the: inputs, the learning rate, and expected outcome (a one hot decoded truth vector)
	- example: `Network->Train({0, 5}, 0.5, {1, 0, 0});` this is training on two inputs and will produce three possible predictions with the learning rate of 0.5.

- To predict inputs in the model we call the `Predict` function of class `Network` and pass in just the inputs. 
	- example: `Network->Predict({5, 0});`
# How it works:

### Background:
Being written in C++ the project is naturally object orientated. This means that the Neuron, Layer and Network have all different objects. 

The documentation will only be covering the main functionality of the network and the "boilerplate" code will be left out for the sake of brevity. 

## Non-Static Objects and their explanation

All objects within the network can be initialised granularly, as in, the neuron, layer and network can be initialised with weights, including bias, pre-set. This helps with testing networks as a whole without having randomised initialisations of the weights.
## Neuron

Link: https://github.com/BigolJude/DNN_Project/blob/master/DNN_Project/Neuron.cpp

The neuron in the system contains a list of weights, a total computed weight, and the activated output of the neuron. The neuron's responsibility is to only calculate it's own activation output and train it's own weights given a gradient. 

Being the first part of the network that was designed the training algorithm originally did not take a gradient but the error.  As this was only calculating a single perceptron the "error" in this case would be changed to a gradient calculated outside of the neuron object via backpropagation and the chain rule.
## Layer

Link: https://github.com/BigolJude/DNN_Project/blob/master/DNN_Project/Layer.cpp

The layer simply contains a list of pointers to different neurons. The list needs to be of pointers because of the the backpropagation will be at the network level, this will give the ability to read and write the instances of the neuron objects. 

On the forward pass the layer iterates through the neuron pointers within it's neuron list and calculates the activation of each neuron. The Activation of each neuron is then retrieved and passed back as a list of doubles as the next set of inputs to give to the next layer.
## Network

Link: https://github.com/BigolJude/DNN_Project/blob/master/DNN_Project/Network.cpp

The network, as expected, has the most responsibility over all of it's components. It contains the top level functions for: training (forwards and backward passes), weighing (forward pass) and adding and removing of layers. 

The network is designed to account for an (n) number of layers within an (n) number of neurons. 
#### Forward Pass

The forward pass runs the `train()` function,  iterates through all of the layers, retrieves the list of activations, and places them in the next layer's inputs. This happens until the last layer is reached in which the program will then run a `softmax` activation on the last layer to get the 'predictions' of the inputs given.

The network will then calculate the loss of the function. This is currently fixed at cross entropy loss.
#### Backward Pass

Before traversing back through the network, the derivative of the loss and output activation function is calculated. In the case of Softmax and Cross-Entropy Loss this comes to a simple vector of outputs: the predictions minus the truth vector `ml-dawn (2020)`. This retrieves the error of each individual neuron and places them in a temporary list of results.

Backpropagation is completed via a backwards pre-order network traversal, that is, unlike a traditional pre order traversal in which the values in the tree a visited once; the values in the network traversal are visited for (n) number of neurons in the layer it has previously traversed.

During traversal, as each neuron weight is reached the delta of the weight is calculated depending on the layer type: input, output, or hidden:

-  **For Input layers** the gradient is calculated as the current weight's input multiplied by one minus input, multiplied by the other derivatives of the previously traversed layers.
	-  **Example:** `input(i) * (1 - input(i)) * deltas(1..n)`

- **For output layers** the gradient is simply the output layers current activation multiplied by its, at the same index, derivative of softmax and cross entropy. 
	-  **Example** `activation(i) * error(i)`

- **For the hidden layers** the gradient is calculated as the activation (in the next layer) multiplied by one minus the activation multiplied by derivatives of the neurons within the previous layers.
	- **Example** `activation(i) * (1 - activation(i)) * deltas(1..n) `

`Odikadze, G. (2022, May 16).`

Each gradient and weight calculated by the network are multiplied before being appended to a list of derivatives. These are then used when the algorithm calculates further derivatives of the weight with respect to the error of the final output. In shorter terms this is how the chain rule is implemented to get the derivative of each weight in respect to each error. `Mazur. (2015, March 17)`
### Static Helper Methods:

The **initialisation**, **loss**, **activation**, and **CSV** static classes were created to separate the concerns of the functionality and help readability. Within each of these classes there is a set of functions used in the networks. They can be called individually to help unit test them and give the potential opportunity to add and remove other methods without inherently damaging the methods within the network.
#### Initialisation

Link: https://github.com/BigolJude/DNN_Project/blob/master/DNN_Project/Initialisation.cpp

Initialisation aids in setting the starting points for each weight within the network. Currently the implemented methods are **HE** and **Xavier** initialisation with an additional function `Random` to give a random number between two given double values.

- **He**   
- **Xavier**

Both functions, takes the previous layer neuron count and uses that in the current layers starting weights. Translated from a python implementation `Brownlee, J. (2021, February 2)`
#### Loss

Link: https://github.com/BigolJude/DNN_Project/blob/master/DNN_Project/Loss.cpp

Loss aids in discovering the overall error of a network given certain inputs. Currently implemented method is **Cross-Entropy** .

- **Cross-Entropy** - works well with softmax  as it gives a vector of prediction and compares the vectors with a truth vector.
#### Activation

- Activation contains all of the functions for activating the neurons in the system. At current this is fixed to LReLu and Softmax.

Activations currently implemented are:
- **LReLu** - Implemented as a simple ternary and based in off a paper by `(Maas et al, n.d)`
- **ReLu** - Implemented as a simple ternary.
- **Step** - Implemented as a simple ternary.
- **Softmax** - Function returning a list of probabilities with the sum equating to one. Mainly translated from python `(Brownlee, 2020)`
#### CSV

Link: https://github.com/BigolJude/DNN_Project/blob/master/DNN_Project/CSV.cpp

The CSV class is a helper method to import datasets into the system. Currently this class only has one function and the function is fixed to import one dataset. The Iris dataset by `(SachGarg, n.d.)`

#### Example Networks

Link: https://github.com/BigolJude/DNN_Project/blob/master/DNN_Project/ExampleNetworks.cpp

The `ExampleNetworks` has three seperate examples of how to build networks within the DNN_Project. One of these examples is used for the iris dataset.

# Current Performance and expected improvements.

Functionally, the forward propagation is complete and working as intended. This can be shown through unit tests. In addition each activation, loss, and initialisation also have unit tests and are functioning correctly. 
### Issues found in the network

As for the backpropagation, while functionally and theoretically the algorithm works as intended the performance is lacking. This is could be due to a few factors:

- **The derivatives are being calculated either wrong or inaccurately.** While the unit tests currently covering networks that are of 2 layers, of which have 2 neurons each, are returning values that are expected (or are near to expected due to floating point inaccuracies); with layers 3 or more the accuracy of the backpropagation begins to decline. 

![Pasted image 20231207134733](https://github.com/BigolJude/DNN_Project/assets/74246561/ba908e2d-864c-4c8e-af62-89089cb2ec0e)

In this case a 2 by 2 network has been created with set weights and a forward and backwards pass has been completed (forward for all neurons and backwards for the two top neurons) after passing in the same value given similar values were returned in the backwards pass (similar again because of floating point inaccuracies) 

- **Bias corrections needed**. The github is currently split with `master` and `ai-1` branches. `ai-1` has the bias weights and values enabled and `master` has them disabled, this is due to a problem with exploding gradients and loss not declining. In `ai-1`, if a network of two possible outputs is given the network will always favour the first inputs given. However in `master`, without the bias, the weights seemingly work fine and loss declines to an optimal point. (The optimal point in this case being the natural point when the loss 'levels out'). Until corrections can be made to the bias, the two branches will remain separate. 

 - **Inaccuracies of the model** overall the model, when working, is fairly inaccurate regardless of technical problems. This could be attributed to the lack of optimizations and the only gradient descent being used is stochastic; the network tends to favour the class trained first. Where as batch gradient descent could help in this case.

- **The architecture of the network** the network is a non-vectorised (linear) network. This means on the backpropagation the network calculates the deltas while traversing the network adding or removing the delta calculations from a list one by one. There are draw-backs and benefits to this: the draw backs being that the network will be much slower because it's calculating each value individually rather than using vectors. The benefits is that the networks will theoretically use much less memory, due to not storing the values when they are not in use.

- **Potential Issues with the dataset** the iris dataset `(SachGarg, n.d.)` could potentially be causing a loss of accuracy. The dataset even on established neural networks will only be accurate to a certain point. After using the dataset with Tensor Flow and Keras the max accuracy with the network was only slightly above 44% on initial run with 40 epochs: ![image](https://github.com/BigolJude/DNN_Project/assets/74246561/d726346a-1a87-49b3-b23e-ed45ad0faec1).
It should be noted although the Dataset comes from different places the Scikit-learn and the dataset from `(SachGarg, n.d.)` are the same. 
 
# References

1. Maas, A., Hannun, A., & Ng, A. (n.d.). _Rectifier Nonlinearities Improve Neural Network Acoustic Models_. http://ai.stanford.edu/~amaas/papers/relu_hybrid_icml2013_final.pdf
2. Brownlee, J. (2020, October 18). _Softmax Activation Function with Python_. Machine Learning Mastery. https://machinelearningmastery.com/softmax-activation-function-with-python/
3. SachGarg (n.d.). Www.kaggle.com. Retrieved December 7, 2023, from https://www.kaggle.com/datasets/sachgarg/iris-classification?rvi=1
4. ml-dawn (2020, February 6). _the derivative of softmax function w-r-t z_. https://www.mldawn.com/the-derivative-of-softmaxz-function-w-r-t-z/
5. ml-dawn (2020, May 20).  _Back-propagation with Cross-Entropy and Softmax_.. https://www.mldawn.com/back-propagation-with-cross-entropy-and-softmax/
6. Mazur. (2015, March 17). _A Step by Step Backpropagation Example_. Matt Mazur; Matt Mazur. https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
7. Odikadze, G. (2022, May 16). _Backpropagation with actual numbers_. Medium. https://medium.com/@givi.odikadze/backpropagation-with-actual-numbers-98de37254d1
8. Brownlee, J. (2021, February 2). _Weight Initialization for Deep Learning Neural Networks_. Machine Learning Mastery. https://machinelearningmastery.com/weight-initialization-for-deep-learning-neural-networks/

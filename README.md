# ijl20 JupyterLab source files

generally for experimenting with machine learning in python
## Some keywords

Training data
* The data (e.g. images) used to train a neural network.

Validation data
* The data (e.g. images) kept separate from the training data, used after training to assess the accuracy of the model.

Test data
* Sometimes used to mean 'Validation data' as above, 
* sometimes used to refer to a users 'own' data (e.g. images) that
are used to test the neural network after development using the training data and validation data is completed.

Overfitting
* The behaviour of a neural network as its prediction accuracy for the training data begins to far exceed the
accuracy it is achieving on the separate validation data

Label
* The 'class' or 'category' given in the training and validation data for each input value. If the training and test
values are referred to as 'x' values, the label is often referred to as the corresponding 'y' value.

Inference Pass, also Predict Pass
* This is a 'simple' forward pass through the neural network, with a data value (or set of data values) provided as 
input. For a successfully trained neural network, this will result in the output from the neural network providing a
good classification of the input.

Activation function
* The default behaviour of a neural network layer will be to output some derived matrix resulting from its input. 
Due to compounded additions or multiplications, the range of values in this output may diverge greatly from 
the range of values in the input (e.g. the input may have every value in the range 0..1) which might cause overflow
or inaccuracies if simply propagated to later layers in the network. An activation function (for example ReLU or
sigmoid) can be applied to the output of the layer to 'normalise' the data in some way to avoid the possible
runaway effect of the values increasing (or decreasing) substantially as the data propagates through the network.

ReLU
* An activation function that can be applied to the output of a neural network layer which replaces negative output
values with zero and leaves positive output values unchanged.

Sigmoid
* An activation function that smoothly squashes layer output values into the range -1..+1.

Optimizer
* The algorithm used to find the minimum loss while iterating through the training data, e.g. Stochastic Gradient Descent
or Adam.

(Stochastic) Gradient Descent
* The effect of the backpropagation algorithm that gradually shifts the weights in the neural network in a 
direction that improves its overall accuracy. I.e. the network is slowly descending down some error gradient until
it reaches its minimum error. The use of 'stochastic' refers to the random choice of training data values. This is a
type of optimizer.


# 1 - Introduction
This section will introduce us to neural networks and how they relate to calculus

# 2 - What is a Neural Network
neuron: a thing that holds a number between 0 and 1

Imagine 28x28 grid of pixels that each hold the number representing how dark they are
  - 784 neurons in our network (28*28)
  - hidden layers in between for mathy stuff
  - output layer of 10 neurons that holds the likilihood that our number corresponds to that neuron's representation between 0 and 9

Activations in one layer determine the activations in the next layers
  - brightest neuron in the output layer is the one that the network thinks is the number

Activation of a neuron is a measure of how positive the relevant weighted sum is
  - can add a bias by adding/subtracting to the weight

Weights tell you what pixel pattern the neurons are picking up on

Bias tells you how high the weighted sum needs to be before the neuron starts getting meaningfully active

Our example network for reading handwritten numbers has over 13000 weights and biases

Each neuron is a function
  - takes in the outputs of the neurons in the previous layer and spits out a number between zero and one

# 3 - Gradient Descent
Each neuron is connected to all of the neurons in the previous layer
  - the weights are the strengths of those connections
  - the bias is some indication of whether that neuron tends to be active/inactive

Start by initializing all weights and biases completely randomly
  - does pretty poorly at identifying the handwritten numbers

Design a cost function
  - a way of telling the computer that it did a bad job
  - add up the square of the differences between the value the neurons had - the value you wanted it to have
    - this will be small when the network classifies the image correctly and get much larger when it's really wrong.
    - take average cost over all of the training samples (This is our measure for how lousy the network is)

Need to tell the computer which direction to go to reduce the output of the cost function
  - compute gradient function
  - take small step in the direction that moves you downhill
  - repeat
  * Computing this is called "backpropagation"

Gradient Vector
  - encodes the relative importance of each weight and bias
  - some will be more important to change than others (that's good!)

# 4 - Backpropagation
Core algorithm behind how neural networks learn
  - it computes the gradient that we talked about in the previous lesson

Ways to change the output
  - adjust the bias
  - increase a particular weight in proportion to the value of the neuron it points to
  - change value of neuron in previous layer in proportion to the value of it's weights (can't directly effect this, but, you can change the weights/biases that input into them)

Calculate backwards from the output layer to it's weights to the hidden layer to it's weights to the other hidden layer to its weights to the input layer
  - this is why it's called backpropagation

Go through this for all of the training examples
  - record how they would like to change each weight and bias
  - average the desired changes together
  - the collection of the average to nudge each weight and bias is the negative gradient of the cost function

Technically this should only be done after going through all of your training data
  - this costs a lot of time and computing power
  - people batch their training data into small groups and adjust the weights and biases each step
  * Stochastic gradient descent (looks like a drunk person quickly but haphazardly walking toward the lowest point)

# 5 - Backpropagation and Calculus
GC/Gw(L) = Gz(L)/Gw(L) * Ga(L)/Gz(L) * GC0/Ga(L)

# 6 - Gradient Descent: Example
Gradient applies to functions that have multiple different inputs but a single number as an output
  - e.g. cost function of high dimensional machine learning algorithm
  - imagine that at any pointon the graph there's a vector that points you in the direction of the local minimum (vector's magnitude indicates how steeply you can go in that direction)
Partial Derivative
  - take derivative of the variable you want to look at and assume the rest of the vars are constants
